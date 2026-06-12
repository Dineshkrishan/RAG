import streamlit as st
import os
import urllib.request
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# OpenRouter API key is read from .env (OPENAI_API_KEY=sk-or-v1-...)
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("📄 RAG Chatbot")
st.subheader("Upload a PDF and ask questions!")

# Helper to fetch active free models from OpenRouter dynamically
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_free_models():
    try:
        url = "https://openrouter.ai/api/v1/models"
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            models_data = data.get("data", [])
            free_models = []
            for m in models_data:
                model_id = m.get("id", "")
                pricing = m.get("pricing", {})
                
                is_free = False
                try:
                    prompt_price = float(pricing.get("prompt", 1))
                    completion_price = float(pricing.get("completion", 1))
                    if prompt_price == 0.0 and completion_price == 0.0:
                        is_free = True
                except ValueError:
                    pass
                
                if model_id.endswith(":free"):
                    is_free = True
                    
                if is_free:
                    free_models.append(model_id)
            
            if free_models:
                free_models.sort()
                return free_models
    except Exception:
        pass
    
    # Fallbacks if API call fails
    return [
        "meta-llama/llama-3.3-70b-instruct:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "openrouter/free"
    ]

# Sidebar Configuration
st.sidebar.header("⚙️ Configuration")

# Get API key from env, check if it's empty or placeholder
env_key = os.getenv("OPENAI_API_KEY", "").strip()

if not env_key or env_key == "YOUR_OPENROUTER_KEY_HERE":
    api_key_default = ""
else:
    api_key_default = env_key

api_key = st.sidebar.text_input(
    "OpenRouter API Key",
    value=api_key_default,
    type="password",
    help="Get your API key from openrouter.ai. If already set in .env, it will be populated here automatically."
)

# Fetch dynamic free models list
free_models = get_free_models()

# Default selection index logic
default_model = "meta-llama/llama-3.3-70b-instruct:free"
default_index = 0
if default_model in free_models:
    default_index = free_models.index(default_model)

# Model selection
model_option = st.sidebar.selectbox(
    "Select LLM Model",
    options=free_models,
    index=default_index,
    help="Select the free LLM you want to use from OpenRouter."
)

# Custom model option
custom_model = st.sidebar.text_input(
    "Or Enter Custom Model ID (optional)",
    value="",
    help="e.g., mistralai/mistral-7b-instruct:free"
)

selected_model = custom_model.strip() if custom_model.strip() else model_option

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    # Ensure docs directory exists
    os.makedirs("docs", exist_ok=True)
    with open("docs/uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and split
    loader = PyPDFLoader("docs/uploaded.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # Free local embeddings - no API key needed
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    st.success(f"✅ PDF loaded! {len(chunks)} chunks created.")

    # Ask question
    question = st.text_input("Ask a question about your PDF:")

    if question:
        if not api_key:
            st.error("🔑 Please provide an OpenRouter API key in the sidebar configuration to run queries.")
        else:
            # Determine models to try (primary + fallback list)
            models_to_try = [selected_model]
            possible_fallbacks = [
                "meta-llama/llama-3.2-3b-instruct:free",
                "openrouter/free",
                "google/gemma-2-9b-it:free"
            ]
            for fb in possible_fallbacks:
                if fb not in models_to_try:
                    models_to_try.append(fb)
            
            success = False
            last_error = None
            
            # Container for status updates
            status_container = st.empty()
            
            with st.spinner("Thinking..."):
                for idx, model_name in enumerate(models_to_try):
                    if idx > 0:
                        status_container.info(f"⚠️ Primary model was rate-limited or failed. Trying fallback: `{model_name}`...")
                    
                    try:
                        llm = ChatOpenAI(
                            model=model_name,
                            openai_api_base="https://openrouter.ai/api/v1",
                            openai_api_key=api_key,
                            max_retries=1
                        )
                        qa_chain = RetrievalQA.from_chain_type(
                            llm=llm,
                            retriever=vectorstore.as_retriever()
                        )
                        
                        answer = qa_chain.invoke({"query": question})["result"]
                        
                        # Clear status messages if fallback succeeded
                        status_container.empty()
                        
                        st.markdown(f"**Answer:** {answer}")
                        
                        if idx > 0:
                            st.caption(f"ℹ️ Note: This response was generated using fallback model `{model_name}` due to rate limits on `{selected_model}`.")
                        
                        success = True
                        break
                    except Exception as e:
                        last_error = e
                        err_msg = str(e)
                        # Stop if it's an API key issue or authentication error
                        if "API key" in err_msg or "401" in err_msg:
                            status_container.empty()
                            st.error(f"🔑 Invalid API Key or authentication error: {err_msg}")
                            success = True  # Break outer loop/handling
                            break
            
            if not success:
                status_container.empty()
                st.error(f"❌ Error invoking model: {str(last_error)}")
                st.info("💡 Tip: Try choosing a different model in the sidebar, or make sure your OpenRouter API key has sufficient quota.")