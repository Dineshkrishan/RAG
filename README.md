# 📄 RAG Chatbot

A Retrieval-Augmented Generation (RAG) based PDF chatbot built with LangChain, 
FAISS, and Streamlit. Upload any PDF and ask questions about it using AI.

## 🚀 Demo
> Upload a PDF → Ask a question → Get AI-powered answers instantly

## 🛠️ Tech Stack
- **LangChain** — RAG pipeline and LLM chaining
- **FAISS** — Vector store for semantic search
- **HuggingFace Embeddings** — all-MiniLM-L6-v2 model
- **OpenRouter API** — LLM inference (Mistral 7B)
- **Streamlit** — Frontend UI
- **PyPDF** — PDF text extraction

## ⚙️ Installation

1. Clone the repository
   git clone https://github.com/yourusername/rag-chatbot.git
   cd rag-chatbot

2. Create virtual environment
   python -m venv .venv
   .venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your API key
   Create a .env file in the root folder:
   OPENAI_API_KEY=your_openrouter_api_key_here

5. Run the app
   streamlit run app.py

## 📋 Features
- Upload any PDF (up to 200MB)
- Automatic text chunking and embedding
- Semantic search using FAISS vector store
- AI-generated answers using Mistral 7B via OpenRouter
- Clean and responsive Streamlit UI

## 🔑 Getting API Key
1. Sign up at openrouter.ai
2. Go to API Keys → Create new key
3. Paste it in your .env file

## 📁 Project Structure
rag-chatbot/
├── app.py              # Main application
├── requirements.txt    # Dependencies
├── .gitignore         # Git ignore rules
└── README.md          # Project documentation

## 👨‍💻 Author
G. Dinesh Krishan
- LinkedIn: your-linkedin-url
- GitHub: your-github-url
