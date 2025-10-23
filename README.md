# DocSphere 🚀

[![Streamlit App](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?&style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/) [![LangChain](https://img.shields.io/badge/LangChain-%235B21B6.svg?&style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/) [![Python](https://img.shields.io/badge/Python-3776AB.svg?&style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)

<div align="center">                                                                                                                                                                                                                                                                                                                                                                                                                          
  
**Upload documents, chat with them effortlessly!**  
A sleek RAG-powered Streamlit app that turns your PDFs, websites, texts, PPTs, and Excels into interactive Q&A sessions. Powered by Groq's lightning-fast inference and Hugging Face embeddings. 🌌✨


## 🌟 Features

- **Multi-Format Support** 📄: Upload PDFs, text files, PowerPoints, Excels, or scrape websites directly.
- **Smart Chunking & Embeddings** 🧠: Uses RecursiveCharacterTextSplitter for precise text splitting and BGE embeddings for semantic search.
- **RAG Magic** ⚡: Retrieval-Augmented Generation with FAISS vector store – get context-aware answers from Groq's Llama-3.1-8B.
- **Nebula UI Theme** 🎨: Cosmic gradients, glassmorphism, and futuristic vibes for an out-of-this-world experience.
- **Real-Time Chat** 💬: Ask questions in a conversational interface; previews extracted content in the sidebar.
- **Session Persistence** 🔄: Keeps your vector DB alive across interactions without re-processing.
- **Copy-to-Clipboard** 📋: One-click copy for AI responses – share insights easily!

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Framework** | Streamlit |
| **LLM** | Groq (Llama-3.1-8B-Instant) |
| **Embeddings** | HuggingFace BGE (BAAI/bge-base-en-v1.5) |
| **Vector Store** | FAISS |
| **Document Loaders** | LangChain (PyPDF, WebBase, Text, Unstructured for PPT/Excel) |
| **Text Splitting** | RecursiveCharacterTextSplitter |
| **Environment** | Python 3.10+, dotenv for API keys |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/keys) (free tier available!)

### Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Jagadish110/Docsphere.git
   cd docsphere
   ```
   

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
  

3. Set up your environment:
   - Create a `.env` file in the root:
     ```
     groq_api_key=your_groq_api_key_here
     ```
   - Load it with `load_dotenv()` (already in the code!).

### Run the App
```bash
streamlit run Chatbot.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser. Boom! 🚀

---

## 📖 Usage Guide

1. **Select Source Type** 📂: Choose from PDF, Website, Text, PPT, or Excel via the dropdown.
2. **Upload/Input** 📥: Drop your file or paste a URL.
3. **Preview & Process** 👀: Sidebar shows extracted text (limited preview for perf). It auto-builds the vector DB.
4. **Chat Away** 🗣️: Type your question in the input box. Get answers grounded in your doc!
   - Example: For a PDF resume, ask "What skills does the candidate have?"
5. **Copy Responses** 📋: Hit the "Copy" button next to any AI answer for quick sharing.
6. **Pro Tip** 💡: Only ask doc-related questions – the app enforces context!

### Example Workflow
- Upload a research paper PDF → Ask "Summarize the methodology section."
- Paste a blog URL → Query "Key takeaways on AI ethics?"

---
## DEMO:
- https://huggingface.co/spaces/jagadishwar/DocSphere
