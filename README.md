# UI-Multi-PDF-TEXT-Metadata-Filter-LLM-Vector-DB-Dockerized-RAG

It is a RAG (Retrieval Augmented Generation) Project.

What does it do ?

* It takes multiple PDF/Markdown/TXT documents on a web UI by providing a metadata filter that you select which files you apply RAG.
* It makes Embedding and stores it on a Vector Database.
* It generates answers from the uploaded documents.

Project Setting Up

1.1 Python Setup

python -m venv rag-env
source rag-env/bin/activate # macOS / Linux
rag-env\Scripts\activate # Windows

1.2 Required Libraries

pip install langchain langchain-community
pip install sentence-transformers
pip install faiss-cpu
pip install pypdf
pip install ollama

Note: When GPU does not exist on your computer, faiss-cpu will be enough.

1.3 Local LLM Setup (Ollama)

Go to <https://ollama.com> and download and install on your computer then run "ollama pull mistral" command to download the model mistral.

Note: Alternative Models:

* llama3
* phi-3

How to run the Project ?

Firs run "docker compose up --build" command and go to the <http://localhost:8501> on your browser.


Important Notes about the Project

* I reduced the hallucination problem owing to the RAG Arhitecture.
* I made a local semantic search with FAISS.
* I optimized the retrieval quality with Chunking & embedding strategies.
* Local LLM (Ollama) provided data privacy.


