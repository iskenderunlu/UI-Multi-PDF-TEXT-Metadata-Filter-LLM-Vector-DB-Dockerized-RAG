import streamlit as st
import os
from rag_backend import build_rag

st.set_page_config(page_title="RAG Document Q&A", layout="wide")
st.title("📄 Multi-PDF RAG Document Q&A")

uploaded_files = st.file_uploader(
    "Birden fazla PDF yükle",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("data/uploads", exist_ok=True)

    for file in uploaded_files:
        with open(f"data/uploads/{file.name}", "wb") as f:
            f.write(file.getbuffer())

    st.success("PDF’ler yüklendi")

    if "vectorstore" not in st.session_state:
        with st.spinner("Dokümanlar işleniyor..."):
            st.session_state.vectorstore, st.session_state.llm = build_rag("data/uploads")

    pdf_list = [f.name for f in uploaded_files]
    selected_pdf = st.selectbox("Hangi dokümandan aransın?", ["ALL"] + pdf_list)

    query = st.text_input("Sorunu yaz")

    if query:
        if selected_pdf == "ALL":
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
        else:
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4, "filter": {"source": selected_pdf}}
            )

        from langchain.chains import RetrievalQA

        qa = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            retriever=retriever,
            return_source_documents=True
        )

        result = qa(query)

        st.subheader("🧠 Cevap")
        st.write(result["result"])

        with st.expander("📚 Kaynaklar"):
            for doc in result["source_documents"]:
                st.write(doc.metadata)