import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

st.title("🧠 RAG App")

# 1. Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"{uploaded_file.name} uploaded!")

    # 2. Load & Split PDF
    with st.spinner("Processing PDF..."):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(docs)

    # 3. Embed with Sentence Transformers
    with st.spinner("Embedding text..."):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local("faiss_index")

    # 4. Load CPU-friendly LLM (Flan-T5)
    with st.spinner("Loading FLAN-T5 model..."):
        model_name = "google/flan-t5-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)
        llm = HuggingFacePipeline(pipeline=pipe)

    # 5. Build RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # 6. Ask a Question
    st.subheader("Ask a question about the document")
    query = st.text_input("Enter your question")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            st.markdown("### Answer")
            st.write(result["result"])

            st.markdown("### Sources")
            for i, doc in enumerate(result["source_documents"]):
                page = doc.metadata.get("page", "Unknown")
                st.markdown(f"**Source {i+1}** (Page {page})")
                st.write(doc.page_content[:300] + "...")
