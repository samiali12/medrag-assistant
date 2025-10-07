from src.data_processor import DataProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.download_data import download_pmc_docs
from src.llm import LLM
from tqdm import tqdm

import streamlit as st 

@st.cache_resource(show_spinner="🔄 Building pipeline...")
def load_pipeline():
    limit = 2000
    download_pmc_docs(limit=limit)
    dp = DataProcessor()
    chunks, document = dp.build()
    chunks_list = [c.page_content for c in tqdm(chunks, desc='Chunking')]
    embd = EmbeddingManager()
    embd_model = embd.get_model()
    chunks_embedding = embd.embed_texts(chunks_list)
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, chunks_embedding)
    retriever = vectorstore.get_retriever(embd_model)
    llm = LLM(retriever)
    return llm


if __name__ == '__main__':

    st.set_page_config(
        page_title="MedRAG: AI-Powered Biomedical Paper Search",
        layout="wide",
        page_icon="🧬",
    )

    st.title("🧬 MedRAG")
    st.caption("Ask questions. Explore research. Ground answers in biomedical literature.")

    llm = load_pipeline()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)

    if query := st.chat_input("Type your biomedical question here..."):
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking... please wait"):
                result = llm.invoke(query)
                answer = result["result"]

                sources = []

                if result['source_documents']:
                    for doc in result['source_documents']:
                        preview = doc.page_content[:200].replace("\n", " ")
                        sources.append(preview + "...")
                st.write(answer)

                if sources:
                    with st.expander('📚 Sources'):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(f"**{idx}.** {src}")

        st.session_state.chat_history.append((query, (answer, sources)))
