from src.data_processor import DataProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore
from src.download_data import download_pmc_docs

if __name__ == '__main__':
    flag = True # download_pmc_docs()
    if flag:
        dp = DataProcessor()
        chunks, document = dp.build()
        chunks_list = [c.page_content for c in chunks]
        embd = EmbeddingManager()
        embd_model = embd.get_model()
        chunks_embedding = embd.embed_texts(chunks_list)
        vectorstore = VectorStore()
        vectorstore.add_documents(chunks, chunks_embedding)
        retriver = vectorstore.get_retriever(embd_model)
    
