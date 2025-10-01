from src.data_processor import DataProcessor
from src.embedding import EmbeddingManager
from src.vectorstore import VectorStore

if __name__ == '__main__':
    dp = DataProcessor()
    chunks, document = dp.build()
    embd = EmbeddingManager()
    chunks_embedding = embd.embed_texts(chunks)
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, chunks_embedding)
    retriver = vectorstore.get_retriever()
