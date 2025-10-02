import os 
import numpy as np 
from typing import List
from pathlib import Path
from src.constant import BASE_DIR
import chromadb
from langchain.vectorstores import Chroma
from langchain.schema import Document
from uuid import uuid4

DATA_DIR = os.path.join(BASE_DIR, "data", "db")


class VectorStore:
    """
    Wrapper around Chroma vector database for persistent storage
    and retrieval of document embeddings.
    """

    def __init__(self,
                 collection_name: str = "medrag",
                 persist_directory: str = DATA_DIR):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize Chroma client and collection."""
        try:
            dir_path = Path(self.persist_directory)
            dir_path.mkdir(parents=True, exist_ok=True)

            self.client = chromadb.PersistentClient(self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAG collection for biomedical research"}
            )
            print(f"Store initialized successfully: {self.collection_name}")
        except Exception as e:
            print(f"Error initializing the store: {e}")
            raise

    def get_len(self) -> int:
        """Return number of documents in the collection."""
        return self.collection.count()

    def add_documents(self, documents: List[Document], embeddings: np.ndarray, batch_size: int = 5000):
        """
        Add documents and their embeddings to the vector store in batches.
        """
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()  # Ensure compatibility

        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start:start + batch_size]
            batch_embeds = embeddings[start:start + batch_size]

            ids, metadatas, texts, embeds = [], [], [], []

            for idx, (doc, emb) in enumerate(zip(batch_docs, batch_embeds)):
                ids.append(f"doc_{uuid4().hex}")
                texts.append(doc.page_content)
                metadata = dict(doc.metadata) if getattr(doc, "metadata", None) else {}
                metadata.update({"doc_index": idx, "content_length": len(doc.page_content)})
                metadatas.append(metadata)
                embeds.append(emb)

            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeds,
                metadatas=metadatas
            )

        print(f"Documents and embeddings added to collection: {self.collection_name}")

    def get_retriever(self, embedding_function, search_kwargs: dict = None):
        """
        Return a retriever interface for semantic search.
        """
        if search_kwargs is None:
            search_kwargs = {"k": 5}

        vectorstore = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=embedding_function
        )
        return vectorstore.as_retriever(search_kwargs=search_kwargs)