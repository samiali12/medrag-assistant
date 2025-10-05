from typing import List
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, model_name: str = "pritamdeka/S-BioBERT-snli-multinli-stsb"):
        self.model_name = model_name
        self.model = None
        self.load_model()

    def load_model(self):
        print("Loading embedding model:", self.model_name)
        self.model = HuggingFaceEmbeddings(model_name=self.model_name)
        print("Model loaded.")

    def get_model(self):
        return self.model

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            emb = self.model.embed_documents(batch)
            embeddings.extend(emb)

        return np.array(embeddings)

    def embed_one(self, text: str) -> np.ndarray:
        return self.model.embed_query(text)