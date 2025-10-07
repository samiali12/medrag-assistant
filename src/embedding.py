import numpy as np
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class EmbeddingManager:
    def __init__(self, model_name: str = "pritamdeka/S-BioBERT-snli-multinli-stsb"):
        self.model_name = model_name
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.load_model()

    def load_model(self):
        print("Loading embedding model:", self.model_name)
        print('Using device', self.device)
        self.model = SentenceTransformer(model_name_or_path=self.model_name, device=self.device)
        print("Model loaded.")

    def get_model(self):
        return self.model

    def embed_texts(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(batch, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            embeddings.extend(emb)
        return np.vstack(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        return self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True).flatten()