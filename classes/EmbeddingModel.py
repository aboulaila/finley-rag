from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.settings import Settings
from typing import List

class EmbeddingModel:
    """Configures and manages the embedding models."""

    def __init__(self, api_key: str, model_name: str, dimensions: int, batch_size: int):
        self.embedding = OpenAIEmbedding(
            model=model_name,
            dimensions=dimensions,
            embed_batch_size=batch_size,
            openai_api_key=api_key,
        )
        Settings.embed_model = self.embedding

    def get_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text."""
        return self.embedding.get_text_embedding(text)
