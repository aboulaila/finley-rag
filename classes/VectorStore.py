from typing import List

from llama_index.core import Document
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from classes.Config import Config
from classes import MongoDBClient


class VectorStore:
    """Manages the storage and retrieval of embedding nodes in MongoDB."""

    def __init__(
            self, mongo_client: MongoDBClient, db_name: str, collection_name: str
    ):
        self.vector_store = MongoDBAtlasVectorSearch(
            mongo_client=mongo_client.client,
            db_name=db_name,
            collection_name=collection_name,
            index_name=Config.INDEX_NAME,
        )

    def add_nodes(self, nodes: List[Document]) -> None:
        """Adds embedding nodes to the vector store."""
        self.vector_store.add(nodes)
        print("Nodes successfully stored in MongoDB.")
