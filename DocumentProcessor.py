import json
from typing import List, Dict, Any
from llama_index.core import Document
from llama_index.core.schema import MetadataMode

class DocumentProcessor:
    """Handles reading and processing of JSON documents."""

    def __init__(self, fields: List[str]):
        self.fields = fields

    @staticmethod
    def read_json(file_path: str) -> List[Dict[str, Any]]:
        """Reads a JSON file and returns the data."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def handle_price_field(price: str) -> float:
        """Converts price to a float, considering European-style decimal separators."""
        try:
            return float(price.replace(",", "."))
        except ValueError as e:
            raise ValueError(
                f"Invalid format for price field: {price}."
            ) from e

    def process_documents(
            self, raw_documents: List[Dict[str, Any]]
    ) -> List[Document]:
        """
        Processes and converts raw document data into a list of llama_index.Document objects.

        Args:
            raw_documents (List[Dict[str, Any]]): The raw JSON data representing the documents.

        Returns:
            List[Document]: A list of processed documents ready for indexing.
        """
        processed_docs = []
        for doc in raw_documents:
            metadata = {}
            for field in self.fields:
                if field == "price":
                    metadata[field] = self.handle_price_field(doc[field])
                else:
                    # Ensure all other fields are strings
                    metadata[field] = str(doc[field])
            processed_doc = Document(
                text=metadata["description"],
                metadata=metadata,
                excluded_llm_metadata_keys=["_id"],
                excluded_embed_metadata_keys=["_id"],
                metadata_template="{key}=>{value}",
                text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
            )
            processed_docs.append(processed_doc)

        # Optional: Display sample document
        if processed_docs:
            print(
                "\nSample Processed Document Price:",
                processed_docs[0].metadata["price"],
            )
            print(
                "\nThe LLM sees this: \n",
                processed_docs[0].get_content(metadata_mode=MetadataMode.LLM),
            )
            print(
                "\nThe Embedding model sees this: \n",
                processed_docs[0].get_content(metadata_mode=MetadataMode.EMBED),
            )
        return processed_docs
