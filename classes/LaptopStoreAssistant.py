from typing import List

from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.schema import MetadataMode
from llama_index.llms.anthropic import Anthropic
from tqdm import tqdm

from classes.Config import Config
from classes.DocumentProcessor import DocumentProcessor
from classes.EmbeddingModel import EmbeddingModel
from MongoDBClient import MongoDBClient
from QueryEngineSetup import QueryEngineSetup
from VectorStore import VectorStore


class LaptopStoreAssistant:
    """Orchestrates the overall workflow for the e-commerce store assistant."""

    def __init__(self):
        self.config = Config()
        self.mongo_client = MongoDBClient(self.config.MONGODB_URI)
        self.document_processor = DocumentProcessor(self.config.LAPTOP_FIELDS)
        self.embedding_model = EmbeddingModel(
            api_key=self.config.OPENAI_API_KEY,
            model_name=self.config.EMBEDDING_MODEL_NAME,
            dimensions=self.config.EMBEDDING_DIMENSIONS,
            batch_size=self.config.EMBED_BATCH_SIZE,
        )
        self.vector_store = VectorStore(
            mongo_client=self.mongo_client,
            db_name=self.config.DB_NAME,
            collection_name=self.config.COLLECTION_NAME,
        )
        self.query_engine = None  # Initialized when needed

    def embed_and_store_laptops(self, json_file_path: str) -> None:
        """Reads, processes, embeds, and stores laptop data."""
        raw_documents = self.document_processor.read_json(json_file_path)
        processed_documents = self.document_processor.process_documents(raw_documents)
        nodes = self.generate_embedding_nodes(processed_documents)
        self.vector_store.add_nodes(nodes)
        self.mongo_client.create_indexes(
            self.config.DB_NAME, self.config.COLLECTION_NAME
        )

    def generate_embedding_nodes(self, documents: List[Document]) -> List[Document]:
        """Generates embedding nodes from processed documents."""
        semantic_splitter = SemanticSplitterNodeParser(
            buffer_size=10, breakpoint_percentile_threshold=95, embed_model=self.embedding_model.embedding
        )
        nodes = semantic_splitter.get_nodes_from_documents(documents)
        with tqdm(total=len(nodes), desc="Embedding Progress", unit="node") as pbar:
            for node in nodes:
                node.embedding = self.embedding_model.get_embedding(
                    node.get_content(metadata_mode=MetadataMode.EMBED)
                )
                pbar.update(1)
        return nodes

    def setup_query_engine(self, llm: Anthropic) -> None:
        """Sets up the query engine."""
        self.query_engine = QueryEngineSetup(
            mongo_client=self.mongo_client,
            db_name=self.config.DB_NAME,
            collection_name=self.config.COLLECTION_NAME,
            llm=llm,
        )

    def query_laptops(self, user_query: str) -> None:
        """Processes a user query and provides laptop recommendations."""
        if not self.query_engine:
            raise ValueError("Query engine is not set up. Call setup_query_engine() first.")

        prompt = self.construct_prompt(user_query)
        response = self.query_engine.query(prompt)
        print(response)

    def construct_prompt(self, query: str) -> str:
        """Constructs a detailed prompt for the LLM based on the user query."""
        prompt = f"""
You are an AI assistant specializing in laptop recommendations for an e-commerce store. Your task is to process user queries about laptops, search a database using specific criteria, and provide relevant laptop suggestions. You will be working with a QueryEngineTool to search for information in the knowledge base.

Here is the user's query:
<user_query>
{query}
</user_query>

The knowledge base can be searched using the following fields:
<searchable_fields>
{self.config.LAPTOP_FIELDS}
</searchable_fields>

Please follow these steps to process the query and provide a response:

1. **Analyze the Query**:
   - Identify the intent of the query (e.g., sorting, filtering).
   - Determine relevant fields based on the query.
   - Extract specific requirements or preferences.

2. **Formulate Search Criteria**:
   - Based on the analysis, create precise search filters or sorting orders.
   - Explain the reasoning behind each search criterion.

3. **Select Laptops**:
   - Retrieve matching laptops from the knowledge base using the formulated criteria.
   - Ensure each selected laptop has a name and price available.
   - For each selected laptop, assess its relevance to the query and rate it on a scale of 1-10.

4. **Format the Response**:
   - Present the selected laptops in a clear and concise manner using the following structure:

Based on your query, here are the most relevant laptops:

1. [Laptop Name 1] - €[Price 1]
   - [Brief note about relevant features]

2. [Laptop Name 2] - €[Price 2]
   - [Brief note about relevant features]

[Continue for all relevant laptops, up to 5 if applicable]

**Important**:
- Only include factual information from the knowledge base in your response.
- If unsure about any details, omit them rather than making assumptions.

**Example Output Structure**:

Based on your query, here are the most relevant laptops:

1. HP Spectre 17-cs0000nf - €5,999.99
   - This is the most expensive laptop in our database, featuring a high-performance processor and ample storage.

2. Apple MacBook Air 13" 512 Go SSD 24 Go RAM Puce M3 - €1,459.00
   - Offers a larger SSD and more RAM compared to other models, making it suitable for intensive tasks.

3. Apple MacBook Air 13" 256 Go SSD 16 Go RAM Puce M3 - €1,099.00
   - Balanced performance with sufficient RAM and SSD storage.

4. Apple MacBook Air 13" 256 Go SSD 16 Go RAM Puce M2 - €949.00
   - An affordable option with decent specifications for everyday use.
        """
        return prompt
