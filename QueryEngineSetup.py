from llama_index.core import VectorStoreIndex
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

import MongoDBClient
from Config import Config


class QueryEngineSetup:
    """Sets up and manages the query engine."""

    def __init__(
            self,
            mongo_client: MongoDBClient,
            db_name: str,
            collection_name: str,
            llm: Anthropic,
    ):
        self.vector_store = MongoDBAtlasVectorSearch(
            mongo_client=mongo_client.client,
            db_name=db_name,
            collection_name=collection_name,
            index_name=Config.INDEX_NAME,
        )
        self.index = VectorStoreIndex.from_vector_store(self.vector_store)
        self.query_engine_tool = QueryEngineTool(
            query_engine=self.index.as_query_engine(similarity_top_k=5, llm=llm),
            metadata=ToolMetadata(
                name="knowledge_base",
                description="Provides information about laptops and specs. Use plain text as input.",
            ),
        )
        self.agent_worker = FunctionCallingAgentWorker.from_tools(
            [self.query_engine_tool], llm=llm, verbose=True
        )

    def query(self, prompt: str) -> str:
        """Processes a query through the agent and returns the response."""
        response = self.agent_worker.as_agent().chat(prompt)
        return str(response)
