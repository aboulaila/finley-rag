from llama_index.llms.anthropic import Anthropic

from classes.LaptopStoreAssistant import LaptopStoreAssistant


def main():
    """Main function to run the Laptop Store Assistant."""
    # Initialize the assistant
    assistant = LaptopStoreAssistant()

    # Uncomment the following line to embed and store laptops initially
    assistant.embed_and_store_laptops("laptops.json")

    # Set up the LLM (Anthropic) with the desired model
    llm = Anthropic(model="claude-3-5-sonnet-20240620")

    # Set up the query engine
    assistant.setup_query_engine(llm)

    # Example query
    user_query = "give me 5 laptops in 2000 euros range"
    assistant.query_laptops(user_query)


if __name__ == "__main__":
    main()
