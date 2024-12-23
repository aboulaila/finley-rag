import os
from dotenv import load_dotenv


class Config:
    load_dotenv()
    """Configuration constants and settings."""
    LAPTOP_FIELDS = [
        "name",
        "description",
        "processor_model",
        "processor_brand",
        "memory_cache_of_processor",
        "processor_frequency",
        "screen_size",
        "ram",
        "graphic_card_brand",
        "graphic_card_model",
        "video_dedicated_memory",
        "video_memory_type",
        "storage",
        "storage_disk_type",
        "graphic_card_resolution",
        "touch_screen",
        "ram_type",
        "wireless_communication",
        "color",
        "gamer",
        "sound",
        "keyboard_type",
        "battery_type",
        "height",
        "width",
        "depth",
        "mass",
        "price",
        "link",
        "image_url",
    ]
    DB_NAME = "fnly"
    COLLECTION_NAME = "laptops"
    MONGODB_URI = os.getenv("MONGODB_URI")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS = 1536
    EMBED_BATCH_SIZE = 10
    INDEX_NAME = "vector_index"
