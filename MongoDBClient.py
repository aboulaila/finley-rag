import pymongo

class MongoDBClient:
    """Manages MongoDB connections and operations."""

    def __init__(self, uri: str):
        self.client = pymongo.MongoClient(uri, appname="devrel.showcase.python")
        self._validate_connection()

    def _validate_connection(self):
        try:
            ok = 1.0
            if self.client.admin.command("ping").get("ok") != ok:
                raise ConnectionError("Failed to connect to MongoDB.")
            print("Connection to MongoDB successful")
        except Exception as e:
            raise ConnectionError(f"MongoDB connection failed: {e}")

    def get_collection(self, db_name: str, collection_name: str):
        return self.client.get_database(db_name).get_collection(collection_name)

    def create_indexes(self, db_name: str, collection_name: str):
        collection = self.get_collection(db_name, collection_name)
        # Create descending index on price for efficient sorting
        collection.create_index([("price", pymongo.DESCENDING)], name="price_desc_idx")
        print("Created index on 'price' field.")

        # Example of a compound index on price and ram
        #collection.create_index(
        #    [("price", pymongo.DESCENDING), ("ram", pymongo.ASCENDING)],
        #    name="price_ram_idx",
        #)
        #print("Created compound index on 'price' and 'ram' fields.")
