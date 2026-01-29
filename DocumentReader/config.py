# config.py
import os
from dotenv import load_dotenv
import uuid

# User dictionary with emails as keys, including org and admin
users = {
    "john.doe123@example.com": {"username": "User1", "id": str(uuid.uuid4()), "org": "Org1"},
    "jane.smith456@example.com": {"username": "User2", "id": str(uuid.uuid4()), "org": "Org1"},
    "bob.jones789@example.com": {"username": "User3", "id": str(uuid.uuid4()), "org": "Org2"},
    "alice.brown321@example.com": {"username": "User4", "id": str(uuid.uuid4()), "org": "Org2"},
    "mike.wilson654@example.com": {"username": "User5", "id": str(uuid.uuid4()), "org": "Org2"},
    "admin@example.com": {"username": "Admin", "id": str(uuid.uuid4()), "org": None}  # Admin has no org
}

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = "gpt-3.5-turbo"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

COLLECTION_NAME = "aiml_vector_db"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
CHUNK_STRATEGY = "semantic"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 32