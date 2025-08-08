import os
from qdrant_client.models import Distance, VectorParams
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()
qdrant_client1 = QdrantClient(url="http://localhost:6333")
# qdrant_client1.create_collection(
#     collection_name="aiml_vector_db",
#     vectors_config=VectorParams(size=4, distance=Distance.DOT),
# )

# print(qdrant_client1.get_collections())
print(F"Q1  {qdrant_client1.get_collections()} ")

qdrant_client2 = QdrantClient(
     url="https://133dd0e9-4c69-4623-a94c-743a1b3e7295.us-west-2-0.aws.cloud.qdrant.io:6333", 
     # api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.5Rrfk68Q9hKW6MBHZKY3jpekffWxTJ1VXWMJeansQ2U",
     api_key=os.getenv("QDRANT_CLUSTER")
 )

print(F"Q2  {qdrant_client2.get_collections()} ")
