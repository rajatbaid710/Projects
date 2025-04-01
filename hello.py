from qdrant_client.http.models import Filter, FieldCondition, MatchAny, MatchValue
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# Initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Generate embedding for query
query = "crossroads"
query_embedding = embeddings.embed_query(query)

# Define filter for matching source_file
keywords = ["uber.pdf"]
query_filter = Filter(
    must=[
        FieldCondition(
            key="source_file",
            match=MatchAny(any=keywords)
        )
    ]
)
source_file = "uber.pdf"

filter_query = Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
        )
# Initialize Qdrant client
client = QdrantClient(host="localhost", port=6333)
collection_name = "aiml_vector_db"
# Query Qdrant
response = client.query_points(
    collection_name=collection_name,
    # query=query_embedding,  # Use 'query' instead of 'query_vector'
    # limit=5,
    with_payload=True,
    query_filter=filter_query,  # Correct argument name

)

# Print response
# print(response.model_dump_json())

# Iterate through response points
for point in response.points:
    print(f"Score: {point.score}, Summary: {point.id}")

pre_deleted_count = client.count(
            collection_name=collection_name,
            exact=False,
            count_filter=filter_query
        ).count


def delete_by_source_file(self, source_file: str) -> int:
    """Delete all points in Qdrant associated with the given source file."""
    try:
        # Create a filter to match points by source_file
        filter_query = Filter(
            must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
        )

        pre_deleted_count = self.client.count(
            collection_name=self.collection_name,
            exact=False,
            query_filter=filter_query
        ).count
        # Delete points matching the filter
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=filter_query
        )

        deleted_count = self.client.count(
            collection_name=self.collection_name,
            exact=False,
            query_filter=filter_query
        ).count  # This will be 0 if deletion succeeds, but we use it to confirm
        print(f"Deleted embeddings for {source_file} from {self.collection_name}")
        return deleted_count
    except Exception as e:
        print(f"Error deleting embeddings for {source_file}: {str(e)}")
        return -1


# delete_by_source_file("uber.pdf")
