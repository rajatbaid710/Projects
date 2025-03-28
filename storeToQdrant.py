import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


class QdrantUploader:
    def __init__(self, collection_name="aiml_vector_db", qdrant_host="localhost", qdrant_port=6333):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.next_id = 0  # Counter for generating unique integer IDs

    def create_collection(self, vector_size, distance=Distance.COSINE):
        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

    def upload_embeddings(self, input_dir):
        if not os.path.exists(input_dir):
            print(f"Input directory {input_dir} does not exist")
            return

        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        if not json_files:
            print(f"No embedded JSON files found in {input_dir}")
            return

        total_points = 0

        for json_file in json_files:
            file_path = os.path.join(input_dir, json_file)
            print(f"Processing {json_file}...")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Handle nested structure from DocumentEmbedder or flat list
                embedded_chunks = data.get("chunks", []) if isinstance(data, dict) else data

                points = []
                for chunk in embedded_chunks:
                    # Look for "text" or "content" key
                    text = chunk.get("text") or chunk.get("content")
                    embedding = chunk.get("embedding")

                    if not embedding or not text:
                        print(f"Skipping chunk {self.next_id} in {json_file}: missing embedding or text/content")
                        self.next_id += 1  # Still increment to keep IDs unique
                        continue

                    payload = {
                        "text": text,
                        "source_file": chunk.get("source_file", json_file),
                        "original_chunk_id": chunk.get("chunk_id", self.next_id)
                        # Preserve original chunk ID if present
                    }
                    point_id = self.next_id  # Use integer ID

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=embedding,
                            payload=payload
                        )
                    )
                    self.next_id += 1  # Increment ID for the next point

                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    total_points += len(points)
                    print(f"Uploaded {len(points)} points from {json_file}")
                else:
                    print(f"No valid points to upload from {json_file}")

            except Exception as e:
                print(f"Error processing {json_file}: {str(e)}")

        print(f"\nCompleted: Uploaded a total of {total_points} points to {self.collection_name}")

    def search(self, query_vector, limit=5):
        """Search Qdrant for the top similar vectors."""
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return results


# def main():
#     embedded_docs_dir = "embedded_docs"
#     collection_name = "aiml_vector_db"
#     vector_size = 1536
#
#     uploader = QdrantUploader(
#         collection_name=collection_name,
#         qdrant_host="localhost",
#         qdrant_port=6333
#     )
#
#     uploader.create_collection(vector_size=vector_size, distance=Distance.COSINE)
#     uploader.upload_embeddings(embedded_docs_dir)
#
#
# if __name__ == "__main__":
#     main()
