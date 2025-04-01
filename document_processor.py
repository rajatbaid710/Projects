import os
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import spacy
import nltk
from typing import List, Dict

# Ensure required models are downloaded (unchanged)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    spacy_nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    spacy_nlp = spacy.load('en_core_web_sm')


class DocumentProcessor:
    def __init__(self,
                 collection_name="aiml_vector_db",
                 qdrant_host="localhost",
                 qdrant_port=6333,
                 chunk_strategy="semantic",
                 chunk_size=800,
                 chunk_overlap=150,
                 embedding_model="text-embedding-3-small",
                 batch_size=32):
        load_dotenv()
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.next_id = 0
        self._create_collection(vector_size=1536, distance=Distance.COSINE)
        self.chunk_strategy = chunk_strategy.lower()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if self.chunk_strategy == "recursive_char":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                add_start_index=True,
                separators=["\n\n", "\n", ".", " ", ""]
            ).split_text
        elif self.chunk_strategy == "semantic":
            self.splitter = lambda text: self._semantic_split(text)
        else:
            raise ValueError(f"Unsupported chunking strategy: {chunk_strategy}")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.batch_size = batch_size

    def _create_collection(self, vector_size: int, distance=Distance.COSINE):
        collections = self.client.get_collections().collections
        if not any(col.name == self.collection_name for col in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance)
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")

    def _semantic_split(self, text: str) -> List[str]:
        doc = spacy_nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        for sent in doc.sents:
            if current_length + len(sent.text) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            current_chunk.append(sent.text)
            current_length += len(sent.text)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def process_document(self, json_data: Dict, source_file: str) -> None:
        markdown_content = json_data["content"]["markdown"]
        metadata = json_data["metadata"]
        chunks = self.splitter(markdown_content)
        if not chunks:
            print(f"No chunks generated for {source_file}")
            return

        chunked_data = []
        texts = []
        for i, chunk in enumerate(chunks):
            chunk_info = {
                "chunk_id": i,
                "text": chunk,
                "source_file": source_file,
                "metadata": {
                    **metadata,
                    "chunk_start_index": i * (
                                self.chunk_size - self.chunk_overlap) if self.chunk_strategy != "recursive_char" else getattr(
                        chunk, 'start_index', 0),
                    "chunk_length": len(chunk)
                }
            }
            chunked_data.append(chunk_info)
            texts.append(chunk)

        points = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            embeddings = self.embeddings.embed_documents(batch_texts)
            for j, embedding in enumerate(embeddings):
                chunk_data = chunked_data[i + j]
                point_id = self.next_id
                self.next_id += 1
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk_data["text"],
                            "source_file": chunk_data["source_file"],
                            "original_chunk_id": chunk_data["chunk_id"],
                            "metadata": chunk_data["metadata"]
                        }
                    )
                )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            print(f"Uploaded {len(points)} points from {source_file} to {self.collection_name}")

    def delete_by_source_file(self, source_file: str) -> int:
        """Delete all points in Qdrant associated with the given source file."""
        try:
            # Create a filter to match points by source_file
            filter_query = Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            )

            # Delete points matching the filter
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_query
            )

            deleted_count = self.client.count(
                collection_name=self.collection_name,
                exact=False,
                count_filter=filter_query
            ).count  # This will be 0 if deletion succeeds, but we use it to confirm
            print(f"Deleted embeddings for {source_file} from {self.collection_name}")
            return deleted_count
        except Exception as e:
            print(f"Error deleting embeddings for {source_file}: {str(e)}")
            return -1

    def search(self, query_vector: List[float], limit: int = 5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True
        )
        return results