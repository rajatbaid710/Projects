# document_processor.py
import spacy
import nltk
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from typing import List
import time
import hashlib
from config import COLLECTION_NAME, QDRANT_HOST, QDRANT_PORT, CHUNK_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_EMBEDDING_MODEL, BATCH_SIZE, users

# Ensure required models are downloaded
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
                 collection_name=COLLECTION_NAME,
                 qdrant_host=QDRANT_HOST,
                 qdrant_port=QDRANT_PORT,
                 chunk_strategy=CHUNK_STRATEGY,
                 chunk_size=CHUNK_SIZE,
                 chunk_overlap=CHUNK_OVERLAP,
                 embedding_model=OPENAI_EMBEDDING_MODEL,
                 batch_size=BATCH_SIZE):
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
                separators=["\n\n", "\n", ".", " ", ""]
            ).split_text
        elif self.chunk_strategy == "semantic":
            self.splitter = lambda text: self._semantic_split(text)
        else:
            raise ValueError(f"Unsupported chunking strategy: {chunk_strategy}")
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
        )
        self.batch_size = batch_size
        self.processed_files = self._fetch_processed_files()

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

    def _fetch_processed_files(self) -> dict:
        try:
            processed_files = {}
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                scroll_filter=None
            )
            points, next_offset = scroll_result

            while points:
                for point in points:
                    source_file = point.payload.get("source_file")
                    user_email = point.payload.get("user_email")
                    if source_file and user_email:
                        processed_files[source_file] = user_email

                if next_offset is None:
                    break

                scroll_result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=next_offset,
                    with_payload=True,
                    scroll_filter=None
                )
                points, next_offset = scroll_result

            print(f"Fetched {len(processed_files)} processed files from Qdrant: {processed_files}")
            return processed_files
        except Exception as e:
            print(f"Error fetching processed files from Qdrant: {str(e)}")
            return {}

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

    def process_document(self, text: str, source_file: str, user_email: str) -> str:
        doc_size = len(text)
        print(f"Document size for '{source_file}': {doc_size} characters")

        content_hash = hashlib.sha256(f"{text}{user_email}".encode()).hexdigest()
        org = users[user_email]["org"] if user_email != "admin@example.com" else None

        existing_points = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="user_email", match=MatchValue(value=user_email)),
                    FieldCondition(key="content_hash", match=MatchValue(value=content_hash))
                ]
            ),
            limit=1,
            with_payload=True
        )[0]

        if existing_points:
            existing_file = existing_points[0].payload.get("source_file", "unknown file")
            print(f"Duplicate content detected for '{source_file}' by {user_email}. Found in '{existing_file}'.")
            return f"A similar document already exists as '{existing_file}' for {user_email}. Upload skipped."

        start_time = time.time()
        chunks = self.splitter(text)
        chunk_time = time.time() - start_time
        print(f"Time taken to chunk '{source_file}': {chunk_time:.2f} seconds")

        if not chunks:
            print(f"No chunks generated for {source_file}")
            return f"No chunks generated for '{source_file}'. Upload failed."

        points = []
        start_time = time.time()
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            embeddings = self.embeddings.embed_documents(batch_chunks)
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                point_id = self.next_id
                self.next_id += 1
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source_file": source_file,
                            "user_email": user_email,
                            "content_hash": content_hash,
                            "org": org  # Add organization to payload
                        }
                    )
                )
        embed_time = time.time() - start_time
        print(f"Time taken to embed '{source_file}': {embed_time:.2f} seconds")

        if points:
            start_time = time.time()
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            upload_time = time.time() - start_time
            print(f"Time taken to upload '{source_file}' to DB: {upload_time:.2f} seconds")

            self.processed_files[source_file] = user_email
            print(f"Uploaded {len(points)} points from {source_file} by user {user_email} to {self.collection_name}")
            return f"Successfully processed '{source_file}' by {user_email} and uploaded to Qdrant."

        return f"Unexpected error processing '{source_file}'. No points generated."

    def delete_by_source_file(self, source_file: str) -> int:
        try:
            filter_query = Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))]
            )
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=filter_query
            )
            deleted_count = self.client.count(
                collection_name=self.collection_name,
                exact=False,
                count_filter=filter_query
            ).count
            if source_file in self.processed_files:
                del self.processed_files[source_file]
            print(f"Deleted embeddings for {source_file} from {self.collection_name}")
            return deleted_count
        except Exception as e:
            print(f"Error deleting embeddings for {source_file}: {str(e)}")
            return -1

    def search(self, query_vector: List[float], user_email: str, limit: int = 5):
        if user_email == "admin@example.com":
            # Admin can search all documents, no filter applied
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True
            )
        else:
            # Non-admin users search within their organization
            org = users[user_email]["org"]
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(must=[FieldCondition(key="org", match=MatchValue(value=org))]),
                limit=limit,
                with_payload=True
            )
        return results.points

    def get_processed_files(self, user_email: str = None):
        if user_email == "admin@example.com":
            return list(self.processed_files.keys())  # Admin sees all files
        elif user_email:
            org = users[user_email]["org"]
            return [file for file, email in self.processed_files.items() if users[email]["org"] == org]
        return list(self.processed_files.keys())