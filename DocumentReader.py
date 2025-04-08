import os
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import spacy
import nltk
from typing import List
import PyPDF2
import uuid
import time
import hashlib

# User dictionary with emails as keys
users = {
    "john.doe123@example.com": {"username": "User1", "id": str(uuid.uuid4())},
    "jane.smith456@example.com": {"username": "User2", "id": str(uuid.uuid4())},
    "bob.jones789@example.com": {"username": "User3", "id": str(uuid.uuid4())},
    "alice.brown321@example.com": {"username": "User4", "id": str(uuid.uuid4())},
    "mike.wilson654@example.com": {"username": "User5", "id": str(uuid.uuid4())}
}

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
                            "content_hash": content_hash
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
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=Filter(must=[FieldCondition(key="user_email", match=MatchValue(value=user_email))]),
            limit=limit,
            with_payload=True
        )
        return results.points

    def get_processed_files(self, user_email: str = None):
        if user_email:
            return [file for file, email in self.processed_files.items() if email == user_email]
        return list(self.processed_files.keys())


# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = "gpt-3.5-turbo"

# Initialize global objects
processor = DocumentProcessor(
    collection_name="aiml_vector_db",
    chunk_strategy="semantic",
    chunk_size=800,
    chunk_overlap=150,
    embedding_model="text-embedding-3-small",
    batch_size=32
)

llm = ChatOpenAI(
    model=OPENAI_LLM_MODEL,
    temperature=0.2,
    openai_api_key=openai_api_key
)


def process_pdf(file, current_files, selected_user_email):
    try:
        filename = os.path.basename(file.name)
        user_email = selected_user_email

        with open(file.name, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text() or ""

        status_message = processor.process_document(text_content, filename, user_email)

        if "Successfully processed" in status_message:
            updated_files = processor.get_processed_files(user_email)
            return status_message, updated_files, updated_files
        else:
            return status_message, current_files, current_files

    except Exception as e:
        return f"Error processing PDF: {str(e)}", current_files, current_files


def delete_pdfs(filenames, current_files, selected_user_email):
    try:
        if not filenames:
            return "No files selected for deletion.", current_files, current_files

        user_email = selected_user_email
        deleted_files = []
        errors = []

        for filename in filenames:
            if filename not in processor.processed_files or processor.processed_files[filename] != user_email:
                errors.append(f"File '{filename}' not found or not owned by {user_email}.")
                continue

            remaining = processor.delete_by_source_file(filename)
            if remaining == -1:
                errors.append(f"Failed to delete embeddings for '{filename}'.")
            elif remaining > 0:
                errors.append(f"Some embeddings for '{filename}' were not deleted ({remaining} remain).")
            else:
                deleted_files.append(filename)

        status = ""
        if deleted_files:
            status += f"Successfully deleted {len(deleted_files)} file(s): {', '.join(deleted_files)}."
        if errors:
            status += "\nErrors:\n" + "\n".join(errors)

        updated_files = processor.get_processed_files(user_email)
        return status, updated_files, updated_files
    except Exception as e:
        return f"Error deleting files: {str(e)}", current_files, current_files


def update_checkbox_choices(file_list):
    return file_list


def search_qdrant(query, selected_user_email):
    try:
        user_email = selected_user_email
        query_embedding = processor.embeddings.embed_query(query)
        results = processor.search(query_vector=query_embedding, user_email=user_email, limit=5)

        context = ""
        source_files = set()
        for i, result in enumerate(results):
            text = result.payload.get("text", "No text available")
            source_file = result.payload.get("source_file", "Unknown source")
            context += f"Source: {source_file}\n"
            context += f"Text: {text}\n\n"
            source_files.add(source_file)

        return context if context else "No relevant information found in your documents.", source_files
    except Exception as e:
        return f"Error searching Qdrant: {str(e)}", set()


def chatbot_response(message, history, selected_user_email):
    try:
        context, source_files = search_qdrant(message, selected_user_email)

        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided document context. "
            "Use the following context to answer the user's question. If the context doesn't contain "
            "relevant information, say so and do not use external knowledge. Do not include source "
            "information in your response; it will be appended separately.\n\n"
            "Context:\n" + context
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": "user", "content": msg["content"]}) if msg["role"] == "user" else \
                messages.append({"role": "assistant", "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        response = llm.invoke(messages).content

        if source_files and "Sources:" not in response:
            source_list = ", ".join(source_files)
            response += f"\n\nSources: {source_list}"
        elif not source_files and "Sources:" not in response:
            response += "\n\nSources: None identified"

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}\n\nSources: None identified"


def chat_handler(message, history, selected_user_email):
    if history is None:
        history = []
    response = chatbot_response(message, history, selected_user_email)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""


def clear_chat():
    return []


def update_file_list(selected_user_email):
    return processor.get_processed_files(selected_user_email)


# Gradio Interface
with gr.Blocks(title="Document Reader") as demo:
    gr.Markdown("# Document Reader with Qdrant")

    file_state = gr.State(value=processor.get_processed_files())
    user_state = gr.State(value="john.doe123@example.com")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Chat with Your Documents")
            chatbot = gr.Chatbot(label="Document Chatbot", type="messages", height=400)
            msg = gr.Textbox(
                label="Your Message",
                placeholder="Ask anything about your uploaded documents..."
            )
            with gr.Row():
                submit_btn = gr.Button("Submit")
                clear_btn = gr.Button("Clear Chat")
            submit_btn.click(
                fn=chat_handler,
                inputs=[msg, chatbot, user_state],
                outputs=[chatbot, msg]
            )
            clear_btn.click(
                fn=clear_chat,
                inputs=None,
                outputs=chatbot
            )

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## Select User")
                user_dropdown = gr.Dropdown(
                    label="User",
                    choices=list(users.keys()),
                    value="john.doe123@example.com",
                    interactive=True
                )

            with gr.Group():
                gr.Markdown("## Upload PDF")
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Process PDF")
                upload_output = gr.Textbox(label="Status")

            with gr.Group():
                gr.Markdown("## Manage Documents")
                delete_checkboxes = gr.CheckboxGroup(
                    label="Select PDFs to Delete",
                    value=[],
                    choices=processor.get_processed_files("john.doe123@example.com"),
                    interactive=True
                )
                delete_btn = gr.Button("Delete Selected PDFs")

    user_dropdown.change(
        fn=update_file_list,
        inputs=[user_dropdown],
        outputs=[delete_checkboxes]
    ).then(
        fn=lambda x: x,
        inputs=[user_dropdown],
        outputs=[user_state]
    )

    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, file_state, user_state],
        outputs=[upload_output, delete_checkboxes, file_state]
    ).then(
        fn=update_checkbox_choices,
        inputs=[file_state],
        outputs=[delete_checkboxes]
    )

    delete_btn.click(
        fn=delete_pdfs,
        inputs=[delete_checkboxes, file_state, user_state],
        outputs=[upload_output, delete_checkboxes, file_state]
    ).then(
        fn=update_checkbox_choices,
        inputs=[file_state],
        outputs=[delete_checkboxes]
    )

if __name__ == "__main__":
    demo.launch()