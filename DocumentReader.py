import os
import json
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import spacy
import nltk
from typing import List, Dict
import shutil

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
                            # "source_file": chunk_data["source_file"],
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
            print(f"Deleted embeddings for {source_file} from {self.collection_name}")
            return deleted_count
        except Exception as e:
            print(f"Error deleting embeddings for {source_file}: {str(e)}")
            return -1

    def search(self, query_vector: List[float], limit: int = 5):
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True
        )
        return results.points


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

# Directories
SOURCE_DIR = "source_dir"
CONVERTED_DIR = "converted_dir"
PROCESSED_DIR = "processed_files"
PROCESSED_TRACKER = "processed_files.json"
JSON_ARCHIVE_DIR = os.path.join(PROCESSED_DIR, "json_files")

# Ensure directories exist
for dir_path in [SOURCE_DIR, CONVERTED_DIR, PROCESSED_DIR, JSON_ARCHIVE_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# Load or initialize processed files tracker
def load_processed_files():
    if os.path.exists(PROCESSED_TRACKER):
        with open(PROCESSED_TRACKER, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"processed": []}


def save_processed_files(processed_list):
    with open(PROCESSED_TRACKER, "w", encoding="utf-8") as f:
        json.dump(processed_list, f, indent=4)


def cleanup():
    temp_dirs = [CONVERTED_DIR]
    for dir_path in temp_dirs:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if filename.endswith(".json"):
                os.remove(file_path)


def get_uploaded_files():
    processed_files = load_processed_files()
    return processed_files["processed"]


def process_pdf(file):
    try:
        filename = os.path.basename(file.name)
        processed_files = load_processed_files()

        if filename in processed_files["processed"]:
            return f"File '{filename}' has already been processed and uploaded to Qdrant.", get_uploaded_files()

        pdf_path = os.path.join(SOURCE_DIR, filename)
        shutil.copy(file.name, pdf_path)

        json_file = os.path.join(CONVERTED_DIR, os.path.splitext(filename)[0] + ".json")
        with open(file.name, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        processor.process_document(json_data, filename)

        processed_files["processed"].append(filename)
        save_processed_files(processed_files)

        cleanup()

        return f"Successfully processed '{filename}' and uploaded to Qdrant.", get_uploaded_files()
    except Exception as e:
        return f"Error processing PDF: {str(e)}", get_uploaded_files()


def delete_pdfs(filenames):
    try:
        if not filenames:
            return "No files selected for deletion.", get_uploaded_files()

        processed_files = load_processed_files()
        deleted_files = []
        errors = []

        for filename in filenames:
            if filename not in processed_files["processed"]:
                errors.append(f"File '{filename}' not found in processed list.")
                continue

            pdf_path = os.path.join(PROCESSED_DIR, filename)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"Deleted PDF file: {pdf_path}")
            else:
                print(f"PDF file not found: {pdf_path}")

            remaining = processor.delete_by_source_file(filename)
            if remaining == -1:
                errors.append(f"Failed to delete embeddings for '{filename}'.")
            elif remaining > 0:
                errors.append(f"Some embeddings for '{filename}' were not deleted ({remaining} remain).")

            processed_files["processed"].remove(filename)
            deleted_files.append(filename)

        save_processed_files(processed_files)

        status = ""
        if deleted_files:
            status += f"Successfully deleted {len(deleted_files)} file(s): {', '.join(deleted_files)}."
        if errors:
            status += "\nErrors:\n" + "\n".join(errors)

        return status, get_uploaded_files()
    except Exception as e:
        return f"Error deleting files: {str(e)}", get_uploaded_files()


def search_qdrant(query):
    try:
        query_embedding = processor.embeddings.embed_query(query)
        results = processor.search(query_vector=query_embedding, limit=5)

        context = ""
        source_files = set()
        for i, result in enumerate(results):
            text = result.payload.get("text", "No text available")
            source_file = result.payload.get("source_file", "Unknown source")
            context += f"Source: {source_file}\n"
            context += f"Text: {text}\n\n"
            source_files.add(source_file)

        return context if context else "No relevant information found in the documents.", source_files
    except Exception as e:
        return f"Error searching Qdrant: {str(e)}", set()


def chatbot_response(message, history):
    try:
        context, source_files = search_qdrant(message)

        system_prompt = (
                "You are a helpful assistant that answers questions based on the provided document context. "
                "Use the following context to answer the user's question. If the context doesn't contain "
                "relevant information, say so and do not use external knowledge.\n\n"
                "Context:\n" + context
        )

        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": "user", "content": msg["content"]}) if msg["role"] == "user" else \
                messages.append({"role": "assistant", "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        response = llm.invoke(messages).content
        if source_files:
            source_list = ", ".join(source_files)
            response += f"\n\nSources: {source_list}"
        else:
            response += "\n\nSources: None identified"
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}\n\nSources: None identified"


def chat_handler(message, history):
    if history is None:
        history = []
    response = chatbot_response(message, history)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""


def clear_chat():
    return []


# Gradio Interface
with gr.Blocks(title="Document Reader") as demo:
    gr.Markdown("# Document Reader with Qdrant")

    with gr.Tab("Upload PDF"):
        gr.Markdown("## Upload or Delete PDFs")
        with gr.Row():
            with gr.Column():
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                upload_btn = gr.Button("Process PDF")
            with gr.Column():
                delete_checkboxes = gr.CheckboxGroup(
                    label="Select PDFs to Delete",
                    choices=get_uploaded_files(),
                    interactive=True
                )
                delete_btn = gr.Button("Delete Selected PDFs")
        upload_output = gr.Textbox(label="Status")

        upload_btn.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=[upload_output, delete_checkboxes]
        )

        delete_btn.click(
            fn=delete_pdfs,
            inputs=delete_checkboxes,
            outputs=[upload_output, delete_checkboxes]
        )

    with gr.Tab("Search Documents"):
        gr.Markdown("## Chat with Your Documents")
        chatbot = gr.Chatbot(label="Document Chatbot", type="messages")
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Ask anything about your uploaded documents...",
        )
        clear = gr.Button("Clear Chat")
        msg.submit(
            fn=chat_handler,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        clear.click(
            fn=clear_chat,
            inputs=None,
            outputs=chatbot
        )

if __name__ == "__main__":
    demo.launch()
