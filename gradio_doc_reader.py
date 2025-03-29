import os
import json
import gradio as gr
from extractor import PdfExtractor
from chunking import ChunkGenerator
from embedding import DocumentEmbedder
from storeToQdrant import QdrantUploader
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import shutil

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize global objects
embedder = DocumentEmbedder(model="text-embedding-3-small")
uploader = QdrantUploader(
    collection_name="aiml_vector_db",
    qdrant_host="localhost",
    qdrant_port=6333
)
uploader.create_collection(vector_size=1536, distance=Distance.COSINE)

# Initialize LLM with model and temperature
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=openai_api_key
)

# Directories
SOURCE_DIR = "source_dir"
CONVERTED_DIR = "converted_dir"
PROCESSED_DIR = "processed_files"
CHUNK_DIR = "chunked_docs"
EMBEDDED_DIR = "embedded_docs"
PROCESSED_TRACKER = "processed_files.json"
JSON_ARCHIVE_DIR = os.path.join(PROCESSED_DIR, "json_files")

# Ensure directories exist
for dir_path in [SOURCE_DIR, CONVERTED_DIR, PROCESSED_DIR, CHUNK_DIR, EMBEDDED_DIR, JSON_ARCHIVE_DIR]:
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


def cleanup_and_archive():
    """Move all JSON files to PROCESSED_DIR/json_files and clean up directories."""
    temp_dirs = [SOURCE_DIR, CONVERTED_DIR, CHUNK_DIR, EMBEDDED_DIR]
    for dir_path in temp_dirs:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if filename.endswith(".json"):
                dest_path = os.path.join(JSON_ARCHIVE_DIR, filename)
                base_name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(JSON_ARCHIVE_DIR, f"{base_name}_{counter}{ext}")
                    counter += 1
                shutil.move(file_path, dest_path)
            elif filename.endswith(".pdf") and dir_path == SOURCE_DIR:
                continue
            else:
                os.remove(file_path)


def get_uploaded_files():
    """Return a formatted string of processed files."""
    processed_files = load_processed_files()
    if not processed_files["processed"]:
        return "No files have been uploaded yet."
    return "## Uploaded Files\n" + "\n".join([f"- {filename}" for filename in processed_files["processed"]])


def process_pdf(file):
    """Process an uploaded PDF through the pipeline if not already processed."""
    try:
        filename = os.path.basename(file.name)
        processed_files = load_processed_files()

        if filename in processed_files["processed"]:
            return f"File '{filename}' has already been processed and uploaded to Qdrant.", get_uploaded_files()

        pdf_path = os.path.join(SOURCE_DIR, filename)
        shutil.copy(file.name, pdf_path)

        extractor = PdfExtractor()
        extractor.extract_pdfs(SOURCE_DIR, CONVERTED_DIR, PROCESSED_DIR)

        chunker = ChunkGenerator(strategy="recursive_char", chunk_size=800, chunk_overlap=150)
        chunker.chunk_document(CONVERTED_DIR, CHUNK_DIR)

        embedder.embed_chunks(CHUNK_DIR, EMBEDDED_DIR)

        uploader.upload_embeddings(EMBEDDED_DIR)

        processed_files["processed"].append(filename)
        save_processed_files(processed_files)

        cleanup_and_archive()

        return (f"Successfully processed '{filename}' and uploaded to Qdrant. "
                f"JSON files archived in {JSON_ARCHIVE_DIR}."), get_uploaded_files()
    except Exception as e:
        return f"Error processing PDF: {str(e)}", get_uploaded_files()


def search_qdrant(query):
    """Search Qdrant and return formatted results as context."""
    try:
        query_embedding = embedder.embeddings.embed_query(query)
        results = uploader.search(query_vector=query_embedding, limit=5)

        context = ""
        for i, result in enumerate(results):
            score = result.score
            text = result.payload.get("text", "No text available")
            source_file = result.payload.get("source_file", "Unknown source")
            context += f"Result {i + 1} (Score: {score:.4f})\n"
            context += f"Source: {source_file}\n"
            context += f"Text: {text}\n\n"

        return context if context else "No relevant information found in the documents."
    except Exception as e:
        return f"Error searching Qdrant: {str(e)}"


def chatbot_response(message, history):
    """Generate a response using LLM with Qdrant search results as context."""
    try:
        # Get context from Qdrant
        context = search_qdrant(message)

        # Prepare the prompt for the LLM
        system_prompt = (
                "You are a helpful assistant that answers questions based on the provided document context. "
                "Use the following context to answer the user's question. If the context doesn't contain "
                "relevant information, say so and provide a general response if applicable.\n\n"
                "Context:\n" + context
        )

        # Convert history to OpenAI-style messages and append new message
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({"role": "user", "content": msg["content"]}) if msg["role"] == "user" else \
                messages.append({"role": "assistant", "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        # Get response from LLM
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating response: {str(e)}"


def chat_handler(message, history):
    """Handle chatbot input and return updated history in messages format."""
    if history is None:
        history = []

    response = chatbot_response(message, history)
    # Append new user message and assistant response in dictionary format
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, ""  # Return updated history and clear the input box


def clear_chat():
    """Clear the chatbot history."""
    return []


# Gradio Interface
with gr.Blocks(title="Document Reader with Qdrant") as demo:
    gr.Markdown("# Document Reader with Qdrant")

    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_output = gr.Textbox(label="Processing Status")
        uploaded_files_display = gr.Markdown(label="Processed Files", value=get_uploaded_files())
        upload_btn = gr.Button("Process PDF")
        upload_btn.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=[upload_output, uploaded_files_display]
        )

    with gr.Tab("Search Documents"):
        gr.Markdown("## Chat with Your Documents")
        chatbot = gr.Chatbot(label="Document Chatbot", type="messages")
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Ask anything about your uploaded documents...",
        )
        clear = gr.Button("Clear Chat")

        # Handle chat submission
        msg.submit(
            fn=chat_handler,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]  # Update chatbot and clear input
        )
        clear.click(
            fn=clear_chat,
            inputs=None,
            outputs=chatbot
        )

# Launch the app
demo.launch()