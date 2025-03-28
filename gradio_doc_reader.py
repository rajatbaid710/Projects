# gradio_doc_reader.py
import os
import json
import gradio as gr
from extractor import PdfExtractor
from chunking import ChunkGenerator
from embedding import DocumentEmbedder
from storeToQdrant import QdrantUploader
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
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

# Directories
SOURCE_DIR = "source_dir"
CONVERTED_DIR = "converted_dir"
PROCESSED_DIR = "processed_files"
CHUNK_DIR = "chunked_docs"
EMBEDDED_DIR = "embedded_docs"
PROCESSED_TRACKER = "processed_files.json"  # File to track processed PDFs
JSON_ARCHIVE_DIR = os.path.join(PROCESSED_DIR, "json_files")  # Subdirectory for JSON files

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
                # Move JSON files to json_files subdirectory
                dest_path = os.path.join(JSON_ARCHIVE_DIR, filename)
                # Avoid overwriting by adding a suffix if file exists
                base_name, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(dest_path):
                    dest_path = os.path.join(JSON_ARCHIVE_DIR, f"{base_name}_{counter}{ext}")
                    counter += 1
                shutil.move(file_path, dest_path)
            elif filename.endswith(".pdf") and dir_path == SOURCE_DIR:
                # Move PDFs to processed_files (already handled by PdfExtractor)
                continue
            else:
                # Remove any other files (e.g., temporary files)
                os.remove(file_path)


def process_pdf(file):
    """Process an uploaded PDF through the pipeline if not already processed."""
    try:
        filename = os.path.basename(file.name)
        processed_files = load_processed_files()

        # Check if file has already been processed
        if filename in processed_files["processed"]:
            return f"File '{filename}' has already been processed and uploaded to Qdrant."

        # Save uploaded file to source_dir
        pdf_path = os.path.join(SOURCE_DIR, filename)
        shutil.copy(file.name, pdf_path)

        # Step 1: Extract PDF to JSON
        extractor = PdfExtractor()
        extractor.extract_pdfs(SOURCE_DIR, CONVERTED_DIR, PROCESSED_DIR)

        # Step 2: Chunk the JSON
        chunker = ChunkGenerator(strategy="recursive_char", chunk_size=512, chunk_overlap=50)
        chunker.chunk_document(CONVERTED_DIR, CHUNK_DIR)

        # Step 3: Embed the chunks
        embedder.embed_chunks(CHUNK_DIR, EMBEDDED_DIR)

        # Step 4: Upload to Qdrant
        uploader.upload_embeddings(EMBEDDED_DIR)

        # Update processed files list
        processed_files["processed"].append(filename)
        save_processed_files(processed_files)

        # Clean up and archive JSON files
        cleanup_and_archive()

        return f"Successfully processed '{filename}' and uploaded to Qdrant. JSON files archived in {JSON_ARCHIVE_DIR}."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"


def search_qdrant(query):
    """Search Qdrant based on user query."""
    try:
        # Generate embedding for the query
        query_embedding = embedder.embeddings.embed_query(query)

        # Search Qdrant
        results = uploader.search(query_vector=query_embedding, limit=5)

        # Format results
        output = ""
        for i, result in enumerate(results):
            score = result.score
            text = result.payload.get("text", "No text available")
            source_file = result.payload.get("source_file", "Unknown source")
            output += f"**Result {i + 1} (Score: {score:.4f})**\n"
            output += f"Source: {source_file}\n"
            output += f"Text: {text}\n\n"

        return output if output else "No results found."
    except Exception as e:
        return f"Error searching Qdrant: {str(e)}"


# Gradio Interface
with gr.Blocks(title="Document Reader with Qdrant") as demo:
    gr.Markdown("# Document Reader with Qdrant")

    with gr.Tab("Upload PDF"):
        pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_output = gr.Textbox(label="Processing Status")
        upload_btn = gr.Button("Process PDF")
        upload_btn.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=upload_output
        )

    with gr.Tab("Search Documents"):
        query_input = gr.Textbox(label="Enter your query",
                                 placeholder="e.g., What does the inspection report say about plumbing?")
        search_output = gr.Markdown(label="Search Results")
        search_btn = gr.Button("Search")
        search_btn.click(
            fn=search_qdrant,
            inputs=query_input,
            outputs=search_output
        )

# Launch the app
demo.launch()