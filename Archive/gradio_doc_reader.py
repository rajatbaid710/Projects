import os
import json
import gradio as gr
from document_processor import DocumentProcessor
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import shutil

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
SOURCE_DIR = "../source_dir"
CONVERTED_DIR = "../converted_dir"
PROCESSED_DIR = "../processed_files"
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
        with open(file, "r", encoding="utf-8") as f:
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

            # Delete PDF from PROCESSED_DIR
            pdf_path = os.path.join(PROCESSED_DIR, filename)
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"Deleted PDF file: {pdf_path}")
            else:
                print(f"PDF file not found: {pdf_path}")

            # Delete embeddings from Qdrant
            remaining = processor.delete_by_source_file(filename)
            if remaining == -1:
                errors.append(f"Failed to delete embeddings for '{filename}'.")
            elif remaining > 0:
                errors.append(f"Some embeddings for '{filename}' were not deleted ({remaining} remain).")

            # Update processed files tracker
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
            # score = result.score
            text = result.payload.get("text", "No text available")
            source_file = result.payload.get("source_file", "Unknown source")
            # context += f"Result {i + 1} (Score: {score:.4f})\n"
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

        # Upload PDF
        upload_btn.click(
            fn=process_pdf,
            inputs=pdf_input,
            outputs=[upload_output, delete_checkboxes]
        )

        # Delete PDFs
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

demo.launch()