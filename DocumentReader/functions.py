# functions.py
import os
import gradio as gr
import PyPDF2
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, OPENAI_LLM_MODEL, users
from document_processor import DocumentProcessor

# Initialize processor and llm
processor = DocumentProcessor()
llm = ChatOpenAI(
    model=OPENAI_LLM_MODEL,
    temperature=0.2,
    openai_api_key=OPENAI_API_KEY
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
            # Return status, update CheckboxGroup choices and clear selection, and update state
            return status_message, gr.update(choices=updated_files, value=[]), updated_files
        else:
            # No change to choices/state if upload failed
            return status_message, gr.update(choices=current_files, value=[]), current_files

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
            if filename not in processor.processed_files or (
                    processor.processed_files[filename] != user_email and user_email != "admin@example.com"
            ):
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
        # Return status, update CheckboxGroup choices (clear selection), and update state
        return status, gr.update(choices=updated_files, value=[]), updated_files
    except Exception as e:
        return f"Error deleting files: {str(e)}", gr.update(choices=current_files, value=[]), current_files


def update_checkbox_choices(file_list):
    # Return a Gradio Update to set the CheckboxGroup choices and clear selection
    return gr.update(choices=file_list, value=[])


# search the qd db
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
    files = processor.get_processed_files(selected_user_email)
    # Return choices update for the CheckboxGroup (clear selection)
    return gr.update(choices=files, value=[])