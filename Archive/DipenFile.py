import os
import json
import uuid
from datetime import datetime
from typing import List
import gradio as gr
from dotenv import load_dotenv
import fitz  # Import PyMuPDF as fitz

print(f"PyMuPDF version: {fitz.__version__}")  # Debug
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

load_dotenv()


class PdfProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
        self.collection_name = "document_embeddings"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists, creating it if necessary."""
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")  # Debug

    def _extract_text(self, file_path):
        """Extract text from a PDF using PyMuPDF."""
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text("text") or ""  # Extract text from each page
        return text

    def _chunk_text(self, text, max_chunk_size=500, min_chunk_size=100, similarity_threshold=0.5):
        """Performs semantic chunking using embeddings and cosine similarity."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentence_embeddings = self.embeddings.embed_documents(sentences)
        chunks, chunk, chunk_embedding = [], [], []
        chunk_token_length = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            if not chunk:
                chunk.append(sentence)
                chunk_embedding.append(embedding)
                chunk_token_length = len(sentence)
                continue

            similarity = cosine_similarity([embedding], [chunk_embedding[-1]])[0][0]
            if chunk_token_length + len(sentence) > max_chunk_size:
                if chunk_token_length >= min_chunk_size:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_embedding = [embedding]
                chunk_token_length = len(sentence)
            elif similarity > similarity_threshold:
                chunk.append(sentence)
                chunk_embedding.append(embedding)
                chunk_token_length += len(sentence)
            else:
                if chunk_token_length >= min_chunk_size:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_embedding = [embedding]
                chunk_token_length = len(sentence)

        if chunk and (chunk_token_length >= min_chunk_size or len(chunks) == 0):
            chunks.append(" ".join(chunk))
        unique_chunks = list(dict.fromkeys(chunks))
        return unique_chunks

    def _generate_summary(self, text: str) -> str:
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        summary = '. '.join(sentences[:2]) if len(sentences) >= 2 else text[:100]
        return summary if summary.endswith('.') else summary + '.'

    def _generate_full_summary(self, text: str) -> str:
        """Generate a 5-point summary for the entire document."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = PromptTemplate(
            input_variables=["text"],
            template="""Generate a concise 5-point summary of the following text in bullet point format:

            {text}

            Summary:"""
        )
        chain = prompt | llm
        summary = chain.invoke({"text": text[:10000]}).content.strip()  # Limit text to avoid token limits
        return summary

    def process_and_store_pdf(self, file_obj):
        self._ensure_collection_exists()
        text = self._extract_text(file_obj.name)
        filename = os.path.basename(file_obj.name)
        chunks = self._chunk_text(text)
        full_summary = self._generate_full_summary(text)
        points = []
        batch_size = 100
        seen_chunks = set()

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = self.embeddings.embed_documents(batch_chunks)

            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                if chunk in seen_chunks:
                    continue
                seen_chunks.add(chunk)
                point_id = str(uuid.uuid4())
                summary = self._generate_summary(chunk)
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source_file": filename,
                            "chunk_id": i + j,
                            "summary": summary,
                            "original_file_name": filename,
                            "conversion_date": datetime.now().isoformat(),
                            "full_summary": full_summary
                        }
                    )
                )

            if points:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                points = []

        return filename

    def get_document_summary(self, filename: str) -> str:
        """Retrieve the full summary for a specific document."""
        scroll_response = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="original_file_name",
                        match=models.MatchValue(value=filename)
                    )
                ]
            ),
            limit=1,
            with_payload=True
        )
        points, _ = scroll_response
        if points and "full_summary" in points[0].payload:
            return points[0].payload["full_summary"]
        return "No summary available."

    def get_processed_files(self):
        self._ensure_collection_exists()
        processed_files = set()
        scroll_response = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True
        )
        points, next_offset = scroll_response
        while points:
            for point in points:
                original_file_name = point.payload.get("original_file_name")
                if original_file_name:
                    processed_files.add(original_file_name)
            if next_offset is None:
                break
            scroll_response = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True
            )
            points, next_offset = scroll_response
        return processed_files

    def delete_file(selected_files):
        if selected_files and len(selected_files) > 0:
            selected_file = selected_files[0]
            processor.delete_file_points(selected_file)
            updated_choices = sorted(processor.get_processed_files())
            return f"Deleted: {selected_file}", updated_choices

    return "No file selected", sorted(processor.get_processed_files())


def query_vector_db(query: str) -> dict:
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    if not qdrant_client.collection_exists(collection_name="document_embeddings"):
        qdrant_client.create_collection(
            collection_name="document_embeddings",
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
        print("Created collection 'document_embeddings' in query_vector_db")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings.embed_query(query)
    query_response = qdrant_client.query_points(
        collection_name="document_embeddings",
        query=query_embedding,
        limit=5,
        with_payload=True
    )
    search_results = query_response.points
    if not search_results:
        return {"context": "No relevant information found.", "source": "None", "chunks": []}
    context = "\n\n".join(
        [result.payload.get("text", "No text available") for result in search_results]
    )
    chunks = [result.payload.get("text", "No text available") for result in search_results]
    source = search_results[0].payload.get("original_file_name", "Unknown")
    return {"context": context, "source": source, "chunks": chunks}


def classify_relevant_file(question, processor):
    processed_files = processor.get_processed_files()
    return list(processed_files)[0] if processed_files else None


def query_documents(question, processor):
    relevant_file = classify_relevant_file(question, processor)
    tool_output = query_vector_db(query=question)
    answer_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template="""Based solely on the provided context, provide a concise, accurate answer to the question. Do not use external knowledge.

Question: {question}
Context: {context}

Answer: """
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = answer_prompt | llm
    answer = chain.invoke({"question": question, "context": tool_output["context"]}).content.strip()
    if "context does not" in answer:
        final_answer = answer
    else:
        final_answer = f"{answer} [Source: {tool_output['source']}]"
    return {"answer": final_answer, "chunks": tool_output["chunks"]}


def delete_file(selected_files, processor):
    if selected_files and len(selected_files) > 0:
        selected_file = selected_files[0]
        processor.delete_file_points(selected_file)
        updated_choices = sorted(processor.get_processed_files())
        return f"Deleted: {selected_file}", updated_choices
    return "No file selected", sorted(processor.get_processed_files())


def summarize_document(selected_files, history):
    if not selected_files or len(selected_files) == 0:
        return history, "No file selected for summary"

    selected_file = selected_files[0]
    summary = processor.get_document_summary(selected_file)

    summary_message = f"""5 Point Summary of {selected_file}

{summary}

If you need specific information, type your question in the box below and click Send."""

    new_history = history + [
        {"role": "assistant", "content": summary_message}
    ]
    return new_history, f"Summarized: {selected_file}"


processor = PdfProcessor()


def chat_function(message, history):
    response = query_documents(message, processor)
    answer = response["answer"]
    chunks = "\n\n".join([f"Chunk {i + 1}: {chunk}" for i, chunk in enumerate(response["chunks"])])
    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer}
    ]
    return new_history, "", chunks


def upload_file(file):
    if file is None:
        return "Please upload a PDF file"
    filename = processor.process_and_store_pdf(file)
    return f"Processed: {filename}"


modern_theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    text_size="md",
)

custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif !important; }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { text-align: center; padding: 1rem 0; }
    .chatbot .message { border-radius: 10px; padding: 10px; margin: 5px 0; }
    .user-message { background-color: #E94560 !important; color: white; }
    .assistant-message { background-color: #0F3460 !important; color: white; }
    .message-input { background-color: #0F3460; color: white; border: 1px solid #E94560; border-radius: 8px; }
    .message-input textarea { height: 60px !important; }
    .send-btn { background-color: #E94560; color: white; border: none; height: 60px !important; display: flex; align-items: center; justify-content: center; font-size: 18px !important; font-weight: 500; }
    .send-btn:hover { background-color: #FF6B6B; }
    .file-upload { background-color: #16213E; border: 1px dashed #E94560; border-radius: 8px; height: 200px; }
    .status-text { color: #E94560; font-size: 14px; text-align: center; }
    .info-box { background-color: #16213E; color: white; border: 1px solid #0F3460; border-radius: 8px; }
    .gr-button { transition: all 0.3s ease; }
    .gr-accordion { background-color: #16213E; border: 1px solid #0F3460; }
    .square-column { min-width: 300px; max-width: 400px; }
    .checkbox-group { max-height: 150px; overflow-y: auto; }
    .gr-accordion > div > button { font-size: 18px !important; font-weight: 500; }
    .header-text { color: white; font-size: 16px; margin-top: -10px; }
    .warning-text { color: #E94560; font-size: 16px; margin-top: -10px; }
"""

with gr.Blocks(theme=modern_theme, css=custom_css) as demo:
    gr.Markdown(
        """
        # Zaplytic
        <p style='color: #E94560; font-size: 16px; margin-top: -10px;'>Upload your document to store it in a vector database and chat with it to get instant answers.</p>
        <p style='color: red; font-size: 16px; margin-top: -10px;'><b style='color: red;'>Do not upload personal or private data</b></p>
        """,
        elem_classes="header"
    )

    with gr.Row(elem_classes="container"):
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                label=None,
                type="messages",
                height=500,
                elem_classes="info-box"
            )
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask anything...",
                    show_label=False,
                    container=False,
                    elem_classes="message-input",
                    lines=3,
                    scale=8
                )
                send_btn = gr.Button(
                    "Send",
                    size="sm",
                    elem_classes="send-btn",
                    scale=2
                )

        with gr.Column(scale=3):
            with gr.Accordion("File Upload", open=False):
                file_upload = gr.File(
                    label="Drop PDF here",
                    file_types=[".pdf"],
                    elem_classes="file-upload",
                    height=200
                )
                upload_btn = gr.Button(
                    "Upload PDF",
                    size="sm"
                )
                upload_status = gr.Textbox(
                    value="Upload a PDF to begin",
                    show_label=False,
                    interactive=False,
                    elem_classes="status-text"
                )

            with gr.Accordion("Manage Documents", open=False):
                files_selection = gr.CheckboxGroup(
                    choices=sorted(processor.get_processed_files()),
                    label="Document Selection (Select one)",
                    interactive=True,
                    elem_classes="info-box checkbox-group"
                )
                with gr.Row():
                    summarize_btn = gr.Button(
                        "Summarize Document",
                        size="sm",
                        scale=1
                    )
                    delete_btn = gr.Button(
                        "Delete Document",
                        size="sm",
                        scale=1
                    )
                chunks_display = gr.Textbox(
                    value="Query results will appear here",
                    label="Retrieved Chunks",
                    interactive=False,
                    lines=5,
                    max_lines=5,
                    elem_classes="info-box"
                )


    def update_selection(selected_files):
        if len(selected_files) > 1:
            return [selected_files[-1]]
        return selected_files


    send_btn.click(
        chat_function,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, chunks_display]
    )
    msg.submit(
        chat_function,
        inputs=[msg, chatbot],
        outputs=[chatbot, msg, chunks_display]
    )
    upload_btn.click(
        upload_file,
        inputs=[file_upload],
        outputs=[upload_status]
    ).then(
        lambda: gr.update(choices=sorted(processor.get_processed_files())),
        outputs=[files_selection]
    ).then(
        lambda: [],  # Clear selection after upload
        outputs=[files_selection]
    )

    files_selection.change(
        update_selection,
        inputs=[files_selection],
        outputs=[files_selection]
    )

    summarize_btn.click(
        summarize_document,
        inputs=[files_selection, chatbot],
        outputs=[chatbot, upload_status]
    )

    delete_btn.click(
        delete_file,
        inputs=[files_selection],
        outputs=[upload_status, files_selection]
    ).then(
        lambda: [],  # Clear selection after deletion
        outputs=[files_selection]
    )

if __name__ == "__main__":
    print(f"===== Application Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)