import os
import json
import gradio as gr
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
import spacy
import nltk
from typing import List, Tuple
import uuid
import time
import hashlib

# File readers
import PyPDF2
import docx                        # python-docx  → Word files
import openpyxl                    # openpyxl     → Excel files
from PIL import Image              # Pillow       → Images
import pytesseract                 # pytesseract  → OCR on images
load_dotenv()
openai_api_key   = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")  # set in .env or falls back to gpt-3.5-turbo

# ── User directory ────────────────────────────────────────────────────────────
users = {
    "john.doe123@example.com":   {"username": "User1", "id": str(uuid.uuid4()), "org": "Org1"},
    "jane.smith456@example.com": {"username": "User2", "id": str(uuid.uuid4()), "org": "Org1"},
    "bob.jones789@example.com":  {"username": "User3", "id": str(uuid.uuid4()), "org": "Org2"},
    "alice.brown321@example.com":{"username": "User4", "id": str(uuid.uuid4()), "org": "Org2"},
    "mike.wilson654@example.com":{"username": "User5", "id": str(uuid.uuid4()), "org": "Org2"},
    "admin@example.com":         {"username": "Admin", "id": str(uuid.uuid4()), "org": None},
}

# ── NLP setup ─────────────────────────────────────────────────────────────────
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    spacy_nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    spacy_nlp = spacy.load("en_core_web_sm")


# ══════════════════════════════════════════════════════════════════════════════
#  FILE-TYPE AGENT
#  Decides how to read a file and returns plain text.
# ══════════════════════════════════════════════════════════════════════════════

# Tool definitions the LLM can choose from
FILE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_pdf",
            "description": "Extract text from a PDF file using PyPDF2.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the PDF file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_word",
            "description": "Extract text from a Microsoft Word (.docx) file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the .docx file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_excel",
            "description": "Extract all cell values from a Microsoft Excel (.xlsx / .xls) file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the Excel file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_image",
            "description": "Use OCR (pytesseract) to extract text from an image file (png, jpg, jpeg, tiff, bmp, webp).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the image file"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_text",
            "description": "Read a plain-text file (.txt, .md, .csv, .json, .xml, .html, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the text file"}
                },
                "required": ["file_path"]
            }
        }
    },
]


# ── Concrete tool implementations ─────────────────────────────────────────────

def read_pdf(file_path: str) -> str:
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text


def read_word(file_path: str) -> str:
    doc = docx.Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    # Also grab table cell text
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text.strip())
    return "\n".join(paragraphs)


def read_excel(file_path: str) -> str:
    wb = openpyxl.load_workbook(file_path, data_only=True)
    lines = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        lines.append(f"=== Sheet: {sheet} ===")
        for row in ws.iter_rows(values_only=True):
            row_text = "\t".join(str(cell) for cell in row if cell is not None)
            if row_text.strip():
                lines.append(row_text)
    return "\n".join(lines)


def read_image(file_path: str) -> str:
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text


def read_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# Map tool names → functions
TOOL_FUNCTIONS = {
    "read_pdf":   read_pdf,
    "read_word":  read_word,
    "read_excel": read_excel,
    "read_image": read_image,
    "read_text":  read_text,
}


def file_type_agent(file_path: str, filename: str, llm_client) -> Tuple[str, str]:
    """
    An agent that decides which reader tool to use based on the filename,
    calls it, and returns (extracted_text, agent_reasoning_log).

    The LLM chooses the right tool — this is the core agentic pattern:
    Observe → Decide → Act → Return result
    """
    log_lines = []

    # ── Step 1: Ask the LLM to pick a tool ───────────────────────────────────
    system_prompt = (
        "You are a file-reading agent. Given a filename you must call exactly ONE "
        "of the available tools to extract its text content. "
        "Choose the most appropriate tool based on the file extension. "
        "Do not explain yourself — just call the tool."
    )
    user_message = (
        f"Extract text from this file: '{filename}'\n"
        f"Full path: {file_path}"
    )

    log_lines.append(f"🤖 Agent received file: {filename}")
    log_lines.append(f"🔍 Asking LLM to pick the right reader tool...")

    response = llm_client.chat.completions.create(
        model=OPENAI_LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        tools=FILE_TOOLS,
        tool_choice="required",   # force the LLM to always call a tool
    )

    message = response.choices[0].message

    # ── Step 2: Execute the chosen tool ──────────────────────────────────────
    if not message.tool_calls:
        log_lines.append("⚠️  LLM returned no tool call. Falling back to plain-text reader.")
        tool_name = "read_text"
        tool_args = {"file_path": file_path}
    else:
        tool_call = message.tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_args["file_path"] = file_path   # always use the real server path

    log_lines.append(f"🛠️  Agent chose tool: **{tool_name}**")

    if tool_name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool chosen by agent: {tool_name}")

    log_lines.append(f"⚙️  Executing {tool_name}...")
    extracted_text = TOOL_FUNCTIONS[tool_name](file_path)
    char_count = len(extracted_text)
    log_lines.append(f"✅ Extraction complete — {char_count:,} characters extracted.")

    if char_count < 50:
        log_lines.append("⚠️  Very little text extracted. File may be empty or unsupported.")

    agent_log = "\n".join(log_lines)
    return extracted_text, agent_log


# ══════════════════════════════════════════════════════════════════════════════
#  DOCUMENT PROCESSOR  (unchanged logic, same class as before)
# ══════════════════════════════════════════════════════════════════════════════

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
                    user_email  = point.payload.get("user_email")
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
            return processed_files
        except Exception as e:
            print(f"Error fetching processed files: {str(e)}")
            return {}

    def _semantic_split(self, text: str) -> List[str]:
        doc = spacy_nlp(text)
        chunks, current_chunk, current_length = [], [], 0
        for sent in doc.sents:
            if current_length + len(sent.text) > self.chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk  = [current_chunk[-1]] if self.chunk_overlap > 0 else []
                current_length = len(current_chunk[0]) if current_chunk else 0
            current_chunk.append(sent.text)
            current_length += len(sent.text)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def process_document(self, text: str, source_file: str, user_email: str) -> str:
        content_hash = hashlib.sha256(f"{text}{user_email}".encode()).hexdigest()
        org = users[user_email]["org"] if user_email != "admin@example.com" else None

        existing_points = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=[
                FieldCondition(key="user_email",    match=MatchValue(value=user_email)),
                FieldCondition(key="content_hash",  match=MatchValue(value=content_hash))
            ]),
            limit=1,
            with_payload=True
        )[0]

        if existing_points:
            existing_file = existing_points[0].payload.get("source_file", "unknown")
            return f"Duplicate content detected — already exists as '{existing_file}'. Upload skipped."

        chunks = self.splitter(text)
        if not chunks:
            return f"No chunks generated for '{source_file}'. Upload failed."

        points = []
        for i in range(0, len(chunks), self.batch_size):
            batch   = chunks[i:i + self.batch_size]
            embeds  = self.embeddings.embed_documents(batch)
            for chunk, emb in zip(batch, embeds):
                points.append(PointStruct(
                    id=self.next_id,
                    vector=emb,
                    payload={
                        "text":         chunk,
                        "source_file":  source_file,
                        "user_email":   user_email,
                        "content_hash": content_hash,
                        "org":          org
                    }
                ))
                self.next_id += 1

        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
            self.processed_files[source_file] = user_email
            return f"✅ Successfully processed '{source_file}' — {len(chunks)} chunks uploaded."

        return f"Unexpected error: no points generated for '{source_file}'."

    def delete_by_source_file(self, source_file: str) -> int:
        try:
            f = Filter(must=[FieldCondition(key="source_file", match=MatchValue(value=source_file))])
            self.client.delete(collection_name=self.collection_name, points_selector=f)
            remaining = self.client.count(collection_name=self.collection_name,
                                          exact=False, count_filter=f).count
            if source_file in self.processed_files:
                del self.processed_files[source_file]
            return remaining
        except Exception as e:
            print(f"Error deleting {source_file}: {e}")
            return -1

    def search(self, query_vector: List[float], user_email: str, limit: int = 5):
        if user_email == "admin@example.com":
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector, limit=limit, with_payload=True
            )
        else:
            org = users[user_email]["org"]
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(must=[FieldCondition(key="org", match=MatchValue(value=org))]),
                limit=limit, with_payload=True
            )
        return results.points

    def get_processed_files(self, user_email: str = None):
        if user_email == "admin@example.com":
            return list(self.processed_files.keys())
        elif user_email:
            org = users[user_email]["org"]
            return [f for f, e in self.processed_files.items() if users[e]["org"] == org]
        return list(self.processed_files.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  APP INIT
# ══════════════════════════════════════════════════════════════════════════════


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

# Raw OpenAI client needed for tool-use API
from openai import OpenAI
openai_client = OpenAI(api_key=openai_api_key)


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO HANDLER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED_EXTENSIONS = {
    # Documents
    ".pdf", ".docx", ".doc",
    # Spreadsheets
    ".xlsx", ".xls",
    # Images
    ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp",
    # Text
    ".txt", ".md", ".csv", ".json", ".xml", ".html",
}


def process_file(file, current_files, selected_user_email):
    """
    Main upload handler.
    1. Validates file type.
    2. Runs the File-Type Agent to extract text.
    3. Passes extracted text to DocumentProcessor → Qdrant.
    """
    if file is None:
        return "No file uploaded.", "", gr.update(choices=current_files, value=[]), current_files

    filename  = os.path.basename(file.name)
    ext       = os.path.splitext(filename)[1].lower()
    user_email = selected_user_email

    if ext not in SUPPORTED_EXTENSIONS:
        msg = (f"❌ Unsupported file type: '{ext}'\n"
               f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return msg, "", gr.update(choices=current_files, value=[]), current_files

    try:
        # ── File-Type Agent ───────────────────────────────────────────────────
        extracted_text, agent_log = file_type_agent(file.name, filename, openai_client)

        if not extracted_text.strip():
            return (f"⚠️ No text could be extracted from '{filename}'.",
                    agent_log,
                    gr.update(choices=current_files, value=[]),
                    current_files)

        # ── Embed & store in Qdrant ───────────────────────────────────────────
        status_message = processor.process_document(extracted_text, filename, user_email)

        if "Successfully" in status_message:
            updated_files = processor.get_processed_files(user_email)
            return (status_message,
                    agent_log,
                    gr.update(choices=updated_files, value=[]),
                    updated_files)
        else:
            return (status_message,
                    agent_log,
                    gr.update(choices=current_files, value=[]),
                    current_files)

    except Exception as e:
        return (f"❌ Error processing '{filename}': {str(e)}",
                "",
                gr.update(choices=current_files, value=[]),
                current_files)


def delete_pdfs(filenames, current_files, selected_user_email):
    if not filenames:
        return "No files selected for deletion.", gr.update(choices=current_files, value=[]), current_files

    user_email   = selected_user_email
    deleted, errors = [], []

    for filename in filenames:
        if filename not in processor.processed_files or (
            processor.processed_files[filename] != user_email and user_email != "admin@example.com"
        ):
            errors.append(f"'{filename}' not found or not owned by {user_email}.")
            continue
        remaining = processor.delete_by_source_file(filename)
        if remaining == -1:
            errors.append(f"Failed to delete embeddings for '{filename}'.")
        elif remaining > 0:
            errors.append(f"Some embeddings for '{filename}' were not deleted ({remaining} remain).")
        else:
            deleted.append(filename)

    status = ""
    if deleted:
        status += f"✅ Deleted {len(deleted)} file(s): {', '.join(deleted)}."
    if errors:
        status += "\nErrors:\n" + "\n".join(errors)

    updated_files = processor.get_processed_files(user_email)
    return status, gr.update(choices=updated_files, value=[]), updated_files


def search_qdrant(query, selected_user_email):
    try:
        query_embedding = processor.embeddings.embed_query(query)
        results = processor.search(query_vector=query_embedding, user_email=selected_user_email, limit=5)

        if not results:
            return "No relevant information found.", set()

        # Step 1: find the best score per file
        file_best_score = {}
        for result in results:
            sf = result.payload.get("source_file", "Unknown")
            file_best_score[sf] = max(file_best_score.get(sf, 0), result.score)

        # Step 2: only keep files whose best score is within 0.15 of the top file
        # e.g. top file scores 0.82, threshold = 0.67 — car insurance at 0.55 gets dropped
        top_score = max(file_best_score.values())
        relevant_files = {f for f, s in file_best_score.items() if s >= top_score - 0.15}

        context, source_files = "", set()
        for result in results:
            source_file = result.payload.get("source_file", "Unknown")
            if source_file not in relevant_files:
                continue
            text = result.payload.get("text", "")
            context += f"{text}\n\n"
            source_files.add(source_file)

        return context or "No relevant information found.", source_files
    except Exception as e:
        return f"Error searching Qdrant: {str(e)}", set()


def chatbot_response(message, history, selected_user_email):
    try:
        context, source_files = search_qdrant(message, selected_user_email)
        system_prompt = (
            "You are a helpful assistant that answers questions based on the provided document context. "
            "Use only the context below. If it lacks relevant info, say so.\n\n"
            f"Context:\n{context}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            role = "user" if msg["role"] == "user" else "assistant"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": message})

        response = llm.invoke(messages).content

        # We control the Sources line — not the LLM
        # This guarantees exactly one Sources entry with only the relevant files
        if source_files:
            response += f"\n\nSources: {', '.join(sorted(source_files))}"
        else:
            response += "\n\nSources: None identified"
        return response
    except Exception as e:
        return f"Error: {str(e)}\n\nSources: None identified"


def chat_handler(message, history, selected_user_email):
    # Gradio Chatbot expects a list of messages in the format:
    #   [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    # Ensure compatibility whether Gradio passes tuples or dicts.
    if history is None:
        history = []

    normalized_history = []
    for pair in history:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            normalized_history.append({"role": "user", "content": pair[0] or ""})
            normalized_history.append({"role": "assistant", "content": pair[1] or ""})
        elif isinstance(pair, dict) and "role" in pair and "content" in pair:
            normalized_history.append(pair)

    response = chatbot_response(message, normalized_history, selected_user_email)
    normalized_history.append({"role": "user", "content": message})
    normalized_history.append({"role": "assistant", "content": response})
    return normalized_history, ""


def clear_chat():
    return []


def update_file_list(selected_user_email):
    files = processor.get_processed_files(selected_user_email)
    return gr.update(choices=files, value=[])


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(title="Document Reader") as demo:
    gr.Markdown("# 📄 Document Reader with Qdrant")
    gr.Markdown(
        "Upload **PDF, Word, Excel, Images, or Text files**. "
        "An AI agent automatically picks the right reader for each file type."
    )

    file_state = gr.State(value=processor.get_processed_files())
    user_state = gr.State(value="john.doe123@example.com")

    with gr.Row():
        # ── LEFT: Chat ────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("## 💬 Chat with Your Documents")
            chatbot = gr.Chatbot(label="Document Chatbot", height=400)
            msg     = gr.Textbox(
                label="Your Message",
                placeholder="Ask anything about your uploaded documents..."
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn  = gr.Button("Clear Chat")

            submit_btn.click(
                fn=chat_handler,
                inputs=[msg, chatbot, user_state],
                outputs=[chatbot, msg]
            )
            clear_btn.click(fn=clear_chat, outputs=chatbot)

        # ── RIGHT: Upload & Manage ────────────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("## 👤 Select User")
                user_dropdown = gr.Dropdown(
                    label="User",
                    choices=list(users.keys()),
                    value="john.doe123@example.com",
                    interactive=True
                )

            with gr.Group():
                gr.Markdown("## 📤 Upload File")
                gr.Markdown(
                    "_Supported: PDF · Word (.docx) · Excel (.xlsx) · "
                    "Images (png/jpg/jpeg/tiff/bmp/webp) · Text (txt/md/csv/json/xml/html)_"
                )
                file_input  = gr.File(
                    label="Upload File",
                    file_types=list(SUPPORTED_EXTENSIONS)
                )
                upload_btn    = gr.Button("Process File", variant="primary")
                upload_status = gr.Textbox(label="Upload Status", interactive=False)

                # Agent reasoning log — shows users what the agent decided
                with gr.Accordion("🤖 Agent Log (what the agent did)", open=False):
                    agent_log_box = gr.Textbox(
                        label="File-Type Agent Reasoning",
                        lines=6,
                        interactive=False
                    )

            with gr.Group():
                gr.Markdown("## 🗂️ Manage Documents")
                delete_checkboxes = gr.CheckboxGroup(
                    label="Select Files to Delete",
                    value=[],
                    choices=processor.get_processed_files("john.doe123@example.com"),
                    interactive=True
                )
                delete_btn    = gr.Button("Delete Selected Files", variant="stop")
                delete_status = gr.Textbox(label="Delete Status", interactive=False)

    # ── Event wiring ──────────────────────────────────────────────────────────
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
        fn=process_file,
        inputs=[file_input, file_state, user_state],
        outputs=[upload_status, agent_log_box, delete_checkboxes, file_state]
    )

    delete_btn.click(
        fn=delete_pdfs,
        inputs=[delete_checkboxes, file_state, user_state],
        outputs=[delete_status, delete_checkboxes, file_state]
    )

if __name__ == "__main__":
    demo.launch()
