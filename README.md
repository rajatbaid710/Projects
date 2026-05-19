# Agentic Document Reader with Qdrant

A Gradio-based web application that lets users upload **PDF, Word, Excel, image, or text files**. An LLM-driven **File-Type Agent** automatically picks the right reader tool for each file, extracts the text, generates embeddings with OpenAI, stores them in Qdrant, and enables multi-user, organization-scoped semantic Q&A over the uploaded documents.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [How the File-Type Agent Works](#how-the-file-type-agent-works)
- [Supported File Types](#supported-file-types)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Multi-format Ingestion**: Upload PDF, Word (`.docx`), Excel (`.xlsx`/`.xls`), images (with OCR), and plain-text files.
- **Agentic File Reading**: An OpenAI tool-calling agent inspects each file and dynamically picks the right reader (PyPDF2, python-docx, openpyxl, pytesseract OCR, or plain-text).
- **Agent Reasoning Log**: A collapsible panel in the UI shows exactly which tool the agent chose and how much text was extracted.
- **Semantic Chunking**: Splits text using spaCy sentence boundaries (or recursive character splitting) before embedding.
- **Vector Embeddings**: Generates embeddings with OpenAI's `text-embedding-3-small` model and stores them in Qdrant (1536-dim, cosine distance).
- **Multi-User & Organization Scoping**: Users belong to organizations; queries and document listings are filtered by org. An `admin@example.com` user can see everything.
- **Document Q&A Chatbot**: Ask questions about uploaded documents and get LLM answers with **source attribution** (filename list appended to each response).
- **Relevance Filtering**: Only files within `0.15` of the top similarity score contribute to the answer context, reducing noise from weakly related documents.
- **Duplicate Detection**: SHA-256 content hashing prevents re-uploading the same content twice for the same user.
- **Document Management**: Per-user file list with multi-select deletion.

---

## Architecture

```
┌──────────────────┐
│   Gradio UI      │
│ Upload · Chat ·  │
│ Manage · UserSel │
└────────┬─────────┘
         │
         │  file
         v
┌──────────────────────────────┐
│   File-Type Agent (LLM)      │
│  Picks ONE tool via          │
│  OpenAI function-calling:    │
│  read_pdf · read_word ·      │
│  read_excel · read_image ·   │
│  read_text                   │
└──────────────┬───────────────┘
               │ extracted text
               v
┌──────────────────────────────┐
│      DocumentProcessor       │
│  - Dedupe (sha256)           │
│  - Chunk (semantic/recursive)│
│  - Embed (OpenAI)            │
│  - Upsert to Qdrant          │
└──────────────┬───────────────┘
               │
               v
        ┌──────────────┐
        │    Qdrant    │
        │  Vector DB   │
        └──────┬───────┘
               │ top-k passages
               v
┌──────────────────────────────┐
│  Chat path: Org-filtered     │
│  retrieval → LLM → response  │
│  + Sources line              │
└──────────────────────────────┘
```

---

## How the File-Type Agent Works

The core agentic pattern is **Observe → Decide → Act → Return**:

1. **Observe** – The agent receives the uploaded filename and path.
2. **Decide** – An OpenAI Chat Completions call with `tool_choice="required"` forces the LLM to pick exactly one of five available tools (`read_pdf`, `read_word`, `read_excel`, `read_image`, `read_text`) based on the file extension.
3. **Act** – The chosen tool is dispatched through `TOOL_FUNCTIONS` and executes on the file path.
4. **Return** – Extracted text plus a human-readable reasoning log is returned to the UI.

If the LLM somehow returns no tool call, the agent falls back to `read_text`.

---

## Supported File Types

| Category | Extensions | Reader |
|----------|------------|--------|
| PDF | `.pdf` | PyPDF2 |
| Word | `.docx`, `.doc` | python-docx |
| Excel | `.xlsx`, `.xls` | openpyxl |
| Images | `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` | pytesseract (OCR) |
| Text | `.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html` | plain read |

> **Note**: OCR requires the **Tesseract** binary to be installed on the host (see [Installation](#installation)).

---

## Installation

### Prerequisites

- Python 3.8+
- A running Qdrant instance (local Docker or cloud)
- An OpenAI API key
- **Tesseract OCR** binary (only required for image uploads)

### Setup Steps

1. **Navigate to the project directory**:
   ```bash
   cd /Users/rajatbaid/Documents/code/Projects
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy English model** (used for semantic chunking):
   ```bash
   python -m spacy download en_core_web_sm
   ```
   The app will also try to auto-download this on first run if missing.

5. **Install Tesseract OCR** (only needed for image files):
   - macOS: `brew install tesseract`
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - Windows: download installer from the [Tesseract releases page](https://github.com/UB-Mannheim/tesseract/wiki).

6. **Create a `.env` file** in the project root:
   ```env
   OPENAI_API_KEY=sk-...
   OPENAI_LLM_MODEL=gpt-3.5-turbo
   ```

7. **Start Qdrant locally** (if not using a cloud cluster):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

8. **Run the application**:
   ```bash
   python AgenticDocReader.py
   ```
   The app launches at `http://localhost:7860`.

---

## Configuration

Configure behavior via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | **Required** | OpenAI API key used for the file-type agent, embeddings, and chat LLM. |
| `OPENAI_LLM_MODEL` | `gpt-3.5-turbo` | Chat model used by the file-type agent and the document Q&A chatbot. |

**Hardcoded settings** (edit `AgenticDocReader.py` to change):

| Setting | Value | Location |
|---------|-------|----------|
| Qdrant host / port | `localhost` / `6333` | `DocumentProcessor.__init__` |
| Collection name | `aiml_vector_db` | `DocumentProcessor.__init__` |
| Chunking strategy | `semantic` (spaCy sentences) | `DocumentProcessor.__init__` |
| Chunk size / overlap | `800` / `150` | `DocumentProcessor.__init__` |
| Embedding model | `text-embedding-3-small` | `DocumentProcessor.__init__` |
| Vector size / distance | `1536` / cosine | `DocumentProcessor._create_collection` |
| Embedding batch size | `32` | `DocumentProcessor.__init__` |
| LLM temperature | `0.2` | `ChatOpenAI` instantiation |
| Relevance window | top score − `0.15` | `search_qdrant` |
| Retrieval `top_k` | `5` | `DocumentProcessor.search` |

---

## Usage

### 1. Start the application

```bash
source venv/bin/activate
python AgenticDocReader.py
```

### 2. Pick a user

Use the **User** dropdown to switch identities. Each user sees only documents belonging to their organization; `admin@example.com` sees every file.

**Predefined users**:

| Email | Org |
|-------|-----|
| `john.doe123@example.com`  | Org1 |
| `jane.smith456@example.com` | Org1 |
| `bob.jones789@example.com`  | Org2 |
| `alice.brown321@example.com` | Org2 |
| `mike.wilson654@example.com` | Org2 |
| `admin@example.com` | _(all orgs)_ |

### 3. Upload a file

1. Drag-and-drop or pick a file in **"Upload File"** (any [supported type](#supported-file-types)).
2. Click **"Process File"**.
3. The **Upload Status** box shows success/failure.
4. Expand **"Agent Log"** to see which reader tool the agent chose and how many characters it extracted.

### 4. Chat with your documents

1. Type a question in **"Your Message"**.
2. Click **"Submit"**.
3. The chatbot answers using only the retrieved org-scoped context and appends a `Sources: ...` line listing every contributing file.
4. Use **"Clear Chat"** to reset conversation history.

### 5. Delete documents

1. Select files in **"Select Files to Delete"**.
2. Click **"Delete Selected Files"**.
3. Status confirms which files were removed; the list refreshes automatically.

---

## API Reference

### File-Type Agent

**`file_type_agent(file_path, filename, llm_client) -> (text, log)`**
Runs the OpenAI tool-calling loop to pick and execute one of the `FILE_TOOLS`. Returns the extracted text plus a multi-line reasoning log shown in the UI.

**Reader implementations**:
- `read_pdf(file_path)` – text extraction via PyPDF2.
- `read_word(file_path)` – paragraph and table-cell extraction via python-docx.
- `read_excel(file_path)` – per-sheet tab-separated dump via openpyxl.
- `read_image(file_path)` – OCR via pytesseract.
- `read_text(file_path)` – UTF-8 read with error-ignoring decoder.

### `DocumentProcessor` class

| Method | Description |
|--------|-------------|
| `__init__(...)` | Connects to Qdrant, ensures the collection exists, configures the splitter (`semantic` or `recursive_char`), wires up OpenAI embeddings, and pre-loads the set of already-processed files. |
| `process_document(text, source_file, user_email)` | Hashes the content, skips duplicates, splits into chunks, embeds in batches of 32, and upserts to Qdrant with payload `{text, source_file, user_email, content_hash, org}`. |
| `delete_by_source_file(source_file)` | Deletes all points whose `source_file` matches. Returns the number of remaining points (or `-1` on error). |
| `search(query_vector, user_email, limit=5)` | Vector search filtered by the user's `org` (admin sees all). |
| `get_processed_files(user_email)` | Returns the file list visible to the user. |

### UI callbacks (Gradio)

| Callback | Purpose |
|----------|---------|
| `process_file(file, current_files, user_email)` | Validates the extension, runs the file-type agent, calls `process_document`, refreshes the file list. |
| `delete_pdfs(filenames, current_files, user_email)` | Deletes selected files (with ownership/org checks; admin bypasses ownership). |
| `chatbot_response(message, history, user_email)` | Runs retrieval, calls the chat LLM, and appends a `Sources:` line. |
| `chat_handler(message, history, user_email)` | Normalizes Gradio chat history into the OpenAI message format and updates the chatbot state. |
| `search_qdrant(query, user_email)` | Embeds the query, retrieves top-k, applies the `0.15` relevance window, and returns `(context, source_files)`. |
| `update_file_list(user_email)` | Refreshes the file checkbox group when the active user changes. |
| `clear_chat()` | Resets chat history. |

---

## Deployment

### Local development

```bash
python AgenticDocReader.py
# http://localhost:7860
```

### Docker

Example `Dockerfile`:

```dockerfile
FROM python:3.10-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -m spacy download en_core_web_sm

COPY AgenticDocReader.py .
EXPOSE 7860
CMD ["python", "AgenticDocReader.py"]
```

Build and run:

```bash
docker build -t agentic-doc-reader .
docker run -p 7860:7860 --env-file .env agentic-doc-reader
```

### Cloud Qdrant

1. Create a cluster on [qdrant.io](https://qdrant.io).
2. Update the `QdrantClient(host=..., port=...)` call in `DocumentProcessor.__init__` to point at your cluster URL and pass the API key.

---

## Troubleshooting

### `OPENAI_API_KEY` is missing or invalid
- Make sure `.env` is in the project root and the venv is activated.
- Re-issue the key in the OpenAI dashboard if requests 401.

### Cannot connect to Qdrant
- Check that the container is running: `docker ps | grep qdrant`.
- The app defaults to `localhost:6333`; update `DocumentProcessor.__init__` if Qdrant is remote.

### `spaCy` model `en_core_web_sm` not found
- Run `python -m spacy download en_core_web_sm`. The app also attempts this download automatically on first launch.

### Image OCR returns empty text
- Confirm Tesseract is installed and on `PATH` (`tesseract --version`).
- Low-resolution or heavily skewed images may need pre-processing before OCR.

### "Very little text extracted" warning in the Agent Log
- The file may be a scanned PDF (no text layer), an empty document, or an unsupported binary payload.
- For scanned PDFs, convert pages to images first and re-upload them as images so the OCR tool is used.

### Duplicate content detected
- The content hash matched an existing upload for the same user. Modify the content or upload from a different user account.

### Admin can't see another user's files
- Admin queries are unfiltered by org, but the file list shown in the UI is `processor.get_processed_files("admin@example.com")` — verify those files were actually upserted (e.g., the user uploaded under a non-admin account).

---

## Future Enhancements

- [ ] Input guardrails (jailbreak/prompt-injection detection on chat queries).
- [ ] Output safety filters (e.g., OpenAI Moderation API).
- [ ] File-size and page-count limits with user feedback.
- [ ] PII redaction before embedding.
- [ ] Persistent ID generation so `next_id` survives restarts.
- [ ] Background job queue for large uploads.
- [ ] Multi-language spaCy models for non-English semantic chunking.

---

## License

Proprietary – internal use only.
