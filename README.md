# Document Reader with Qdrant

A Gradio-based web application that allows users to upload PDF documents, process them with semantic chunking, store embeddings in Qdrant, and perform intelligent document search and Q&A using OpenAI's language models.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Guardrails](#guardrails)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Features

- **PDF Upload & Processing**: Upload PDF files with automatic text extraction and semantic chunking.
- **Vector Embeddings**: Generate embeddings using OpenAI's `text-embedding-3-small` model and store in Qdrant.
- **Multi-User & Organization Support**: Role-based access control with org-based document filtering.
- **Semantic Search**: Find relevant document passages using vector similarity search.
- **Document Q&A**: Ask questions about uploaded documents and get AI-powered answers with source attribution.
- **Input Validation & Guardrails**: File type/size validation, jailbreak detection, and output safety checks.
- **Duplicate Detection**: Content-hash based deduplication to prevent redundant uploads.
- **Document Management**: Delete and manage uploaded documents with proper cleanup.

---

## Architecture

```
┌──────────────────┐
│   Gradio UI      │
│  (Upload/Chat)   │
└────────┬─────────┘
         │
         ├─────────────────────────────────────┐
         │                                     │
         v                                     v
┌──────────────────────┐         ┌─────────────────────┐
│ DocumentProcessor    │         │  Search/Chat Path   │
│  - Validate file     │         │                     │
│  - Extract text      │         │  - Retrieve context │
│  - Chunk (semantic)  │         │  - LLM call         │
│  - Embed chunks      │         │  - Append sources   │
│  - Dedupe check      │         └─────────────────────┘
└──────────────┬───────┘
               │
               v
        ┌──────────────┐
        │ OpenAI APIs  │
        │  - Embeddings│
        │  - LLM (GPT) │
        └──────┬───────┘
               │
               v
        ┌──────────────┐
        │    Qdrant    │
        │  Vector DB   │
        └──────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- Qdrant instance (local Docker or cloud)
- OpenAI API key

### Setup Steps

1. **Clone/Navigate to project directory**:
   ```bash
   cd /Users/rajatbaid/Documents/code/Projects
   ```

2. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

   Or manually:
   ```bash
   pip install gradio qdrant-client langchain langchain-openai python-dotenv spacy nltk PyPDF2
   ```

4. **Download spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Set up environment variables** (create `.env` file):
   ```env
   OPENAI_API_KEY=sk-...
   QDRANT_CLUSTER=<api-key>  # If using cloud Qdrant
   MAX_UPLOAD_MB=20
   MAX_PDF_PAGES=500
   ENABLE_PII_REDACTION=true
   ```

6. **Start Qdrant** (if using local):
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

7. **Run the application**:
   ```bash
   python DocumentReader.py
   ```

   The app will launch at `http://localhost:7860`

---

## Configuration

Configure behavior via environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key for embeddings and LLM |
| `MAX_UPLOAD_MB` | 20 | Max file size for PDF uploads (MB) |
| `MAX_PDF_PAGES` | 500 | Max pages allowed per PDF |
| `MAX_CHUNKS` | 2000 | Max chunks to embed per document |
| `ENABLE_PII_REDACTION` | true | Enable PII masking before chunking |
| `QDRANT_CLUSTER` | Optional | Qdrant cloud cluster API key |

**LLM Configuration** (hardcoded in code):
- Model: `gpt-3.5-turbo` (adjust `OPENAI_LLM_MODEL`)
- Temperature: `0.2` (low randomness for consistency)
- Embedding Model: `text-embedding-3-small`
- Vector Size: `1536`

---

## Usage

### 1. Start the Application

```bash
source venv/bin/activate
python DocumentReader.py
```

### 2. Select User

Use the dropdown to select a user. Each user sees only documents from their organization.

**Predefined Users**:
- `john.doe123@example.com` (Org1)
- `jane.smith456@example.com` (Org1)
- `bob.jones789@example.com` (Org2)
- `alice.brown321@example.com` (Org2)
- `mike.wilson654@example.com` (Org2)
- `admin@example.com` (Admin – sees all documents)

### 3. Upload PDF

1. Click **"Upload PDF"** and select a `.pdf` file.
2. Click **"Process PDF"** button.
3. Status message shows success or error (file too large, invalid, etc.).
4. Document appears in **"Manage Documents"** section after successful upload.

### 4. Chat with Documents

1. Type a question in the **"Your Message"** textbox.
2. Click **"Submit"** or press Enter.
3. Response appears in the chatbot with sources listed.
4. Click **"Clear Chat"** to reset conversation history.

### 5. Delete Documents

1. Check the document(s) you want to delete in **"Manage Documents"**.
2. Click **"Delete Selected PDFs"**.
3. Status confirms deletion; the file list updates.

---

## Guardrails

### Input Guardrails

#### 1. File Validation
- **Type check**: Only `.pdf` files allowed.
- **Size check**: Default max `20 MB` (configurable).
- **Page count**: Default max `500` pages (configurable).

#### 2. Jailbreak Detection
User input is scanned for potentially malicious patterns:
- `"ignore previous instructions"`
- `"you are now"` / `"act as"`
- `"jailbreak"` / `"DAN"`
- `"reveal your"` / `"show me the code"`
- `"list all users"` / `"delete document"`
- And more...

If detected, response is: _"I'm sorry, I can't help with that request."_

#### 3. Duplicate Detection
Content hashes prevent re-uploading identical documents.

### Output Guardrails

#### 1. Strict System Prompt
The LLM is instructed to:
- Use only provided document context.
- Say "I don't have information about that..." if answer not found.
- Never reveal other users' documents or system internals.
- Never make up information.

#### 2. Source Attribution
All responses include a `Sources:` section listing document files used to answer the question.

#### 3. Response Safety
- Responses are checked for banned terms (e.g., "kill", "bomb", "terror").
- Maximum response length: `400` tokens (configurable).

---

## API Reference

### Core Functions

#### `DocumentProcessor` Class

**`__init__(...)`**
Initialize the document processor with Qdrant connection and embedding config.

**`process_document(text, source_file, user_email)`**
- Extract, chunk, embed, and upsert a document.
- Returns: Success/error message.
- **Checks**: Dedupe, chunk count limit, user permissions.

**`delete_by_source_file(source_file)`**
- Delete all embeddings for a file.
- Returns: 0 if successful, -1 on error.

**`search(query_vector, user_email, limit=5)`**
- Semantic search filtered by user's organization.
- Returns: List of top-k passages with metadata.

**`get_processed_files(user_email)`**
- Return files visible to the user (org-filtered).
- Admin sees all; regular users see only their org's files.

---

### UI Callbacks

#### `process_pdf(file, current_files, selected_user_email)`
Handles PDF upload: validates, extracts, processes, and upserts to Qdrant.

**Returns**: `(status_message, updated_checkbox, file_state)`

#### `delete_pdfs(filenames, current_files, selected_user_email)`
Delete selected PDFs and update UI.

**Returns**: `(status_message, updated_checkbox, file_state)`

#### `chatbot_response(message, history, selected_user_email)`
Process user query, search documents, call LLM, and return answer with sources.

**Returns**: Response string (with sources appended).

#### `chat_handler(message, history, selected_user_email)`
Wrapper that appends user message and assistant response to history.

**Returns**: `(updated_history, "")`

#### `search_qdrant(query, selected_user_email)`
Perform semantic search on user's documents.

**Returns**: `(context_string, set_of_source_files)`

---

### Guardrail Functions

#### `is_potentially_malicious(question: str) -> bool`
Checks if input matches known jailbreak/injection patterns.

#### `validate_pdf_file(file) -> (bool, str)`
Validates file type, size, and page count. Returns `(is_valid, error_msg)`.

---

## Deployment

### Local Development

```bash
python DocumentReader.py
# Access at http://localhost:7860
```

### Production Deployment (Uvicorn)

```bash
pip install uvicorn
uvicorn DocumentReader:app --host 0.0.0.0 --port 7860
```

### Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt && python -m spacy download en_core_web_sm
COPY DocumentReader.py .
CMD ["python", "DocumentReader.py"]
```

Build and run:
```bash
docker build -t document-reader .
docker run -p 7860:7860 --env-file .env document-reader
```

### Cloud Qdrant

1. Create a cluster on [qdrant.io](https://qdrant.io).
2. Set `QDRANT_CLUSTER` API key in `.env`.
3. Modify `DocumentProcessor.__init__` to use cloud URL instead of `localhost:6333`.

---

## Troubleshooting

### "OPENAI_API_KEY not set"
- Ensure `.env` file exists in project root.
- Run: `source venv/bin/activate` before running the app.
- Verify key is valid in OpenAI dashboard.

### "Could not connect to Qdrant"
- Ensure Qdrant is running: `docker ps` should show Qdrant container.
- If using cloud, verify `QDRANT_CLUSTER` is set and accessible.
- Check firewall/network settings.

### "File too large" / "PDF too long"
- Adjust `MAX_UPLOAD_MB` and `MAX_PDF_PAGES` in `.env`.
- Or split large PDFs into smaller files.

### "No chunks generated"
- PDF may be image-scanned (no extractable text).
- Use OCR tools to convert scanned PDFs to text first.

### Slow uploads
- Large PDFs take time to chunk and embed.
- Consider splitting into smaller documents.
- Check OpenAI API quota and rate limits.

### "I don't have information about that..."
- The question is not answerable from uploaded documents.
- Upload relevant documents or rephrase the question.

### Admin can't see other users' documents
- This is intentional (privacy by design).
- Only admins can see org-filtered results; non-admins see only their org's docs.

---

## Security Considerations

1. **Never commit `.env`** with real API keys.
2. **Use Vault or cloud secrets manager** in production.
3. **Enable HTTPS** when deployed publicly.
4. **Audit logs**: Log all uploads, deletions, and queries for compliance.
5. **Rate limiting**: Implement Redis-based rate limits on production.
6. **PII redaction**: Enable `ENABLE_PII_REDACTION=true` by default.

---

## Future Enhancements

- [ ] Add Redis for rate limiting and caching.
- [ ] Integrate OpenAI Moderation API for output safety.
- [ ] Add advanced hallucination detection (fact verification per claim).
- [ ] Support for other file formats (DOCX, TXT, etc.).
- [ ] Background job queue (Celery) for large uploads.
- [ ] Multi-language support.
- [ ] Fine-tuned models for domain-specific Q&A.
- [ ] Analytics dashboard (usage, costs, performance metrics).

---

## License

Proprietary – Internal use only.

---

## Support

For issues or questions, contact the development team.
