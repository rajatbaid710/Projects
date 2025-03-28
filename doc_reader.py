from extractor import PdfExtractor
from chunking import ChunkGenerator
from embedding import DocumentEmbedder
from storeToQdrant import QdrantUploader
from qdrant_client.http.models import Distance, VectorParams, PointStruct


def main():
    print("starting the project")
    source_directory = "source_dir"
    converted_directory = "converted_dir"
    process_dir = "processed_files"
    pdf_filename = "homeInspection.pdf"

    # Convert the PDF
    convert_pdf_json(source_directory, converted_directory, process_dir)

    # Chunk generator
    chunk_output_dir = "chunked_docs"
    generate_chunk(converted_directory, chunk_output_dir)

    # Embedding generator
    embedding_output_dir = "embedded_docs"
    embedding_input_dir = "chunked_docs"
    generate_embedding(embedding_input_dir, embedding_output_dir)

    # Store to Qdrant
    embedded_docs_dir = "embedded_docs"
    collection_name = "aiml_vector_db"
    vector_size = 1536
    storeToQdrant(embedded_docs_dir, collection_name, vector_size)


def convert_pdf_json(source_directory, converted_directory, process_dir):
    extractor = PdfExtractor()
    extractor.extract_pdfs(source_directory, converted_directory, process_dir)


def generate_chunk(input_dir, output_dir):
    chunker = ChunkGenerator(chunk_size=1000, chunk_overlap=200)
    # Process the files
    chunker.chunk_document(input_dir, output_dir)


def generate_embedding(input_dir, output_dir):
    embedder = DocumentEmbedder()
    embedder.embed_chunks(input_dir, output_dir)


def storeToQdrant(embedded_docs_dir, collection_name, vector_size):
    uploader = QdrantUploader()
    # Initialize QdrantUploader
    uploader = QdrantUploader(
        collection_name=collection_name,
        qdrant_host="localhost",  # Adjust if using a remote Qdrant instance
        qdrant_port=6333
    )

    # Create collection (adjust vector_size if using a different embedding model)
    uploader.create_collection(vector_size=vector_size, distance=Distance.COSINE)

    # Upload embeddings
    uploader.upload_embeddings(embedded_docs_dir)


if __name__ == "__main__":
    main()
