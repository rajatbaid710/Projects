from docling.document_converter import DocumentConverter
from pathlib import Path
import json
import os
from extractor import PdfExtractor
from chunking import ChunkGenerator

def main():
    # Specify your directories and filename
    print("starting the project")
    source_directory = "source_dir"  # Replace with your source directory
    converted_directory = "converted_dir"  # Replace with your output directory
    pdf_filename = "homeInspection.pdf"  # Replace with your PDF filename

    # Convert the PDF
    # extractor = PdfExtractor()
    # extractor.extract_pdfs(source_directory,converted_directory)

    # chunk generator
    chunker = ChunkGenerator(chunk_size=1000, chunk_overlap=200)
    chunk_input_dir = converted_directory
    chunk_output_dir = "chunked_docs"
    # Process the files
    chunker.chunk_json_files(chunk_input_dir, chunk_output_dir)


if __name__ == "__main__":
    main()
