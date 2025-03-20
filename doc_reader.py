from docling.document_converter import DocumentConverter
from pathlib import Path
import json
import os

def convert_pdf_to_markdown_and_json(source_dir, converted_dir, filename):
    try:
        # Create full path for source file
        source_path = Path(source_dir) / filename
        print("source_path", source_path)
        if not source_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {source_path}")

        # Create converted directory if it doesn't exist
        output_path = Path(converted_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize the DocumentConverter
        converter = DocumentConverter()
        print("converter done")

        # Convert the PDF file
        result = converter.convert(str(source_path))
        print("result Done")
        # Get the document filename without extension
        doc_filename = Path(filename).stem
        print("doc_filename", doc_filename)

        # Export to Markdown
        markdown_content = result.document.export_to_markdown()
        markdown_output_file = output_path / f"{doc_filename}.md"
        with open(markdown_output_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        print(f"Markdown file saved as: {markdown_output_file}")

        # Export to JSON (Workaround: Create JSON from Markdown or other data)
        # Since export_to_json() doesn't exist, use available data
        json_content = {
            "filename": doc_filename,
            "markdown_content": markdown_content,
            "source_path": str(source_path),
            "converted_at": "2025-03-19"  # You can use a dynamic date here if needed
        }
        json_output_file = output_path / f"{doc_filename}.json"
        with open(json_output_file, "w", encoding="utf-8") as f:
            json.dump(json_content, f, indent=2)
        print(f"JSON file saved as: {json_output_file}")

        # Optional: Print the markdown content preview
        print("\nMarkdown content preview:")
        print(markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    # Specify your directories and filename
    print("starting the project")
    source_directory = "source_dir"  # Replace with your source directory
    converted_directory = "converted_dir"  # Replace with your output directory
    pdf_filename = "homeInspection.pdf"  # Replace with your PDF filename

    # Convert the PDF
    convert_pdf_to_markdown_and_json(source_directory, converted_directory, pdf_filename)

if __name__ == "__main__":
    main()