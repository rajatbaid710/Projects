import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ChunkGenerator:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize the ChunkGenerator with chunking parameters.

        Args:
            chunk_size (int): Target size of each chunk in characters
            chunk_overlap (int): Number of overlapping characters between chunks
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_json_files(self, input_dir, output_dir):
        """
        Split all JSON files from input directory into chunks.

        Args:
            input_dir (str): Directory containing input JSON files
            output_dir (str): Base directory to save chunked files
        """
        # Create base output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get all JSON files from input directory
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

        if not json_files:
            print(f"No JSON files found in {input_dir}")
            return

        total_processed = 0

        # Process each JSON file
        for json_file in json_files:
            total_processed += self._process_single_file(json_file, input_dir, output_dir)

        print(f"\nCompleted: Processed {total_processed} out of {len(json_files)} JSON files")

    def _process_single_file(self, json_file, input_dir, output_dir):
        """
        Process a single JSON file and return 1 if successful, 0 if failed.
        """
        input_path = os.path.join(input_dir, json_file)
        file_base_name = os.path.splitext(json_file)[0]
        file_output_dir = os.path.join(output_dir, file_base_name + "_chunks")

        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        try:
            # Read the JSON file
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert JSON to string for splitting
            json_string = json.dumps(data)

            # Split the text
            chunks = self.text_splitter.split_text(json_string)

            # Process each chunk
            for i, chunk in enumerate(chunks, 1):
                self._write_chunk(chunk, file_output_dir, i)

            print(f"\nProcessed {json_file}: split into {len(chunks)} chunks")
            return 1

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            return 0

    def _write_chunk(self, chunk, output_dir, chunk_number):
        """
        Write a single chunk to a file.
        """
        try:
            # Try to parse chunk back to JSON
            chunk_data = json.loads(chunk)
        except json.JSONDecodeError:
            # If parsing fails, store as a string in a simple JSON structure
            chunk_data = {"content": chunk}

        output_file = os.path.join(output_dir, f'chunk_{chunk_number}.json')

        # Write chunk to file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=4)

        print(f"Created {output_file} with size {len(chunk)} characters")

# Example usage
# if __name__ == "__main__":
#     # Install required package: pip install langchain
#
#     # Initialize the ChunkGenerator
#     chunker = ChunkGenerator(chunk_size=1000, chunk_overlap=200)
#
#     # Define directories
#     input_directory = "input_json_files"
#     output_directory = "output_chunks"
#
#     # Process the files
#     chunker.chunk_json_files(input_directory, output_directory)