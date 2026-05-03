import pandas as pd
import os

# Hardcode the Excel file path here
excel_file = "/Users/rajatbaid/Documents/data/NutritionDataSetxlsx.xlsx"  # Replace with your actual Excel file path

# Generate output JSON file path in the same folder
json_file = os.path.splitext(excel_file)[0] + '.json'

# Read the Excel file
df = pd.read_excel(excel_file)

# Convert to JSON
df.to_json(json_file, orient='records', indent=4)

print(f"Successfully converted {excel_file} to {json_file}")
