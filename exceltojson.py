import pandas as pd
import json
import os

def sanitize_category(category):
    """
    Sanitize category name to be filesystem-friendly.
    Replace spaces and slashes with underscores.
    """
    return category.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")

# Read the Excel file
excel_file_path = 'news_summary.xlsx'  # replace with your actual file path
df = pd.read_excel(excel_file_path)

# Ensure the output directory exists
output_dir = 'newsdata_by_category'
os.makedirs(output_dir, exist_ok=True)

# Group the data by 'category'
grouped = df.groupby('category')

# Process each group
for category, group in grouped:
    # Convert the group to a list of dictionaries
    news_data = group.to_dict(orient='records')
    
    # Convert the list of dictionaries to JSON
    news_data_json = json.dumps(news_data, indent=4)
    
    # Sanitize category name and create a subdirectory for the category if it doesn't exist
    sanitized_category = sanitize_category(category)
    category_dir = os.path.join(output_dir, sanitized_category)
    os.makedirs(category_dir, exist_ok=True)
    
    # Save the JSON data to a file
    json_file_path = os.path.join(category_dir, f'{sanitized_category}_newsdata.json')
    with open(json_file_path, 'w') as json_file:
        json_file.write(news_data_json)
    
    print(f"JSON data for category '{category}' has been written to {json_file_path}")

print("All category JSON data has been written.")
