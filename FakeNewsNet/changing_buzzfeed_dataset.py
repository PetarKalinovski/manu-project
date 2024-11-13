import os
import json
import pandas as pd

# Directory containing the JSON files
json_directory = 'RealNewsContent'

# List to store the text content and URLs
data = []

# Loop through each JSON file
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        file_path = os.path.join(json_directory, filename)

        # Open and read the JSON file
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)

            # Extract the 'text' and 'top_img' fields
            text = json_data.get('text', '')
            url = json_data.get('url', '')

            # Append as a tuple to the list
            data.append((text, url))

# Create a pandas DataFrame with two columns: 'text' and 'url'
df = pd.DataFrame(data, columns=['text', 'url'])
df["type"]=0
df.to_csv('RealNewsContent.csv', index=False)