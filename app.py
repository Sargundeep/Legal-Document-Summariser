import pandas as pd
import os

# Specify the folder containing the text files and the path to the Excel file.
text_folder = "text"
excel_file = "text_labels.csv"

# Read the Excel file into a DataFrame.
df = pd.read_csv(excel_file)

# Function to read the text content from a file and preprocess it.
def read_text_content(file_name):
    file_name = str(file_name) + ".txt"
    file_path = os.path.join(text_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        # Additional preprocessing steps:
        
        # 1. Convert text to lowercase
        content = content.lower()
        
        # 2. Remove special characters and punctuation
        import string
        content = content.translate(str.maketrans('', '', string.punctuation))
        
        # 3. Tokenize the text (split it into words)
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(content)
        content = ' '.join(tokens)
        
        # 4. Remove stop words (common words that may not be relevant)
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]
        content = ' '.join(tokens)

    return content

# Apply the function to create a new column with the entire content as one string.
df['Entire_Content'] = df['file_number'].apply(read_text_content)
print(df.head)

output_csv_file = "output.csv"
df.to_csv(output_csv_file, index=False)

