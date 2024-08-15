import json
import pandas as pd
import re
import unicodedata
from tqdm import tqdm
import logging
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_text(text: str) -> str:
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C')
    
    # Remove any remaining invalid unicode
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text.strip()

def is_image_generation_request(text: str) -> bool:
    train_data = [
        "Generate an image of a cat",
        "Create a picture of a sunset",
        "Draw a landscape",
        "Make an illustration of a car",
        "What is the capital of France?",
        "Explain quantum physics",
        "Write a poem about love",
        "Calculate 15 * 7"
    ]
    train_labels = [1, 1, 1, 1, 0, 0, 0, 0] 
    
    # Training
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    classifier = MultinomialNB()
    classifier.fit(X_train, train_labels)

    X_test = vectorizer.transform([text])
    return bool(classifier.predict(X_test)[0])

def process_entry(entry: Dict) -> Dict:
    instruction = clean_text(entry['instruction'])
    input_text = clean_text(entry['input']) if entry['input'] else "no input"
    output = clean_text(entry['output'])

    if not is_image_generation_request(instruction):
        return {
            'instruction': instruction,
            'input': input_text,
            'output': output,
            'instruction_length': len(instruction),
            'input_length': len(input_text),
            'output_length': len(output)
        }
    return None

def process_dataset(file_path: str) -> pd.DataFrame:
    # Explicitly load the dataset
    logging.info(f"Loading dataset from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return pd.DataFrame()

    logging.info(f"Loaded {len(data)} entries from the dataset")

    processed_data = []
    for entry in tqdm(data, desc="Processing entries"):
        try:
            processed_entry = process_entry(entry)
            if processed_entry:
                processed_data.append(processed_entry)
        except Exception as e:
            logging.error(f"Error processing entry: {str(e)}")

    df = pd.DataFrame(processed_data)
    return df

if __name__ == "__main__":
    input_file = "alpaca_data.json" 
    output_file = "processed_alpaca_data.csv"

    logging.info(f"Starting processing of {input_file}")
    df = process_dataset(input_file)

    if df.empty:
        logging.error("No data processed. Exiting.")
    else:
        # Data cleaning
        df = df[df['output'].notna() & (df['output'] != '')]  # Remove entries with no output
        df.drop_duplicates(subset=['instruction', 'input'], inplace=True)  # Remove duplicates

        # Save the processed dataset
        df.to_csv(output_file, index=False)
        logging.info(f"Processing complete. Output saved to {output_file}")

        # Print some statistics
        logging.info(f"Total processed entries: {len(df)}")
        logging.info(f"Average output length: {df['output_length'].mean():.2f}")



        