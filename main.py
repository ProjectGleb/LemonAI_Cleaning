import json
import pandas as pd
import re
import unicodedata
import torch
import logging
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # or another suitable model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
model.eval()

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

    # Training the model
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_data)
    classifier = MultinomialNB()
    classifier.fit(X_train, train_labels)

    X_test = vectorizer.transform([text])
    return bool(classifier.predict(X_test)[0])


def process_batch(batch: List[Dict]) -> List[Dict]:
    instructions = [clean_text(item['instruction']) for item in batch]
    inputs = [clean_text(item['input']) if item['input'] else "no input" for item in batch]
    original_outputs = [clean_text(item['output']) for item in batch]

    prompts = [
        f"Instruction: {instr}\nInput: {inp}\nOriginal Output: {out}\n\nProvide an enhanced and detailed response:"
        for instr, inp, out in zip(instructions, inputs, original_outputs)
    ]

    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(**tokenized, max_new_tokens=256, num_return_sequences=1)

    enhanced_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    processed_batch = []
    for item, enhanced_output in zip(batch, enhanced_outputs):
        if not is_image_generation_request(item['instruction']):
            processed_item = {
                'instruction': clean_text(item['instruction']),
                'input': clean_text(item['input']) if item['input'] else "no input",
                'output': clean_text(enhanced_output),
                'instruction_length': len(item['instruction']),
                'input_length': len(item['input']) if item['input'] else 7,  # length of "no input"
                'output_length': len(enhanced_output)
            }
            processed_batch.append(processed_item)

    return processed_batch

def process_dataset(file_path: str, num_examples: int = 10, batch_size: int = 8) -> pd.DataFrame:
    logging.info(f"Loading dataset from {file_path}")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return pd.DataFrame()
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        return pd.DataFrame()

    logging.info(f"Processing first {num_examples} entries from the dataset")

    processed_data = []
    for i in range(0, min(num_examples, len(data)), batch_size):
        batch = data[i:min(i+batch_size, num_examples)]
        try:
            processed_batch = process_batch(batch)
            processed_data.extend(processed_batch)
        except Exception as e:
            logging.error(f"Error processing batch {i//batch_size}: {str(e)}")

    df = pd.DataFrame(processed_data)
    return df

if __name__ == "__main__":
    input_file = "alpaca_data.json"
    output_file = "processed_alpaca_data_10_examples.csv"

    logging.info(f"Starting processing of {input_file}")
    df = process_dataset(input_file, num_examples=10)

    if df.empty:
        logging.error("No data processed. Exiting.")
    else:
        # Clean the data
        df = df[df['output'].notna() & (df['output'] != '')]  # Remove entries with no output
        df.drop_duplicates(subset=['instruction', 'input'], inplace=True)  # Remove duplicates

        # Save the processed dataset
        df.to_csv(output_file, index=False)
        logging.info(f"Processing complete. Output saved to {output_file}")

        # Print some statistics
        logging.info(f"Total processed entries: {len(df)}")
        logging.info(f"Average output length: {df['output_length'].mean():.2f}")
