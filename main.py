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
    original_outputs = [clean_text(item