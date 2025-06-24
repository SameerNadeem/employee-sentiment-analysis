import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import AutoTokenizer
import torch

class FeedbackPreprocessor:
    def __init__(self, model_name="roberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text data"""
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def tokenize_data(self, texts, max_length=512):
        """Tokenize text for RoBERTa model"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def preprocess_dataset(self, df, text_column='feedback'):
        """Complete preprocessing pipeline"""
        # Clean text
        df['cleaned_feedback'] = df[text_column].apply(self.clean_text)
        
        # Remove empty feedback
        df = df[df['cleaned_feedback'].str.len() > 10]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df