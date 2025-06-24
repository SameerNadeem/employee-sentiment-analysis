import torch
import torch.nn as nn
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class RoBERTaSentimentClassifier(nn.Module):
    def __init__(self, model_name="roberta-base", num_classes=3):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class SentimentTrainer:
    def __init__(self, model_name="roberta-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = RoBERTaSentimentClassifier(model_name)
        
    def train_model(self, train_texts, train_labels, val_texts, val_labels):
        """Train the sentiment analysis model"""
        
        # Create datasets
        train_dataset = SentimentDataset(
            train_texts, train_labels, self.tokenizer
        )
        val_dataset = SentimentDataset(
            val_texts, val_labels, self.tokenizer
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./models/roberta_sentiment_model',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained('./models/roberta_sentiment_model')
        
        return trainer
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            'accuracy': accuracy_score(labels, predictions),
            'f1': f1_score(labels, predictions, average='weighted')
        }