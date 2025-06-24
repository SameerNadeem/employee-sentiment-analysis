import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from collections import Counter

class EmployeeSentimentAnalyzer:
    def __init__(self, model_path="./models/roberta_sentiment_model"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
        self.model.eval()
        
        self.sentiment_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.concern_keywords = {
            'workload': ['overworked', 'stressed', 'burnout', 'too much work', 'overwhelmed'],
            'management': ['bad manager', 'poor leadership', 'micromanage', 'unsupportive'],
            'communication': ['poor communication', 'not informed', 'unclear', 'miscommunication'],
            'work_life_balance': ['work life balance', 'long hours', 'weekend work', 'no time'],
            'compensation': ['underpaid', 'low salary', 'benefits', 'raise', 'bonus'],
            'growth': ['no growth', 'stagnant', 'promotion', 'career development', 'learning'],
            'culture': ['toxic culture', 'company culture', 'team culture', 'workplace']
        }
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        return {
            'sentiment': self.sentiment_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
                label: prob.item() 
                for label, prob in zip(self.sentiment_labels.values(), predictions[0])
            }
        }
    
    def analyze_feedback_batch(self, feedback_list):
        """Analyze sentiment for multiple feedback texts"""
        results = []
        for feedback in feedback_list:
            result = self.predict_sentiment(feedback)
            result['text'] = feedback
            results.append(result)
        return results
    
    def identify_key_concerns(self, feedback_list):
        """Identify key areas of concern from feedback"""
        concern_counts = {category: 0 for category in self.concern_keywords.keys()}
        negative_feedback = []
        
        # Analyze each feedback
        for feedback in feedback_list:
            sentiment_result = self.predict_sentiment(feedback)
            
            if sentiment_result['sentiment'] == 'Negative':
                negative_feedback.append(feedback)
                
                # Check for concern keywords
                feedback_lower = feedback.lower()
                for category, keywords in self.concern_keywords.items():
                    if any(keyword in feedback_lower for keyword in keywords):
                        concern_counts[category] += 1
        
        # Sort concerns by frequency
        sorted_concerns = sorted(
            concern_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            'concern_frequency': dict(sorted_concerns),
            'negative_feedback_count': len(negative_feedback),
            'total_feedback_count': len(feedback_list),
            'negative_percentage': (len(negative_feedback) / len(feedback_list)) * 100
        }
    
    def generate_insights_report(self, feedback_df):
        """Generate comprehensive insights report"""
        feedback_list = feedback_df['feedback'].tolist()
        
        # Analyze all feedback
        sentiment_results = self.analyze_feedback_batch(feedback_list)
        
        # Get overall sentiment distribution
        sentiment_dist = Counter([r['sentiment'] for r in sentiment_results])
        
        # Identify key concerns
        concerns = self.identify_key_concerns(feedback_list)
        
        # Calculate average confidence
        avg_confidence = np.mean([r['confidence'] for r in sentiment_results])
        
        report = {
            'total_feedback': len(feedback_list),
            'sentiment_distribution': dict(sentiment_dist),
            'sentiment_percentages': {
                sentiment: (count / len(feedback_list)) * 100 
                for sentiment, count in sentiment_dist.items()
            },
            'key_concerns': concerns['concern_frequency'],
            'negative_feedback_percentage': concerns['negative_percentage'],
            'average_model_confidence': avg_confidence,
            'top_3_concerns': list(concerns['concern_frequency'].items())[:3]
        }
        
        return report