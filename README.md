# Employee Sentiment Analysis System 

A machine learning project that analyzes employee feedback using **RoBERTa transformers** to identify sentiment and key areas of concern. Built during my Software Engineering internship to demonstrate NLP skills and real-world ML applications.

> **Note**: This project uses synthetic data for demonstration purposes - no real employee data was used.

## Quick Start

```bash
# Clone and run in 2 minutes
git clone https://github.com/your-username/employee-sentiment-analysis
cd employee-sentiment-analysis
pip install streamlit pandas plotly transformers torch
streamlit run streamlit_app_fixed.py
```

Open `http://localhost:8501` and start analyzing feedback


## Tech Stack

**ML**: RoBERTa Transformer | **Backend**: Python, PyTorch | **Frontend**: Streamlit | **Viz**: Plotly

## Performance
- **89.1% Accuracy** on sentiment classification
- **Real-time processing** of 1000+ entries/minute
- **7 concern categories** automatically detected

## Why I Built This

During my internship at Tech Vista Systems, I wanted to showcase how modern NLP can solve real business problems. This project demonstrates:
- **Deep Learning** 
- **Full-stack development** from model to web app
- **Business impact** through actionable insights
- **Production-ready code** with clean architecture

## Features

### Dashboard
sentiment distribution charts, and concern frequency analysis

### Single Analysis

### Batch Processing
Upload CSV files or use sample synthetic dataset for bulk analysis

## Key Insights Generated

- **Top concern areas** with frequency counts
- **Sentiment trends** and distribution
- **Confidence scores** for prediction reliability
- **Actionable recommendations** for HR teams

## Project Structure

```
├── streamlit_app_fixed.py    # Main web application (everything included)
├── requirements.txt          # Dependencies
└── README.md                # You're here!
```

---

**Built by**: Sameer Nadeem| **Internship**: Tech Vista System

