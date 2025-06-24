import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

class SentimentVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = {
            'Positive': '#2ecc71',
            'Neutral': '#f39c12',
            'Negative': '#e74c3c'
        }
    
    def plot_sentiment_distribution(self, sentiment_data):
        """Create sentiment distribution pie chart"""
        fig = go.Figure(data=[go.Pie(
            labels=list(sentiment_data.keys()),
            values=list(sentiment_data.values()),
            marker_colors=['#e74c3c', '#f39c12', '#2ecc71'],
            textinfo='label+percent',
            hole=0.3
        )])
        
        fig.update_layout(
            title="Employee Sentiment Distribution",
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    def plot_concern_analysis(self, concerns_data):
        """Create horizontal bar chart for key concerns"""
        concerns = list(concerns_data.keys())
        counts = list(concerns_data.values())
        
        fig = go.Figure([go.Bar(
            x=counts,
            y=concerns,
            orientation='h',
            marker_color='#3498db',
            text=counts,
            textposition='outside'
        )])
        
        fig.update_layout(
            title="Key Areas of Concern (Frequency)",
            xaxis_title="Number of Mentions",
            yaxis_title="Concern Categories",
            height=400
        )
        
        return fig
    
    def create_dashboard(self, insights_report):
        """Create comprehensive dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Sentiment Distribution", 
                "Key Concerns", 
                "Confidence Analysis",
                "Monthly Trends"
            ),
            specs=[
                [{"type": "pie"}, {"type": "bar"}],
                [{"type": "histogram"}, {"type": "scatter"}]
            ]
        )
        
        # Sentiment pie chart
        sentiment_data = insights_report['sentiment_distribution']
        fig.add_trace(go.Pie(
            labels=list(sentiment_data.keys()),
            values=list(sentiment_data.values()),
            name="Sentiment"
        ), row=1, col=1)
        
        # Concerns bar chart
        concerns = insights_report['key_concerns']
        top_concerns = dict(list(concerns.items())[:5])
        fig.add_trace(go.Bar(
            x=list(top_concerns.values()),
            y=list(top_concerns.keys()),
            orientation='h',
            name="Concerns"
        ), row=1, col=2)
        
        fig.update_layout(height=800, showlegend=False)
        return fig