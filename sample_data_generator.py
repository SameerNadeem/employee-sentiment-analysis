# Create sample_data_generator.py
import pandas as pd
import numpy as np

def create_sample_data():
    np.random.seed(42)
    
    positive_feedback = [
        "Great work environment and supportive colleagues",
        "Love the company culture and growth opportunities",
        "Excellent management and clear communication",
        "Perfect work-life balance and flexible hours",
        "Competitive salary and amazing benefits package",
        "Team collaboration is outstanding",
        "Professional development opportunities are excellent",
        "Office facilities are modern and comfortable"
    ]
    
    neutral_feedback = [
        "The job is okay, nothing special but not bad either",
        "Work is standard, meets expectations",
        "Company policies are reasonable",
        "Average workplace with normal procedures",
        "Decent benefits package",
        "Standard work environment",
        "Regular team meetings and updates"
    ]
    
    negative_feedback = [
        "Management needs to improve communication significantly",  
        "Work-life balance could be much better, too many late nights",
        "Salary is below market rate for this position",
        "Poor leadership and micromanagement issues",
        "Stressful work environment with unrealistic deadlines",
        "Limited career growth opportunities available",
        "Toxic workplace culture needs immediate attention",
        "Overworked and understaffed department"
    ]
    
    # Generate dataset
    data = []
    
    # Add positive samples
    for _ in range(200):
        feedback = np.random.choice(positive_feedback)
        data.append({'feedback': feedback, 'sentiment': 'Positive'})
    
    # Add neutral samples  
    for _ in range(100):
        feedback = np.random.choice(neutral_feedback)
        data.append({'feedback': feedback, 'sentiment': 'Neutral'})
        
    # Add negative samples
    for _ in range(100):
        feedback = np.random.choice(negative_feedback)
        data.append({'feedback': feedback, 'sentiment': 'Negative'})
    
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Generate and save data
df = create_sample_data()
df.to_csv('data/synthetic_feedback.csv', index=False)
print(f"Generated {len(df)} samples")
print(df['sentiment'].value_counts())