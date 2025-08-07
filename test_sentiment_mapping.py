import pandas as pd

# Load sentiment data
sentiment_df = pd.read_excel('data/sentiment analysis- Final combined .xlsx')
print('Sentiment data groups:')
print(sentiment_df['Academic_Division'].unique())

def map_division_to_major_group(division):
    """Map detailed academic divisions to three major groups: Humanities, Natural Science, Social Science."""
    
    social_science_keywords = [
        'psychology', 'sociology', 'economics', 'political', 'education',
        "social work", 'public health', 'business', 'management', 'marketing',
        'finance', 'accounting', 'law', 'criminal justice', 'urban planning',
        'international relations', 'communication', 'journalism', 'public policy'
    ]
    
    division_lower = str(division).lower()
    
    for keyword in social_science_keywords:
        if keyword in division_lower:
            return 'Social Science'
    
    return division

# Test Economics mapping
mapped = map_division_to_major_group('Economics')
print(f'\nEconomics -> {mapped}')
print(f'Available in sentiment data: {mapped in sentiment_df["Academic_Division"].unique()}')

# Check sentiment data for Social Science
social_science_data = sentiment_df[sentiment_df['Academic_Division'] == 'Social Science']
print(f'\nSocial Science sentiment data: {len(social_science_data)} rows')
if len(social_science_data) > 0:
    print(f'Sample comments:')
    for i, row in social_science_data.head(3).iterrows():
        print(f"  {i+1}. {str(row['Comments'])[:100]}...") 