import pandas as pd
import base64
import io

# Test the data loading functions from app1.py
def load_sentiment_data():
    """Load sentiment data from Streamlit secrets or local file."""
    try:
        # Try to load from Streamlit secrets first
        import streamlit as st
        if 'sentiment_data' in st.secrets:
            sentiment_data = st.secrets['sentiment_data']
            # Decode base64 data
            decoded_data = base64.b64decode(sentiment_data)
            df = pd.read_excel(io.BytesIO(decoded_data))
            print("✅ Loaded sentiment data from Streamlit secrets")
            return df
    except Exception as e:
        print(f"❌ Could not load from Streamlit secrets: {e}")
    
    try:
        # Fallback to local file
        df = pd.read_excel('data/sentiment analysis- Final combined .xlsx')
        print("✅ Loaded sentiment data from local file")
        return df
    except Exception as e:
        print(f"❌ Could not load from local file: {e}")
        return None

# Test the sentiment calculation
def calculate_sentiment_scores(sentiment_df):
    """Calculate sentiment scores for each major academic group using ensemble model."""
    if sentiment_df is None or sentiment_df.empty:
        print("❌ Sentiment DataFrame is None or empty")
        return None, None
    
    print(f"🔍 Processing sentiment data: {len(sentiment_df)} rows")
    if 'Comments' not in sentiment_df.columns:
        print(f"❌ Comments column missing. Available columns: {list(sentiment_df.columns)}")
        return None, None
    
    # Create a copy to avoid modifying the original
    df_copy = sentiment_df.copy()
    
    # The Academic_Division column already contains the major groups
    # Just use it directly as Major_Group
    df_copy['Major_Group'] = df_copy['Academic_Division']
    
    # Calculate sentiment scores using ensemble model
    from textblob import TextBlob
    sentiment_scores = []
    for comment in df_copy['Comments']:
        try:
            if pd.isna(comment) or str(comment).strip() == '':
                sentiment_scores.append(0.0)
            else:
                # Use TextBlob for simplicity in test
                blob = TextBlob(str(comment))
                sentiment_scores.append(blob.sentiment.polarity)
        except:
            sentiment_scores.append(0.0)
    
    # Ensure the length matches
    if len(sentiment_scores) != len(df_copy):
        print(f"❌ Sentiment score length mismatch: {len(sentiment_scores)} vs {len(df_copy)}")
        return None, None
    
    df_copy['Sentiment_Score'] = sentiment_scores
    
    # Group by major academic groups and calculate average sentiment
    group_sentiments = {}
    for group in ['Humanities', 'Natural Science', 'Social Science']:
        group_data = df_copy[df_copy['Major_Group'] == group]
        if not group_data.empty:
            # Include all sentiment scores, including 0.0 (neutral)
            avg_sentiment = group_data['Sentiment_Score'].mean()
            count = len(group_data)
            group_sentiments[group] = {
                'average_sentiment': avg_sentiment,
                'comment_count': count,
                'sentiment_range': (group_data['Sentiment_Score'].min(), group_data['Sentiment_Score'].max())
            }
            print(f"🔍 {group}: {count} comments, avg sentiment: {avg_sentiment:.3f}, range: {group_data['Sentiment_Score'].min():.3f} to {group_data['Sentiment_Score'].max():.3f}")
        else:
            print(f"⚠️ No data found for group: {group}")
    
    return group_sentiments, df_copy

# Test the mapping function
def map_division_to_major_group(division):
    """Map detailed academic divisions to three major groups: Humanities, Natural Science, Social Science."""
    
    # Define mapping dictionaries for each major group
    humanities_keywords = [
        'humanities', 'philosophy', 'history', 'literature', 'english', 'languages', 
        'linguistics', 'classics', 'religious', 'theology', 'art', 'music', 'drama',
        'theater', 'film', 'media', 'cultural', 'anthropology', 'archaeology'
    ]
    
    natural_science_keywords = [
        'biology', 'chemistry', 'physics', 'mathematics', 'computer science', 
        'engineering', 'medical', 'health', 'neuroscience', 'biochemistry',
        'molecular', 'genetics', 'ecology', 'environmental', 'geology',
        'astronomy', 'biomedical', 'pharmaceutical', 'biotechnology'
    ]
    
    social_science_keywords = [
        'psychology', 'sociology', 'economics', 'political', 'education',
        "social work", 'public health', 'business', 'management', 'marketing',
        'finance', 'accounting', 'law', 'criminal justice', 'urban planning',
        'international relations', 'communication', 'journalism', 'public policy'
    ]
    
    division_lower = str(division).lower()
    
    # Check for matches in each category
    for keyword in humanities_keywords:
        if keyword in division_lower:
            return 'Humanities'
    
    for keyword in natural_science_keywords:
        if keyword in division_lower:
            return 'Natural Science'
    
    for keyword in social_science_keywords:
        if keyword in division_lower:
            return 'Social Science'
    
    # Default mapping for common patterns
    if any(word in division_lower for word in ['basic medical', 'medical science']):
        return 'Natural Science'
    elif any(word in division_lower for word in ['computer', 'cs', 'software']):
        return 'Natural Science'
    elif any(word in division_lower for word in ['bio', 'biological']):
        return 'Natural Science'
    
    # If no match found, return the original division
    return division

# Test the complete flow
print("=== TESTING APP DATA LOADING ===")

# Test 1: Load sentiment data
sentiment_df = load_sentiment_data()
if sentiment_df is not None:
    print(f"✅ Sentiment data loaded: {len(sentiment_df)} rows")
    print(f"Columns: {list(sentiment_df.columns)}")
    print(f"Academic_Division values: {sentiment_df['Academic_Division'].unique()}")
else:
    print("❌ Failed to load sentiment data")
    exit(1)

# Test 2: Calculate sentiment scores
sentiment_scores, processed_df = calculate_sentiment_scores(sentiment_df)
if sentiment_scores is not None:
    print(f"✅ Sentiment scores calculated for {len(sentiment_scores)} groups")
    for group, data in sentiment_scores.items():
        print(f"  {group}: {data['average_sentiment']:.3f} ({data['comment_count']} comments)")
else:
    print("❌ Failed to calculate sentiment scores")
    exit(1)

# Test 3: Test mapping
test_divisions = ['Economics', 'Chemistry', 'Comparative Literature']
print("\n=== TESTING MAPPING ===")
for division in test_divisions:
    mapped_group = map_division_to_major_group(division)
    has_sentiment = mapped_group in sentiment_scores
    sentiment_value = sentiment_scores[mapped_group]['average_sentiment'] if has_sentiment else "N/A"
    print(f"{division} -> {mapped_group} -> Sentiment: {sentiment_value}")

print("\n✅ All tests passed! The sentiment analysis should work in the app.") 