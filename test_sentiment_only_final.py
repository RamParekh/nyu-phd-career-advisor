import pandas as pd
from textblob import TextBlob
import re
import contractions

# Copy the exact sentiment functions from app1.py
def preprocess_text_for_sentiment(text):
    """Expand contractions, normalize repeated chars, handle sarcasm cues, etc."""
    if not isinstance(text, str):
        return ""
    # Expand contractions
    text = contractions.fix(text)
    # Lowercase
    text = text.lower()
    # Normalize repeated characters (e.g., sooooo -> soo)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    # Remove excessive punctuation (keep ! and ? for sentiment)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def detect_sarcasm(text):
    SARCASM_CUES = [
        'yeah right', 'as if', 'totally', 'sure', 'obviously', 'great...', 'nice...', 'love that',
        'just perfect', 'wonderful', 'amazing', 'sure thing', 'can\'t wait', '/s', 'right...',
        'fantastic', 'brilliant', 'awesome', 'lovely', 'what a surprise', 'what a joy', 'so fun',
        'so helpful', 'thanks a lot', 'thanks for nothing', 'good luck with that', 'i bet',
    ]
    text = text.lower()
    for cue in SARCASM_CUES:
        if cue in text:
            return True
    return False

def calculate_sentiment_with_multiple_models(text):
    """Calculate sentiment using multiple models and return ensemble result. Hypertuned for sarcasm/informal."""
    results = {}
    pre_text = preprocess_text_for_sentiment(text)
    sarcasm = detect_sarcasm(pre_text)

    # TextBlob
    try:
        blob = TextBlob(pre_text)
        results['TextBlob'] = blob.sentiment.polarity
        results['TextBlob_subjectivity'] = blob.sentiment.subjectivity
    except:
        results['TextBlob'] = 0.0
        results['TextBlob_subjectivity'] = 0.0

    # VADER (with custom lexicon)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        vader_scores = analyzer.polarity_scores(pre_text)
        results['VADER'] = vader_scores['compound']
    except:
        results['VADER'] = 0.0

    # Dynamic ensemble weighting
    if sarcasm:
        vader_weight = 0.8
        tb_weight = 0.2
    else:
        vader_weight = 0.6
        tb_weight = 0.4

    # If TextBlob is highly subjective but neutral, nudge negative if sarcasm
    tb_score = results.get('TextBlob', 0.0)
    tb_subj = results.get('TextBlob_subjectivity', 0.0)
    if sarcasm and abs(tb_score) < 0.05 and tb_subj > 0.5:
        tb_score = -0.2
        results['TextBlob'] = tb_score

    # Weighted ensemble - always calculate ensemble score
    results['Ensemble'] = vader_weight * results['VADER'] + tb_weight * tb_score

    # If both models are neutral but sarcasm cues, nudge negative
    if sarcasm and abs(results['Ensemble']) < 0.05:
        results['Ensemble'] = -0.2

    return results

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
    sentiment_scores = []
    for comment in df_copy['Comments']:
        try:
            if pd.isna(comment) or str(comment).strip() == '':
                sentiment_scores.append(0.0)
            else:
                # Use ensemble model for better accuracy
                ensemble_result = calculate_sentiment_with_multiple_models(comment)
                sentiment_scores.append(ensemble_result['Ensemble'])
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

# Test the sentiment analysis
print("=== TESTING SENTIMENT ANALYSIS ONLY ===")
print("Loading sentiment data...")

# Load the sentiment data
sentiment_df = pd.read_excel('data/sentiment analysis- Final combined .xlsx')
print(f"✅ Loaded {len(sentiment_df)} rows")

# Test individual sentiment calculation
test_comments = [
    "I love this program!",
    "This is terrible.",
    "Very fond memories.",
    "There is a dire need for professional career services.",
    "I would have liked more emphasis on how to get a paper turned into a journal article."
]

print("\n=== Testing individual sentiment calculation ===")
for i, comment in enumerate(test_comments):
    result = calculate_sentiment_with_multiple_models(comment)
    print(f"Comment {i+1}: '{comment}'")
    print(f"  TextBlob: {result['TextBlob']:.3f}")
    print(f"  VADER: {result['VADER']:.3f}")
    print(f"  Ensemble: {result['Ensemble']:.3f}")
    print()

# Test full sentiment calculation
print("=== Testing full sentiment calculation ===")
sentiment_scores, processed_df = calculate_sentiment_scores(sentiment_df)

print(f"\nFinal sentiment scores:")
for group, data in sentiment_scores.items():
    print(f"  {group}: {data['average_sentiment']:.3f} ({data['comment_count']} comments)")

print("\n=== Testing mapping ===")
def map_division_to_major_group(division):
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

test_divisions = ['Economics', 'Chemistry', 'Comparative Literature']
for division in test_divisions:
    mapped_group = map_division_to_major_group(division)
    has_sentiment = mapped_group in sentiment_scores
    sentiment_value = sentiment_scores[mapped_group]['average_sentiment'] if has_sentiment else "N/A"
    print(f"{division} -> {mapped_group} -> Sentiment: {sentiment_value}")

print("\n✅ Sentiment analysis test complete!") 