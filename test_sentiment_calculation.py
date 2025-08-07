import pandas as pd
from textblob import TextBlob
import re
import contractions

# Test the sentiment calculation function
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

    # VADER (simplified for testing)
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

# Load sentiment data
print("Loading sentiment data...")
sentiment_df = pd.read_excel('data/sentiment analysis- Final combined .xlsx')
print(f"Loaded {len(sentiment_df)} rows")

# Test sentiment calculation on sample comments
print("\nTesting sentiment calculation on sample comments:")
sample_comments = sentiment_df['Comments'].head(10)

for i, comment in enumerate(sample_comments):
    print(f"\nComment {i+1}: '{str(comment)[:100]}...'")
    
    # Test simple TextBlob
    simple_score = TextBlob(str(comment)).sentiment.polarity
    print(f"  Simple TextBlob: {simple_score:.3f}")
    
    # Test ensemble model
    ensemble_result = calculate_sentiment_with_multiple_models(comment)
    print(f"  Ensemble result: {ensemble_result['Ensemble']:.3f}")
    print(f"  TextBlob: {ensemble_result['TextBlob']:.3f}")
    print(f"  VADER: {ensemble_result['VADER']:.3f}")

# Test specific comments that should have clear sentiment
test_comments = [
    "I love this program!",
    "This is terrible.",
    "Very fond memories.",
    "There is a dire need for professional career services.",
    "I would have liked more emphasis on how to get a paper turned into a journal article."
]

print("\n\nTesting specific comments:")
for i, comment in enumerate(test_comments):
    ensemble_result = calculate_sentiment_with_multiple_models(comment)
    print(f"  '{comment}' -> Ensemble: {ensemble_result['Ensemble']:.3f}") 