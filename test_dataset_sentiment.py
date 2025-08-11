import pandas as pd
from textblob import TextBlob

# Load the dataset
df = pd.read_excel('/Users/ramparekh/Downloads/sentiment analysis- Final combined .xlsx')

print("=== DATASET ANALYSIS ===")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Academic_Division unique values: {df['Academic_Division'].unique()}")

print("\n=== SAMPLE COMMENTS WITH SENTIMENT ===")
for i, comment in enumerate(df['Comments'].head(10)):
    blob = TextBlob(str(comment))
    print(f"{i+1}. \"{comment[:50]}...\" -> TextBlob: {blob.sentiment.polarity:.3f}")

print("\n=== GROUP AVERAGES ===")
for group in df['Academic_Division'].unique():
    group_data = df[df['Academic_Division'] == group]
    sentiments = [TextBlob(str(comment)).sentiment.polarity for comment in group_data['Comments']]
    avg = sum(sentiments) / len(sentiments)
    print(f"{group}: {avg:.3f} ({len(group_data)} comments)")

print("\n=== DETAILED GROUP ANALYSIS ===")
for group in df['Academic_Division'].unique():
    group_data = df[df['Academic_Division'] == group]
    print(f"\n{group} ({len(group_data)} comments):")
    
    # Calculate sentiment scores
    sentiments = []
    for comment in group_data['Comments']:
        blob = TextBlob(str(comment))
        sentiments.append(blob.sentiment.polarity)
    
    # Statistics
    avg_sentiment = sum(sentiments) / len(sentiments)
    min_sentiment = min(sentiments)
    max_sentiment = max(sentiments)
    positive_count = sum(1 for s in sentiments if s > 0.1)
    negative_count = sum(1 for s in sentiments if s < -0.1)
    neutral_count = len(sentiments) - positive_count - negative_count
    
    print(f"  Average: {avg_sentiment:.3f}")
    print(f"  Range: {min_sentiment:.3f} to {max_sentiment:.3f}")
    print(f"  Positive: {positive_count} ({positive_count/len(sentiments)*100:.1f}%)")
    print(f"  Negative: {negative_count} ({negative_count/len(sentiments)*100:.1f}%)")
    print(f"  Neutral: {neutral_count} ({neutral_count/len(sentiments)*100:.1f}%)") 