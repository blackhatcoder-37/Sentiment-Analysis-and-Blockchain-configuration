import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Set display options
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# Download required NLTK data
nltk.download('vader_lexicon', quiet=True)

# Load data
df = pd.read_csv('C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/Reviews.csv')
print(f"Total reviews loaded: {df.shape[0]}")
df = df.head(500)
print(f"Processing first {df.shape[0]} reviews\n")

# Initialize models
print("Initializing sentiment analysis models...")
sia = SentimentIntensityAnalyzer()
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
roberta_model = AutoModelForSequenceClassification.from_pretrained(MODEL)
print("Models loaded successfully!\n")


class HybridSentimentAnalyzer:
    """
    Hybrid Sentiment Analyzer combining VADER and RoBERTa
    This ensemble approach increases model capacity by leveraging:
    - VADER: Fast, rule-based, good with social media text and emojis
    - RoBERTa: Deep learning model, contextual understanding, handles complex sentences
    """
    
    def __init__(self, vader_analyzer, roberta_model, tokenizer, vader_weight=0.3, roberta_weight=0.7):
        """
        Initialize hybrid analyzer with customizable weights
        
        Args:
            vader_analyzer: NLTK VADER sentiment analyzer
            roberta_model: Pre-trained RoBERTa model
            tokenizer: RoBERTa tokenizer
            vader_weight: Weight for VADER scores (default 0.3)
            roberta_weight: Weight for RoBERTa scores (default 0.7)
        """
        self.vader = vader_analyzer
        self.roberta = roberta_model
        self.tokenizer = tokenizer
        self.vader_weight = vader_weight
        self.roberta_weight = roberta_weight
        
        # Normalize weights
        total = vader_weight + roberta_weight
        self.vader_weight = vader_weight / total
        self.roberta_weight = roberta_weight / total
    
    def get_vader_scores(self, text):
        """Get VADER sentiment scores"""
        scores = self.vader.polarity_scores(text)
        return {
            'vader_neg': scores['neg'],
            'vader_neu': scores['neu'],
            'vader_pos': scores['pos'],
            'vader_compound': scores['compound']
        }
    
    def get_roberta_scores(self, text):
        """Get RoBERTa sentiment scores"""
        # Truncate text if too long (RoBERTa has 512 token limit)
        encoded = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        output = self.roberta(**encoded)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        return {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }
    
    def get_hybrid_scores(self, text):
        """
        Get combined sentiment scores from both models
        Creates ensemble predictions with weighted averaging
        """
        try:
            # Get individual model scores
            vader_scores = self.get_vader_scores(text)
            roberta_scores = self.get_roberta_scores(text)
            
            # Calculate weighted hybrid scores
            hybrid_neg = (vader_scores['vader_neg'] * self.vader_weight + 
                         roberta_scores['roberta_neg'] * self.roberta_weight)
            hybrid_neu = (vader_scores['vader_neu'] * self.vader_weight + 
                         roberta_scores['roberta_neu'] * self.roberta_weight)
            hybrid_pos = (vader_scores['vader_pos'] * self.vader_weight + 
                         roberta_scores['roberta_pos'] * self.roberta_weight)
            
            # Calculate hybrid compound score
            # Normalize VADER compound from [-1, 1] to [0, 1] to match RoBERTa scale
            vader_compound_normalized = (vader_scores['vader_compound'] + 1) / 2
            roberta_compound = roberta_scores['roberta_pos'] - roberta_scores['roberta_neg']
            roberta_compound_normalized = (roberta_compound + 1) / 2
            
            hybrid_compound_normalized = (vader_compound_normalized * self.vader_weight + 
                                         roberta_compound_normalized * self.roberta_weight)
            # Convert back to [-1, 1] scale
            hybrid_compound = hybrid_compound_normalized * 2 - 1
            
            # Determine final sentiment label
            sentiment_label = self.get_sentiment_label(hybrid_neg, hybrid_neu, hybrid_pos)
            
            # Combine all scores
            all_scores = {
                **vader_scores,
                **roberta_scores,
                'hybrid_neg': hybrid_neg,
                'hybrid_neu': hybrid_neu,
                'hybrid_pos': hybrid_pos,
                'hybrid_compound': hybrid_compound,
                'hybrid_sentiment': sentiment_label,
                'confidence': max(hybrid_neg, hybrid_neu, hybrid_pos)  # Confidence in prediction
            }
            
            return all_scores
            
        except Exception as e:
            print(f"Error processing text: {e}")
            return None
    
    def get_sentiment_label(self, neg, neu, pos):
        """Determine sentiment label based on scores"""
        max_score = max(neg, neu, pos)
        if max_score == pos:
            return 'Positive'
        elif max_score == neg:
            return 'Negative'
        else:
            return 'Neutral'


# Initialize hybrid analyzer
print("Creating Hybrid Sentiment Analyzer...")
print("Configuration:")
print("- VADER weight: 30% (rule-based, fast, emoji-aware)")
print("- RoBERTa weight: 70% (deep learning, contextual understanding)")
print("- Ensemble method: Weighted averaging\n")

hybrid_analyzer = HybridSentimentAnalyzer(
    vader_analyzer=sia,
    roberta_model=roberta_model,
    tokenizer=tokenizer,
    vader_weight=0.3,
    roberta_weight=0.7
)

# Test on a sample
print("Testing on sample review...")
sample_text = df['Text'].values[50]
print(f"Sample text: {sample_text[:200]}...")
sample_scores = hybrid_analyzer.get_hybrid_scores(sample_text)
print("\nHybrid Analysis Results:")
print(f"- Sentiment: {sample_scores['hybrid_sentiment']}")
print(f"- Confidence: {sample_scores['confidence']:.3f}")
print(f"- Hybrid Compound Score: {sample_scores['hybrid_compound']:.3f}")
print(f"- Negative: {sample_scores['hybrid_neg']:.3f}")
print(f"- Neutral: {sample_scores['hybrid_neu']:.3f}")
print(f"- Positive: {sample_scores['hybrid_pos']:.3f}\n")

# Process all reviews
print("Processing all reviews with hybrid model...")
results = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        scores = hybrid_analyzer.get_hybrid_scores(text)
        if scores:
            results[myid] = scores
    except Exception as e:
        print(f'Error for id {myid}: {e}')

# Create results dataframe
print("\nCreating results dataframe...")
results_df = pd.DataFrame(results).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')

print(f"Processed {len(results_df)} reviews successfully")
print("\nResults preview:")
print(results_df[['Id', 'Score', 'hybrid_sentiment', 'hybrid_compound', 'confidence']].head(10))

# Save results
output_file = 'C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/hybrid_sentiment_results.csv'
results_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")

# Visualizations
print("\nGenerating visualizations...")

# 1. Compare hybrid compound scores by star rating
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(data=results_df, x='Score', y='hybrid_compound', ax=ax, palette='coolwarm')
ax.set_title('Hybrid Sentiment Compound Score by Amazon Star Rating', fontsize=14, fontweight='bold')
ax.set_xlabel('Star Rating', fontsize=12)
ax.set_ylabel('Hybrid Compound Score', fontsize=12)
plt.tight_layout()
plt.savefig('C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/hybrid_compound_by_rating.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Distribution of hybrid sentiments
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Sentiment distribution
sentiment_counts = results_df['hybrid_sentiment'].value_counts()
colors = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
ax1.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%',
        colors=[colors[x] for x in sentiment_counts.index], startangle=90)
ax1.set_title('Distribution of Hybrid Sentiment Predictions', fontsize=14, fontweight='bold')

# Confidence distribution
ax2.hist(results_df['confidence'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Confidence Score', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
ax2.axvline(results_df['confidence'].mean(), color='red', linestyle='--', 
            label=f'Mean: {results_df["confidence"].mean():.3f}')
ax2.legend()

plt.tight_layout()
plt.savefig('C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/hybrid_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Compare all three scores (neg, neu, pos) for hybrid model
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.barplot(data=results_df, x='Score', y='hybrid_neg', ax=axs[0], color='#e74c3c')
sns.barplot(data=results_df, x='Score', y='hybrid_neu', ax=axs[1], color='#95a5a6')
sns.barplot(data=results_df, x='Score', y='hybrid_pos', ax=axs[2], color='#2ecc71')

axs[0].set_title('Negative Sentiment', fontsize=12, fontweight='bold')
axs[1].set_title('Neutral Sentiment', fontsize=12, fontweight='bold')
axs[2].set_title('Positive Sentiment', fontsize=12, fontweight='bold')

for ax in axs:
    ax.set_xlabel('Star Rating', fontsize=10)
    ax.set_ylabel('Hybrid Score', fontsize=10)

plt.tight_layout()
plt.savefig('C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/hybrid_sentiment_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Model comparison: VADER vs RoBERTa vs Hybrid
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# VADER compound scores
vader_compound_cols = results_df['vader_compound']
sns.barplot(data=results_df, x='Score', y='vader_compound', ax=axs[0], color='#3498db')
axs[0].set_title('VADER Model', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Compound Score', fontsize=10)
axs[0].set_xlabel('Star Rating', fontsize=10)

# RoBERTa compound (calculated from pos - neg)
results_df['roberta_compound'] = results_df['roberta_pos'] - results_df['roberta_neg']
sns.barplot(data=results_df, x='Score', y='roberta_compound', ax=axs[1], color='#9b59b6')
axs[1].set_title('RoBERTa Model', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Compound Score', fontsize=10)
axs[1].set_xlabel('Star Rating', fontsize=10)

# Hybrid compound scores
sns.barplot(data=results_df, x='Score', y='hybrid_compound', ax=axs[2], color='#e67e22')
axs[2].set_title('Hybrid Model (VADER + RoBERTa)', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Compound Score', fontsize=10)
axs[2].set_xlabel('Star Rating', fontsize=10)

plt.suptitle('Model Comparison: Sentiment Scores by Star Rating', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Performance metrics
print("\n" + "="*60)
print("HYBRID MODEL PERFORMANCE SUMMARY")
print("="*60)
print(f"Total Reviews Analyzed: {len(results_df)}")
print(f"Average Confidence: {results_df['confidence'].mean():.3f}")
print(f"Median Confidence: {results_df['confidence'].median():.3f}")
print(f"\nSentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(results_df)) * 100
    print(f"  {sentiment}: {count} ({percentage:.1f}%)")

print(f"\nHybrid Compound Score Statistics:")
print(f"  Mean: {results_df['hybrid_compound'].mean():.3f}")
print(f"  Median: {results_df['hybrid_compound'].median():.3f}")
print(f"  Std Dev: {results_df['hybrid_compound'].std():.3f}")
print(f"  Min: {results_df['hybrid_compound'].min():.3f}")
print(f"  Max: {results_df['hybrid_compound'].max():.3f}")

# Correlation with actual ratings
correlation = results_df[['Score', 'vader_compound', 'roberta_compound', 'hybrid_compound']].corr()['Score'][1:]
print(f"\nCorrelation with Star Ratings:")
print(f"  VADER: {correlation['vader_compound']:.3f}")
print(f"  RoBERTa: {correlation['roberta_compound']:.3f}")
print(f"  Hybrid: {correlation['hybrid_compound']:.3f}")
print("="*60)

print("\n✅ Hybrid sentiment analysis complete!")
print(f"📊 All visualizations saved to: C:/Users/VYSHNAVI R/OneDrive/Sentiment Analysis/")
