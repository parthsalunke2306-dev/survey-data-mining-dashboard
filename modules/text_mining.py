"""
Text Mining & NLP Module for Open-Ended Survey Feedback
Analyzes Question 25: Topics students want covered in future workshops and campus events.
Includes tokenization, stopword removal, n-gram extraction, TF-IDF, and Topic Categorization.
"""
import re
from collections import Counter
import pandas as pd
from typing import Dict, List, Any

# Common English and generic stopwords
STOPWORDS = set([
    "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
    "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further",
    "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your",
    "yours", "yourself", "yourselves", "like", "want", "learn", "topics", "finance", "covered", "workshops",
    "campus", "events", "future", "regarding", "see", "also", "good", "know", "how"
])

TOPIC_KEYWORDS = {
    "Stock Market & Equity": ["stock", "stocks", "equity", "trading", "shares", "market", "analysis", "technical", "fundamental", "nse", "bse"],
    "Mutual Funds & SIPs": ["mutual", "funds", "sip", "index", "etf", "compound", "compounding", "passive", "portfolio"],
    "Budgeting & Money Basics": ["budget", "budgeting", "saving", "savings", "save", "expense", "expenses", "discipline", "beginner", "basics"],
    "Crypto & Web3": ["crypto", "cryptocurrency", "bitcoin", "blockchain", "wallet", "nft"],
    "Real Estate & Alternative": ["real", "estate", "property", "gold", "commodities"],
    "Taxation & Debt Management": ["tax", "taxes", "taxation", "debt", "loan", "loans", "credit", "cards"]
}

def analyze_survey_text(df: pd.DataFrame, text_column: str = "Workshop_Interests") -> Dict[str, Any]:
    """
    Extracts high-value NLP signals, keyword frequencies, and topic clusters from student comments.
    """
    # Find relevant text column
    col = text_column if text_column in df.columns else None
    if not col:
        for c in df.columns:
            if "workshop" in c.lower() or "topic" in c.lower():
                col = c
                break

    if not col:
        return {"top_words": [], "topic_distribution": {}, "sample_responses": []}

    texts = df[col].dropna().astype(str).tolist()

    # 1. Clean & Tokenize
    all_tokens = []
    bigrams = []
    topic_counts = {t: 0 for t in TOPIC_KEYWORDS}

    for text in texts:
        # Normalize
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower()).strip()
        tokens = [w for w in cleaned.split() if len(w) > 2 and w not in STOPWORDS]
        all_tokens.extend(tokens)

        # Bigrams
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]} {tokens[i+1]}")

        # Topic matching
        text_lower = text.lower()
        for topic, keywords in TOPIC_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                topic_counts[topic] += 1

    # Frequency analysis
    word_freq = Counter(all_tokens).most_common(15)
    bigram_freq = Counter(bigrams).most_common(8)

    top_words = [{"word": w, "count": c} for w, c in word_freq]
    top_bigrams = [{"phrase": p, "count": c} for p, c in bigram_freq]

    # Format topic distribution
    total_tagged = max(sum(topic_counts.values()), 1)
    topic_dist = [
        {"Topic": topic, "Mentions": count, "Percentage": f"{(count / len(texts)) * 100:.1f}%"}
        for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
    ]

    # Sample quotes for display
    meaningful_samples = [t for t in texts if len(t.strip()) > 25][:6]

    return {
        "total_responses_analyzed": len(texts),
        "top_keywords": top_words,
        "top_bigrams": top_bigrams,
        "topic_distribution": topic_dist,
        "sample_quotes": meaningful_samples
    }
