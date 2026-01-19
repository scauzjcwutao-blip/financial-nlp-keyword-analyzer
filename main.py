"""
Financial NLP Keyword Analyzer - Main Entry Point
Analyzes financial text (e.g., 10-K filings, earnings calls) for keywords and sentiment.
"""

import os
import argparse
import sys

# Import your custom modules
try:
    from src.preprocess import clean_text, lemmatize_text
    from src.keyword_extractor import extract_keywords_tfidf, extract_keywords_rake
    from src.sentiment import analyze_sentiment_vader
    from src.visualize import plot_wordcloud
except ImportError as e:
    print(f"❌ Module import error: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def read_text_file(filepath):
    """Read and return the content of a text file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze financial documents for keywords and sentiment using NLP."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/sample_10k.txt",
        help="Path to the input financial text file (default: data/sample_10k.txt)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top keywords/phrases to extract (default: 15)"
    )
    args = parser.parse_args()

    # Step 1: Load text
    print("🔍 Loading financial text...")
    try:
        raw_text = read_text_file(args.input)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if not raw_text.strip():
        print("⚠️  Warning: Input file is empty!")
        sys.exit(0)

    # Step 2: Preprocess
    print("🧹 Cleaning and lemmatizing text...")
    cleaned = clean_text(raw_text)
    if not cleaned.strip():
        print("⚠️  Text became empty after cleaning. Check input content.")
        sys.exit(0)
    lemmatized = lemmatize_text(cleaned)

    # Step 3: Extract keywords
    print("🔑 Extracting keywords...")
    tfidf_kw = extract_keywords_tfidf([lemmatized], top_n=args.top_n)
    rake_kw = extract_keywords_rake(lemmatized, top_n=args.top_n)

    # Step 4: Sentiment analysis
    print("😊 Performing sentiment analysis (VADER)...")
    sentiment = analyze_sentiment_vader(raw_text)

    # Step 5: Display results
    print("\n" + "=" * 60)
    print("📊 FINANCIAL TEXT ANALYSIS RESULTS")
    print("=" * 60)

    print(f"\n✅ Top {args.top_n} TF-IDF Keywords:")
    print("   • " + "\n   • ".join(tfidf_kw))

    print(f"\n✅ Top {args.top_n} RAKE Phrases:")
    print("   • " + "\n   • ".join(rake_kw))

    print("\n✅ VADER Sentiment Scores:")
    print(f"   Positive : {sentiment['pos']:.3f}")
    print(f"   Neutral  : {sentiment['neu']:.3f}")
    print(f"   Negative : {sentiment['neg']:.3f}")
    print(f"   Compound : {sentiment['compound']:.3f}  (range: -1 [most neg] to +1 [most pos])")

    # Step 6: Visualize word cloud
    print("\n🖼️  Generating word cloud (close the window to exit)...")
    try:
        plot_wordcloud(tfidf_kw, title="Top Financial Keywords (TF-IDF)")
    except Exception as e:
        print(f"⚠️  Failed to show word cloud: {e}")

    print("\n✨ Analysis completed successfully!")


if __name__ == "__main__":
    main()
