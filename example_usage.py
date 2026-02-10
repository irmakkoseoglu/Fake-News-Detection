"""
Example Usage Script for Fake News Detection

This script demonstrates how to use the trained model to make predictions.
"""

import sys
sys.path.append('src')

from prediction import FakeNewsDetector


def main():
    """Main function to demonstrate prediction."""
    
    print("=" * 70)
    print("FAKE NEWS DETECTION - EXAMPLE USAGE")
    print("=" * 70)
    
    # Initialize detector
    print("\nInitializing detector...")
    
    try:
        detector = FakeNewsDetector(
            model_path='models/best_mlp_model.pkl',
            vectorizer_title_path='models/vectorizer_title.pkl',
            vectorizer_text_path='models/vectorizer_text.pkl',
            scaler_path='models/scaler.pkl'
        )
    except FileNotFoundError:
        print("\n⚠️  Error: Model files not found!")
        print("Please train the model first using the Jupyter notebook.")
        return
    
    # Example 1: Real news
    print("\n" + "-" * 70)
    print("Example 1: Real News Article")
    print("-" * 70)
    
    title1 = "Senate Passes New Healthcare Reform Bill"
    text1 = """
    Washington - The U.S. Senate passed a comprehensive healthcare reform 
    bill on Tuesday by a vote of 52-48. The legislation, which has been 
    debated for months, aims to expand coverage while reducing costs. 
    The bill now moves to the House of Representatives for consideration.
    """
    
    print(f"Title: {title1}")
    print(f"Text: {text1.strip()[:100]}...")
    
    prediction1, confidence1 = detector.predict(title1, text1, return_proba=True)
    print(f"\n✓ Prediction: {prediction1}")
    print(f"✓ Confidence: {confidence1:.2%}")
    
    # Example 2: Potentially fake news
    print("\n" + "-" * 70)
    print("Example 2: Suspicious Article")
    print("-" * 70)
    
    title2 = "BREAKING: SHOCKING Discovery That Will CHANGE EVERYTHING!!!"
    text2 = """
    You won't BELIEVE what scientists just found! This incredible 
    discovery will revolutionize the world as we know it! Click here 
    to find out more! Doctors HATE this simple trick! Share this with 
    everyone you know before it's too late!!! Don't let THEM hide the truth!
    """
    
    print(f"Title: {title2}")
    print(f"Text: {text2.strip()[:100]}...")
    
    prediction2, confidence2 = detector.predict(title2, text2, return_proba=True)
    print(f"\n✓ Prediction: {prediction2}")
    print(f"✓ Confidence: {confidence2:.2%}")
    
    # Example 3: Batch prediction
    print("\n" + "-" * 70)
    print("Example 3: Batch Prediction")
    print("-" * 70)
    
    titles = [
        "President Signs New Economic Bill",
        "UNBELIEVABLE: Aliens Found in Antarctica!!!",
        "Stock Market Reaches Record High"
    ]
    
    texts = [
        "The president signed a major economic stimulus bill today...",
        "Top secret documents reveal shocking truth about aliens!!!",
        "Wall Street celebrated as the stock market reached new highs..."
    ]
    
    predictions = detector.predict_batch(titles, texts)
    
    for i, (title, pred) in enumerate(zip(titles, predictions), 1):
        print(f"\n{i}. {title}")
        print(f"   → {pred}")
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETED")
    print("=" * 70)
    print("\nNote: These are example predictions. Always verify news from")
    print("multiple reliable sources before accepting as fact.")


if __name__ == "__main__":
    main()
