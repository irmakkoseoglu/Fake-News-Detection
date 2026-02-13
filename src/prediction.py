"""
Prediction Module for Fake News Detection
Contains functions for making predictions on new data.
"""

import re
import pickle
import numpy as np
from scipy.sparse import hstack


class FakeNewsDetector:
    """
    Fake News Detector class for making predictions.
    """
    
    def __init__(self, model_path, vectorizer_title_path, vectorizer_text_path, scaler_path):
        """
        Initialize the detector with saved models.
        
        Parameters:
        -----------
        model_path : str
            Path to saved model
        vectorizer_title_path : str
            Path to title vectorizer
        vectorizer_text_path : str
            Path to text vectorizer
        scaler_path : str
            Path to feature scaler
        """
        self.model = self._load_pickle(model_path)
        self.vectorizer_title = self._load_pickle(vectorizer_title_path)
        self.vectorizer_text = self._load_pickle(vectorizer_text_path)
        self.scaler = self._load_pickle(scaler_path)
        
        print("Fake News Detector initialized successfully!")
    
    def _load_pickle(self, filepath):
        """Load pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _clean_text(self, text):
        """Clean text data."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())
    
    def _count_capital_words(self, text):
        """Count capitalized words."""
        if not isinstance(text, str):
            return 0
        words = text.split()
        return sum(1 for word in words if word.isupper() and len(word) > 1)
    
    def _avg_sentence_length(self, text):
        """Calculate average sentence length."""
        if not isinstance(text, str) or len(text) == 0:
            return 0
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) == 0:
            return 0
        
        total_words = sum(len(s.split()) for s in sentences)
        return total_words / len(sentences)
    
    def _extract_features(self, title, text):
        """
        Extract all features from title and text.
        
        Parameters:
        -----------
        title : str
            News title
        text : str
            News text
            
        Returns:
        --------
        array-like
            Feature vector
        """
        # Custom features
        title_caps_words = self._count_capital_words(title)
        title_exclamation = title.count('!')
        text_exclamation = text.count('!')
        title_avg_length = self._avg_sentence_length(title)
        text_question = text.count('?')
        
        # Clean text
        cleaned_title = self._clean_text(title)
        cleaned_text = self._clean_text(text)
        
        # TF-IDF features
        X_title = self.vectorizer_title.transform([cleaned_title])
        X_text = self.vectorizer_text.transform([cleaned_text])
        
        # Other features
        other_features = [[title_caps_words, title_avg_length, title_exclamation, 
                          text_exclamation, text_question]]
        
        # Combine all features
        X = hstack([X_title, X_text, other_features])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def predict(self, title, text, return_proba=False):
        """
        Predict if news is fake or real.
        
        Parameters:
        -----------
        title : str
            News title
        text : str
            News text
        return_proba : bool, default=False
            Whether to return probability scores
            
        Returns:
        --------
        str or tuple
            Prediction label or (label, probability)
        """
        # Extract features
        X = self._extract_features(title, text)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        label = "REAL NEWS" if prediction == 1 else "FAKE NEWS"
        
        if return_proba and hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = proba[prediction]
            return label, confidence
        
        return label
    
    def predict_batch(self, titles, texts):
        """
        Predict multiple news articles.
        
        Parameters:
        -----------
        titles : list
            List of news titles
        texts : list
            List of news texts
            
        Returns:
        --------
        list
            List of predictions
        """
        predictions = []
        
        for title, text in zip(titles, texts):
            pred = self.predict(title, text)
            predictions.append(pred)
        
        return predictions


def simple_predict(title, text, model_path='models/best_mlp_model.pkl',
                   vectorizer_title_path='models/vectorizer_title.pkl',
                   vectorizer_text_path='models/vectorizer_text.pkl',
                   scaler_path='models/scaler.pkl'):
    """
    Simple prediction function without creating detector object.
    
    Parameters:
    -----------
    title : str
        News title
    text : str
        News text
    model_path : str
        Path to saved model
    vectorizer_title_path : str
        Path to title vectorizer
    vectorizer_text_path : str
        Path to text vectorizer
    scaler_path : str
        Path to feature scaler
        
    Returns:
    --------
    str
        Prediction label
    """
    detector = FakeNewsDetector(model_path, vectorizer_title_path, 
                                vectorizer_text_path, scaler_path)
    
    return detector.predict(title, text)


if __name__ == "__main__":
    # Example usage
    print("Prediction module loaded successfully!")
    
    # Example
    example_title = "Breaking: Shocking Discovery!"
    example_text = "Scientists have made an incredible discovery that will change everything..."
    
    print(f"\nExample prediction:")
    print(f"Title: {example_title}")
    print(f"Text: {example_text[:50]}...")
    
    # Note: This will fail without trained models
    # prediction = simple_predict(example_title, example_text)
    # print(f"Prediction: {prediction}")
