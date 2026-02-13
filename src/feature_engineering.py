"""
Feature Engineering Module for Fake News Detection
Contains functions for extracting features from text data.
"""

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def count_capital_words(text):
    """
    Count the number of fully capitalized words in text.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    int
        Number of capitalized words (length > 1)
    """
    if not isinstance(text, str):
        return 0
    
    words = text.split()
    capital_words = sum(1 for word in words if word.isupper() and len(word) > 1)
    
    return capital_words


def count_punctuation(text, punctuation='!'):
    """
    Count specific punctuation marks in text.
    
    Parameters:
    -----------
    text : str
        Input text
    punctuation : str, default='!'
        Punctuation mark to count
        
    Returns:
    --------
    int
        Count of punctuation marks
    """
    if not isinstance(text, str):
        return 0
    
    return text.count(punctuation)


def avg_sentence_length(text):
    """
    Calculate average sentence length in words.
    
    Parameters:
    -----------
    text : str
        Input text
        
    Returns:
    --------
    float
        Average sentence length
    """
    if not isinstance(text, str) or len(text) == 0:
        return 0
    
    # Split by sentence delimiters
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) == 0:
        return 0
    
    # Calculate average
    total_words = sum(len(s.split()) for s in sentences)
    avg_length = total_words / len(sentences)
    
    return avg_length


def extract_text_features(df, title_col='title', text_col='text'):
    """
    Extract custom features from title and text columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    title_col : str, default='title'
        Name of title column
    text_col : str, default='text'
        Name of text column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional feature columns
    """
    df = df.copy()
    
    # Title features
    df['title_caps_words'] = df[title_col].apply(count_capital_words)
    df['title_exclamation'] = df[title_col].apply(lambda x: count_punctuation(x, '!'))
    df['title_avg_length'] = df[title_col].apply(avg_sentence_length)
    
    # Text features
    df['text_exclamation'] = df[text_col].apply(lambda x: count_punctuation(x, '!'))
    df['text_question'] = df[text_col].apply(lambda x: count_punctuation(x, '?'))
    
    print(f"Extracted {5} custom features")
    
    return df


def create_tfidf_features(texts, max_features=500, ngram_range=(1, 2)):
    """
    Create TF-IDF features from text.
    
    Parameters:
    -----------
    texts : list or pd.Series
        List of text documents
    max_features : int, default=500
        Maximum number of features
    ngram_range : tuple, default=(1, 2)
        Range of n-grams to consider
        
    Returns:
    --------
    vectorizer : TfidfVectorizer
        Fitted vectorizer
    features : scipy.sparse matrix
        TF-IDF features
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        lowercase=True,
        strip_accents='unicode'
    )
    
    features = vectorizer.fit_transform(texts)
    
    print(f"Created {features.shape[1]} TF-IDF features")
    
    return vectorizer, features


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully!")
