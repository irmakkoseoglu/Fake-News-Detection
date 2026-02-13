"""
Data Preprocessing Module for Fake News Detection
Contains functions for cleaning and preparing text data.
"""

import re
import pandas as pd
import numpy as np


def clean_text(text):
    """
    Clean and normalize text data.
    
    Parameters:
    -----------
    text : str
        Raw text to be cleaned
        
    Returns:
    --------
    str
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text


def remove_duplicates(df):
    """
    Remove duplicate rows from dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe without duplicates
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed_rows = initial_rows - len(df_clean)
    
    print(f"Removed {removed_rows} duplicate rows")
    
    return df_clean


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    strategy : str, default='drop'
        Strategy to handle missing values ('drop', 'fill', or 'flag')
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    if strategy == 'drop':
        df_clean = df.dropna()
        print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
        return df_clean
    
    elif strategy == 'fill':
        df_clean = df.fillna('')
        print("Filled missing values with empty strings")
        return df_clean
    
    elif strategy == 'flag':
        df_clean = df.copy()
        df_clean['has_missing'] = df.isnull().any(axis=1)
        print("Added flag column for missing values")
        return df_clean
    
    else:
        raise ValueError("Strategy must be 'drop', 'fill', or 'flag'")


def load_and_preprocess_data(filepath, drop_cols=None):
    """
    Load dataset and perform initial preprocessing.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    drop_cols : list, optional
        List of columns to drop
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"Loaded {len(df)} rows")
    
    # Drop specified columns
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
        print(f"Dropped columns: {drop_cols}")
    
    # Handle missing values
    df = handle_missing_values(df, strategy='drop')
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    print(f"Final dataset shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Preprocessing module loaded successfully!")
