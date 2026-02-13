# Methodology

## Project Overview

This document provides a detailed explanation of the methodology used in the Fake News Detection project.

## 1. Data Collection & Exploration

### Dataset Description
- **Source**: News articles dataset
- **Size**: ~40,000+ articles
- **Features**:
  - `title`: Article headline
  - `text`: Full article content
  - `label`: Binary label (0 = Fake, 1 = Real)

### Exploratory Data Analysis
- Checked for missing values and duplicates
- Analyzed label distribution (balanced dataset)
- Created word clouds to visualize common words in fake vs. real news
- Examined text length distributions

## 2. Data Preprocessing

### Steps Performed:
1. **Handling Missing Values**: Removed rows with null values (~1.5% of data)
2. **Duplicate Removal**: Eliminated duplicate articles
3. **Text Cleaning**:
   - Converted to lowercase
   - Removed special characters
   - Normalized whitespace

## 3. Feature Engineering

### Custom Features:
1. **Title Features**:
   - `title_caps_words`: Count of fully capitalized words (clickbait indicator)
   - `title_exclamation`: Number of exclamation marks (sensationalism)
   - `title_avg_length`: Average sentence length in title

2. **Text Features**:
   - `text_exclamation`: Exclamation marks in body text
   - `text_question`: Question marks in body text

### TF-IDF Vectorization:
- **Title Vectorization**:
  - Max features: 500
  - N-gram range: (1, 2)
  - Stop words: English
  
- **Text Vectorization**:
  - Max features: 5000
  - N-gram range: (1, 2)
  - Stop words: English

### Feature Combination:
- Combined TF-IDF features from title and text
- Added custom engineered features
- Total feature space: ~5,505 features

## 4. Feature Scaling

- Used `StandardScaler` with `with_mean=False` (for sparse matrices)
- Scaled all features to have zero mean and unit variance
- Applied only to training data first, then transformed test data

## 5. Model Selection & Training

### Models Evaluated:
1. **Logistic Regression**
   - Simple baseline model
   - Linear decision boundary
   - Fast training and inference

2. **Random Forest**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Feature importance analysis

3. **Support Vector Machine (SVM)**
   - Linear kernel
   - Effective in high-dimensional spaces
   - Good generalization

4. **Multi-Layer Perceptron (MLP)**
   - Neural network architecture
   - Hidden layers: (100, 50)
   - Early stopping to prevent overfitting
   - **Best performing model**

### Train-Test Split:
- Ratio: 80% training, 20% testing
- Stratified split to maintain label distribution
- Random state: 42 (for reproducibility)

## 6. Hyperparameter Tuning

### MLP Hyperparameters (Best Model):
```python
{
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.0001,
    'learning_rate': 'adaptive',
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'random_state': 42
}
```

### Tuning Method:
- GridSearchCV with 5-fold cross-validation
- Scoring metric: Accuracy
- Explored multiple parameter combinations

## 7. Model Evaluation

### Metrics Used:
- **Accuracy**: Overall correctness
- **Precision**: Percentage of correctly identified fake/real news
- **Recall**: Ability to find all instances of each class
- **F1-Score**: Harmonic mean of precision and recall

### Results:
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.5% | 98.6% | 98.4% | 98.5% |
| Random Forest | 99.1% | 99.2% | 99.0% | 99.1% |
| SVM | 98.7% | 98.8% | 98.6% | 98.7% |
| **MLP** | **99.3%** | **99.4%** | **99.2%** | **99.3%** |

## 8. Model Interpretation

### Key Findings:
1. **TF-IDF Features**: Most discriminative for classification
2. **Title Features**: Strong indicators of fake news (caps, exclamation)
3. **Linguistic Patterns**: Fake news uses more sensational language
4. **Model Performance**: Deep learning (MLP) slightly outperforms traditional ML

### Feature Importance (from Random Forest):
- Top words in fake news: "trump", "breaking", "shocking"
- Top words in real news: "reuters", "washington", "government"

## 9. Model Deployment

### Saved Artifacts:
- Best MLP model: `best_mlp_model.pkl`
- Title vectorizer: `vectorizer_title.pkl`
- Text vectorizer: `vectorizer_text.pkl`
- Feature scaler: `scaler.pkl`

### Prediction Pipeline:
1. Extract custom features from title and text
2. Vectorize title and text using saved TF-IDF vectorizers
3. Combine all features
4. Scale features using saved scaler
5. Predict using trained MLP model
6. Return classification (Fake/Real)

## 10. Limitations & Future Work

### Current Limitations:
- Dataset may not represent all types of fake news
- Language-specific (English only)
- Binary classification (no uncertainty measure)
- Context and source credibility not considered

### Future Improvements:
- Implement BERT or other transformer models
- Add multi-language support
- Include source credibility analysis
- Incorporate temporal features
- Add explainability (LIME/SHAP)
- Real-time deployment via web API

## References

1. Natural Language Processing with Python (NLTK Book)
2. Scikit-learn Documentation
3. Research papers on fake news detection
4. TF-IDF and text classification techniques

---

*Last Updated: February 2025*
