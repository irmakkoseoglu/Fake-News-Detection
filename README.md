# 🔍 Fake News Detection Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

A comprehensive machine learning project that detects fake news articles using Natural Language Processing (NLP) techniques and multiple classification algorithms.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview

In the digital age, misinformation spreads rapidly across social media and news platforms. This project implements a machine learning solution to automatically classify news articles as real or fake, helping combat the spread of misinformation.

The project explores multiple machine learning algorithms including:
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Multi-Layer Perceptron (MLP/Neural Network)

## ✨ Features

- **Text Preprocessing**: Advanced NLP techniques for cleaning and preparing text data
- **Feature Engineering**: Custom features including:
  - TF-IDF vectorization for title and text
  - Capitalization patterns
  - Punctuation analysis
  - Sentence structure metrics
- **Multiple Models**: Comparison of 4 different machine learning algorithms
- **Hyperparameter Tuning**: GridSearchCV for optimal model performance
- **Visualization**: Comprehensive data exploration with word clouds and statistical plots
- **Model Persistence**: Save and load trained models for production use

## 📊 Dataset

The dataset contains news articles with the following features:
- **title**: The headline of the news article
- **text**: The full content of the article
- **label**: Binary classification (0 = Fake, 1 = Real)

**Dataset Statistics:**
- Total articles: ~40,000+ samples
- Balanced distribution between fake and real news
- Multiple topics and news categories

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Jupyter Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/fake_news_detection.ipynb`

3. Run all cells sequentially to:
   - Load and explore the data
   - Preprocess the text
   - Train multiple models
   - Evaluate performance
   - Make predictions

### Using Trained Models

```python
import pickle
import numpy as np
from scipy.sparse import hstack

# Load the saved model
with open('models/best_mlp_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load vectorizers and scaler
with open('models/vectorizer_title.pkl', 'rb') as f:
    vectorizer_title = pickle.load(f)
    
with open('models/vectorizer_text.pkl', 'rb') as f:
    vectorizer_text = pickle.load(f)
    
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make prediction
def predict_news(title, text):
    # Feature extraction
    title_caps = sum(1 for w in title.split() if w.isupper() and len(w) > 1)
    # ... (add other features)
    
    # Transform
    X_title = vectorizer_title.transform([title])
    X_text = vectorizer_text.transform([text])
    X_other = [[title_caps, ...]]  # other features
    
    X = hstack([X_title, X_text, X_other])
    X_scaled = scaler.transform(X)
    
    # Predict
    prediction = model.predict(X_scaled)[0]
    return "REAL NEWS" if prediction == 1 else "FAKE NEWS"
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 98.5% | 98.6% | 98.4% | 98.5% |
| Random Forest | 99.1% | 99.2% | 99.0% | 99.1% |
| **MLP (Best)** | **99.3%** | **99.4%** | **99.2%** | **99.3%** |

*Note: Metrics are from test set evaluation with 80-20 train-test split*

### Key Insights
- MLP Neural Network achieved the highest performance
- All models showed excellent generalization with >98% accuracy
- Feature engineering significantly improved model performance
- TF-IDF vectorization captured important linguistic patterns

## 📁 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── gitkeep/                    # Original dataset link
│             
│
├── notebooks/
│   └── fake_news_detection.ipynb  # Main analysis notebook
│
│
├── models/
│   ├── best_mlp_model.pkl     # Saved MLP model
│   ├── vectorizer_title.pkl   # Title vectorizer
│   ├── vectorizer_text.pkl    # Text vectorizer
│   └── scaler.pkl             # Feature scaler
│
│
├── docs/
│   └── methodology.md         # Detailed methodology
│
├── requirements.txt           # Python dependencies
├── README.md                 # Project documentation
├── LICENSE                   # MIT License
└── .gitignore               # Git ignore file
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning algorithms and tools
- **NLTK / SpaCy**: Natural language processing
- **Matplotlib & Seaborn**: Data visualization
- **WordCloud**: Text visualization
- **Jupyter Notebook**: Interactive development environment

## 📊 Results

### Key Findings

1. **Feature Importance**:
   - TF-IDF features from article text are most predictive
   - Title features contribute significantly to classification
   - Punctuation patterns (especially exclamation marks) are strong indicators

2. **Model Comparison**:
   - Neural Network (MLP) performed best overall
   - Random Forest showed strong performance with interpretability
   - All models demonstrated excellent accuracy (>98%)

3. **Challenges Overcome**:
   - Handling missing values and duplicates
   - Balancing model complexity vs. interpretability
   - Optimizing feature extraction pipeline

### Visualizations

#### Word Cloud - Fake News
Most common words in fake news titles show sensationalist language and clickbait patterns.

#### Word Cloud - Real News
Real news titles demonstrate more formal, factual language patterns.

## 🔮 Future Improvements

- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add multilingual support
- [ ] Create a web application for real-time predictions
- [ ] Incorporate source credibility analysis
- [ ] Add explainability features (LIME/SHAP)
- [ ] Expand dataset with more recent articles
- [ ] Deploy model as REST API

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

**Irmak Koseoglu**
- Email: rmakoseoglu@hotmail.com
- LinkedIn: [Irmak Koseoglu](www.linkedin.com/in/irmakkoseoglu)
- GitHub: [@irmakkoseoglu](https://github.com/irmakkoseoglu)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle - Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification/data)
- Inspired by research in NLP and fake news detection
- Thanks to the open-source community for amazing tools

---

⭐ If you found this project helpful, please consider giving it a star!

**Note**: This project is for educational and research purposes. Always verify news from multiple reliable sources.

