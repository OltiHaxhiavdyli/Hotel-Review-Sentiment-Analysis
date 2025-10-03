# Hotel-Review-Sentiment-Analysis

## Goal
This project demonstrates a simple AI/ML pipeline for text classification, focusing on **sentiment analysis of hotel reviews**. The objective is to preprocess text data, train models, evaluate their performance, and provide a prediction script.

## Dataset
The dataset contains **515,000 customer reviews** and scores of luxury hotels across Europe, collected from Booking.com.  

**Fields used:**
- `Positive_Review` – the positive text from the reviewer
- `Negative_Review` – the negative text from the reviewer
- `Reviewer_Score` – score given by the reviewer (used to generate sentiment label)

**Dataset link/source:** [Hotel Reviews Dataset](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

### Sentiment Labeling
- Reviews with `Reviewer_Score >= 8` → `positive`  
- Reviews with `Reviewer_Score < 8` → `negative`  

---

## Project Features

### 1. Data Preparation
- Loaded dataset and inspected columns.
- Combined positive and negative reviews into a single text column.
- Removed "No Positive"/"No Negative" placeholders.
- Preprocessed text: lowercased, removed punctuation and numbers, removed stopwords, lemmatization.

### 2. Exploratory Analysis
- Visualized sentiment distribution.
- Created WordClouds for positive and negative reviews.
- Listed the 20 most frequent words across all reviews.

### 3. Model Training
- Split data into training and test sets.
- TF-IDF vectorization with 1000 features.
- Trained three models:
  - **Logistic Regression**
  - **Multinomial Naive Bayes**
  - **MLP Neural Network** (optional)
- Evaluated accuracy, precision, recall, F1-score.
- Plotted confusion matrices.

### 4. Prediction
- `predict_sentiment(text, model)` function allows a user to input a review and get the predicted sentiment.

### 5. Bonus (Implemented)
- TF-IDF used instead of raw bag-of-words.
- Comparison of Logistic Regression vs Naive Bayes performance.
- Confusion matrices for all models.

---

## How to Run
1. Clone the repository:

```bash
git clone https://github.com/yourusername/hotel-review-sentiment.git
cd hotel-review-sentiment
```

2. Install Dependencies

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud joblib
```

3. Download the dataset and place it in the project folder. Update the path in the notebook if needed.

4. Open the Jupyter Notebook and run all cells sequentially.

5. To predict sentiment for a new review:

```bash
from predict import predict_sentiment

review = "The hotel was amazing, very clean and friendly staff."
sentiment = predict_sentiment(review)
print(sentiment)
```

## Models Saved

logreg_hotel_model.pkl – Logistic Regression

nb_hotel_model.pkl – Naive Bayes

mlp_hotel_model.pkl – MLP Neural Network

tfidf_vectorizer.pkl – TF-IDF vectorizer

## Results

Logistic Regression accuracy: ~79%

Naive Bayes accuracy: ~77%
