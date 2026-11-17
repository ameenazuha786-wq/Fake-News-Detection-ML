# Fake News Detection Using Machine Learning

## Project Overview
Fake News Detection is a machine learning project developed to classify news articles as Fake or Real using Natural Language Processing (NLP).

The system analyzes news text and predicts its category using TF-IDF feature extraction and Logistic Regression.

## Objectives
- Identify potentially fake news articles.
- Classify news as Fake or Real.
- Apply Natural Language Processing techniques.
- Build an interactive web application for news prediction.

## Technologies Used
- Python
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Scikit-learn
- Streamlit
- Pandas

## Project Structure

    Fake_News_Detection/
    ├── Dataset/
    │   ├── Fake.csv
    │   └── True.csv
    ├── app.py
    ├── data_analysis.py
    ├── train_model.py
    ├── fake_news_model.pkl
    ├── tfidf_vectorizer.pkl
    └── README.md

## Methodology
1. Dataset Collection
2. Data Preprocessing
3. Text Cleaning
4. TF-IDF Feature Extraction
5. Logistic Regression Model Training
6. Model Evaluation
7. News Classification
8. Streamlit Web Application Development

## Machine Learning Algorithm

### TF-IDF
Converts text into numerical features based on word importance.

### Logistic Regression
Classifies processed news text into Fake or Real categories.

## Installation and Execution

### 1. Clone Repository

    git clone https://github.com/ameenazuha786-wq/Fake-News-Detection-ML.git

### 2. Install Dependencies

    pip install pandas scikit-learn streamlit

### 3. Run Application

    streamlit run app.py

## Project Output
The application accepts news text and predicts whether the news is Fake or Real.

## Disclaimer
This is a machine learning prediction and not independent verification of news.

## Developed By
Ameena Zuha

Machine Learning Project
