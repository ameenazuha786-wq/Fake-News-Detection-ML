# Fake News Detection System Using Machine Learning

## Project Overview
The Fake News Detection System is a machine learning-based application that predicts whether a given news article is REAL or FAKE.

The system uses Natural Language Processing (NLP), TF-IDF Vectorization, and Logistic Regression to analyze news text and generate predictions with a confidence score.

## Objectives
- To identify fake and real news articles.
- To preprocess and analyze news text.
- To implement a machine learning classification model.
- To develop a simple and interactive web application.
- To display prediction results and model confidence.

## Technologies Used
- Python
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF Vectorization
- Logistic Regression
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- VS Code

## Algorithm Used

### 1. TF-IDF Vectorization
TF-IDF converts text into numerical features based on the importance of words in the news articles.

### 2. Logistic Regression
Logistic Regression is used as a classification algorithm to predict whether the given news article is REAL or FAKE.

## Project Structure

```text
Fake_News_Detection/
│
├── Dataset/
│   ├── Fake.csv
│   └── True.csv
│
├── Screenshots/
│   ├── Fake_News_Test.png
│   ├── Fake_News_Test1.png
│   ├── Real_News_test.png
│   └── homepage.png
│
├── app.py
├── data_analysis.py
├── train_model.py
├── cleaned_news.csv
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

## Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/ameenazuha786-wq/Fake-News-Detection-ML.git
```

### Step 2: Open the Project Folder

```bash
cd Fake-News-Detection-ML
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will open in your web browser.

## How It Works
1. Enter a news headline or article.
2. Click the Predict News button.
3. The application processes the text.
4. TF-IDF converts the text into numerical features.
5. Logistic Regression predicts the news category.
6. The application displays the prediction and confidence score.

## Screenshots

The Screenshots folder contains:
- Homepage of the application.
- Fake news testing results.
- Real news testing results.

## Results
The application successfully runs and generates predictions for the entered news articles.

**Note:** Predictions are based on the trained machine learning model and should not be considered independent verification of news.

## Future Enhancements
- Improve model accuracy using larger and more diverse datasets.
- Explore advanced machine learning and deep learning algorithms.
- Integrate real-time news verification sources.
- Develop a more advanced user interface.

## Author
**Ameena Zuha**

## GitHub Repository
https://github.com/ameenazuha786-wq/Fake-News-Detection-ML

---
Developed as a Machine Learning project using Python, TF-IDF and Logistic Regression.
