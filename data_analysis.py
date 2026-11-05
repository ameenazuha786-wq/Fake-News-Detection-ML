
import pandas as pd

# Load the datasets
fake_news = pd.read_csv("Dataset/Fake.csv")
real_news = pd.read_csv("Dataset/True.csv")

# Display dataset information
print("FAKE NEWS DATASET")
print(fake_news.head())

print("\nREAL NEWS DATASET")
print(real_news.head())

# Display number of records
print("\nFake News Records:", len(fake_news))
print("Real News Records:", len(real_news))

# Display column names
print("\nColumns:", fake_news.columns.tolist())

# Check missing values
print("\nMissing values in Fake News:")
print(fake_news.isnull().sum())

print("\nMissing values in Real News:")
print(real_news.isnull().sum())

# Step 5: Combine both datasets

# Add labels
fake_news["label"] = 0
real_news["label"] = 1

# Combine datasets
news_data = pd.concat([fake_news, real_news], axis=0)

# Remove duplicate records
news_data = news_data.drop_duplicates()

# Remove missing values
news_data = news_data.dropna()

# Combine title and text
news_data["content"] = news_data["title"] + " " + news_data["text"]

# Display dataset information
print("\nCOMBINED DATASET")
print(news_data.head())

print("\nTotal News Records:", len(news_data))

print("\nLabel Distribution:")
print(news_data["label"].value_counts())

print("\nMissing Values:")
print(news_data.isnull().sum())

# Save cleaned dataset
news_data.to_csv("cleaned_news.csv", index=False)

print("\nCleaned dataset saved successfully!")
