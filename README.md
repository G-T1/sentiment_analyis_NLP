# sentiment_analyis_NLP
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import nltk

# Download NLTK data (ensure this runs only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample Dataset
df = pd.read_csv("C:/Users/Gnan Tejas D/OneDrive/Desktop/reviews.csv")

print(df.head())
print(df['Sentiment'].value_counts())



# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in tokens
                      if word not in stop_words and word not in string.punctuation]
    return " ".join(cleaned_tokens)

# Apply Preprocessing
df["Cleaned Review"] = df["Review Text"].apply(preprocess_text)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["Cleaned Review"])
y = df["Sentiment"].apply(lambda x: 1 if x == "positive" else 0)  # Encode Sentiment

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Output Evaluation
print("Model Evaluation Metrics:")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-Score :", f1)

# Confusion Matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Sentiment Analysis")
plt.show()

# Insights
df["Predicted Sentiment"] = model.predict(X)
df["Predicted Sentiment"] = df["Predicted Sentiment"].apply(lambda x: "positive" if x == 1 else "negative")

print("\nCorrectly Classified Reviews:")
print(df[df["Sentiment"] == df["Predicted Sentiment"]][["Review Text", "Sentiment"]])

print("\nIncorrectly Classified Reviews:")
print(df[df["Sentiment"] != df["Predicted Sentiment"]][["Review Text", "Sentiment", "Predicted Sentiment"]])
