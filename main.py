import pandas as pd

# Load the datasets
fake_df = pd.read_csv('data/fake.csv')
real_df = pd.read_csv('data/real.csv')

# Add a label column to each
fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

# Combine the datasets
df = pd.concat([fake_df, real_df], ignore_index=True)

# Show basic info
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nLabel distribution:")
print(df['label'].value_counts())

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')

# Use only the 'text' and 'label' columns
X = df['text']
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF Vectorizer with stopwords
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_df=0.7)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF transformation completed!")
print("Training data shape:", X_train_tfidf.shape)
print("Test data shape:", X_test_tfidf.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# Function to predict if a news article is FAKE or REAL
def predict_news(news_text):
    vector = vectorizer.transform([news_text])  # Transform text to TF-IDF
    prediction = model.predict(vector)          # Predict label
    return prediction[0]

# Test it with your own input
sample_news = "The Prime Minister announced a new policy today aimed at increasing employment across rural areas."
print("\nPrediction for sample news:")
print("This news is:", predict_news(sample_news))







# -----------------------------------------------------------
#  Test News Samples (for manual testing)
# -----------------------------------------------------------

#  REAL News Examples:
# 1. India launches its first solar mission, Aditya-L1, to study the Sun after the successful landing of Chandrayaan-3.
# 2. The Reserve Bank of India announced that the repo rate will remain unchanged at 6.5% during the latest monetary policy meeting.
# 3. Scientists in Japan have created a new type of battery that can fully charge in under 10 minutes.

#  FAKE News Examples:
# 4. NASA confirms the discovery of a secret alien base on the dark side of the Moon.
# 5. Drinking boiled Coca-Cola cures diabetes, says viral WhatsApp message.
# 6. Government to ban internet for a full week to test new surveillance system, claims anonymous source.

# -----------------------------------------------------------
# Example usage:
sample_news = "NASA confirms the discovery of a secret alien base on the dark side of the Moon."
print("This news is:", predict_news(sample_news))
