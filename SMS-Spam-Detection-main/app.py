# sms_message_classification.py
# SMS Spam Message Classifier (No CSV Needed)
# Author: sujan
# Algorithm: Multinomial Naive Bayes

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords (only runs once)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Step 1: Small inbuilt dataset
data = {
    'label': ['spam', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam'],
    'message': [
        "Congratulations! You won a $1000 Walmart gift card. Click to claim now.",
        "Hey, are we still on for dinner tonight?",
        "I'll call you later when I'm free.",
        "URGENT! Your account is suspended. Verify your identity immediately.",
        "Don't forget to bring the documents tomorrow.",
        "You’ve been selected for a free vacation! Call now to redeem.",
        "Let's go for a movie this weekend!",
        "You have won lottery worth $5000. Send bank details to claim.",
        "Meeting postponed to 3 PM. Please confirm.",
        "Get cheap loans instantly, apply online now."
    ]
}

df = pd.DataFrame(data)

# Step 2: Clean and preprocess messages
df['message'] = df['message'].str.lower()
df['message'] = df['message'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df['message'] = df['message'].apply(lambda x: ' '.join(
    [word for word in x.split() if word not in stop_words]
))
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Convert text to vectors
cv = CountVectorizer()
X = cv.fit_transform(df['message'])
y = df['label_num']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
print("\n📊 Model Evaluation:")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 2))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Function to predict new messages
def predict_message(text):
    text = text.lower()
    text = ''.join([c for c in text if c.isalpha() or c.isspace()])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    vector = cv.transform([text])
    pred = model.predict(vector)[0]
    return "Spam" if pred == 1 else "Not Spam"

# Step 8: Test predictions
print("\n--- 🔍 Sample Predictions ---")
print("1:", predict_message("Win a free iPhone! Click here to claim now."))
print("2:", predict_message("Hey, can you send me the report by evening?"))
print("3:", predict_message("Limited offer! Buy 1 get 1 free."))
print("4:", predict_message("Let's catch up tomorrow at the cafe."))

