# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score
# from joblib import dump

# # Load dataset
# def load_data():
#     url = 'C:\\Users\\kk\\Desktop\\training.1600000.processed.noemoticon.csv'  # Your dataset path
#     data = pd.read_csv(url, encoding='ISO-8859-1', names=['target', 'id', 'date', 'flag', 'user', 'text'])
#     return data

# # Preprocess text
# def preprocess_text(text):
#     return text.lower()  # You can add more preprocessing steps if necessary

# # Load data and preprocess
# data = load_data()
# data['text'] = data['text'].apply(preprocess_text)

# # Split the data into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

# # Convert text data into TF-IDF features
# vectorizer = TfidfVectorizer(max_features=5000)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Train a Naive Bayes model
# model = MultinomialNB()
# model.fit(X_train_tfidf, y_train)

# # Evaluate the model
# predictions = model.predict(X_test_tfidf)
# accuracy = accuracy_score(y_test, predictions)
# print(f'Model Accuracy: {accuracy * 100:.2f}%')

# # Save the trained model and vectorizer for deployment
# dump(model, 'sentiment_model.pkl')
# dump(vectorizer, 'tfidf_vectorizer.pkl')
# print("Model and vectorizer saved successfully!")
