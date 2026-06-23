import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

nltk.download('punkt')
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.models = {}
        self.vectorizer = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
    
    def create_sample_data(self):
        reviews = [
            ("Great product! Works perfectly.", "Positive"),
            ("Amazing device, highly recommended!", "Positive"),
            ("Love it! Best purchase ever.", "Positive"),
            ("Excellent quality and fast shipping.", "Positive"),
            ("Fantastic features and great support.", "Positive"),
            ("Best Alexa device I've ever used!", "Positive"),
            ("Amazing sound quality and response time.", "Positive"),
            ("Highly recommend this to everyone.", "Positive"),
            ("Works as expected, satisfied with purchase.", "Positive"),
            ("Absolutely love this device!", "Positive"),
            ("Terrible quality, broke after one use.", "Negative"),
            ("Poor customer service and bad quality.", "Negative"),
            ("Waste of money, doesn't work at all.", "Negative"),
            ("Very disappointed with this purchase.", "Negative"),
            ("Horrible experience, wouldn't recommend.", "Negative"),
            ("Frequently disconnects from WiFi.", "Negative"),
            ("Battery life is terrible.", "Negative"),
            ("Voice recognition is poor.", "Negative"),
            ("Cheap build quality.", "Negative"),
            ("Overpriced for what it offers.", "Negative"),
            ("Decent product, but could be better.", "Neutral"),
            ("Average product, nothing special.", "Neutral"),
            ("Good for basic use, lacks advanced features.", "Neutral"),
            ("It's okay, does the job.", "Neutral"),
            ("Works fine, nothing extraordinary.", "Neutral"),
        ]
        return pd.DataFrame(reviews, columns=['review', 'sentiment'])
    
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def train_models(self):
        print("Training models...")
        df = self.create_sample_data()
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        df = df[df['cleaned_review'].str.len() > 0]
        
        sentiment_map = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
        df['sentiment_encoded'] = df['sentiment'].map(sentiment_map)
        
        X = df['cleaned_review']
        y = df['sentiment_encoded']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.vectorizer = TfidfVectorizer(max_features=1000)
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'SVM': SVC(kernel='linear', probability=True)
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_vec, y_train)
            self.models[name] = model
            y_pred = model.predict(X_test_vec)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[name] = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
            
            os.makedirs('models', exist_ok=True)
            joblib.dump(model, f'models/{name.lower().replace(" ", "_")}.pkl')
            self.db_manager.save_model_performance(name, accuracy, precision, recall, f1)
            
            print(f"{name}: Accuracy = {accuracy:.2%}")
        
        joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        return results
    
    def predict_sentiment(self, text, model_name='Logistic Regression'):
        if model_name not in self.models:
            model_path = f'models/{model_name.lower().replace(" ", "_")}.pkl'
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
            else:
                return 'Neutral', 0.33
        
        if self.vectorizer is None:
            if os.path.exists('models/vectorizer.pkl'):
                self.vectorizer = joblib.load('models/vectorizer.pkl')
            else:
                return 'Neutral', 0.33
        
        cleaned_text = self.clean_text(text)
        if not cleaned_text:
            return 'Neutral', 0.33
        
        text_vec = self.vectorizer.transform([cleaned_text])
        model = self.models[model_name]
        sentiment_num = model.predict(text_vec)[0]
        
        try:
            confidence = max(model.predict_proba(text_vec)[0])
        except:
            confidence = 0.5
        
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        return sentiment_map.get(sentiment_num, 'Neutral'), float(confidence)