from datetime import datetime
import sqlite3
import pandas as pd

class DatabaseManager:
    def __init__(self, db_file='alexa_reviews.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_text TEXT,
                cleaned_text TEXT,
                sentiment TEXT,
                confidence REAL,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                accuracy REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                training_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_prediction(self, review_text, cleaned_text, sentiment, confidence, model_used):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (review_text, cleaned_text, sentiment, confidence, model_used)
            VALUES (?, ?, ?, ?, ?)
        ''', (review_text, cleaned_text, sentiment, confidence, model_used))
        self.conn.commit()
    
    def save_model_performance(self, model_name, accuracy, precision, recall, f1):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO model_performance (model_name, accuracy, precision, recall, f1_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (model_name, accuracy, precision, recall, f1))
        self.conn.commit()
    
    def get_all_predictions(self):
        return pd.read_sql("SELECT * FROM predictions ORDER BY timestamp DESC", self.conn)
    
    def get_sentiment_stats(self):
        query = '''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as total,
                SUM(CASE WHEN sentiment = 'Positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment = 'Negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment = 'Neutral' THEN 1 ELSE 0 END) as neutral
            FROM predictions
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
        '''
        return pd.read_sql(query, self.conn)
    
    def close(self):
        self.conn.close()