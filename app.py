from flask import Flask, render_template, request, jsonify
from database import DatabaseManager
from model_training import SentimentAnalyzer
import os

app = Flask(__name__)
db = DatabaseManager()
analyzer = SentimentAnalyzer(db)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get('review', '')
    model_name = data.get('model', 'Logistic Regression')
    
    if not review.strip():
        return jsonify({'error': 'Please enter a review'}), 400
    
    sentiment, confidence = analyzer.predict_sentiment(review, model_name)
    cleaned_text = analyzer.clean_text(review)
    db.save_prediction(review, cleaned_text, sentiment, confidence, model_name)
    
    return jsonify({'sentiment': sentiment, 'confidence': confidence, 'model_used': model_name})

@app.route('/api/stats')
def get_stats():
    stats = db.get_sentiment_stats()
    return jsonify(stats.to_dict('records'))

@app.route('/api/trends')
def get_trends():
    stats = db.get_sentiment_stats()
    if len(stats) > 0:
        return jsonify({
            'dates': stats['date'].astype(str).tolist(),
            'positive': stats['positive'].tolist(),
            'negative': stats['negative'].tolist(),
            'neutral': stats['neutral'].tolist(),
            'total': stats['total'].tolist()
        })
    return jsonify({'dates': [], 'positive': [], 'negative': [], 'neutral': [], 'total': []})

@app.route('/api/models')
def get_models():
    return jsonify(['Logistic Regression', 'Naive Bayes', 'SVM'])

@app.route('/api/predictions/recent')
def get_recent_predictions():
    df = db.get_all_predictions()
    return jsonify(df.head(20).to_dict('records'))

if __name__ == '__main__':
    if not os.path.exists('models/logistic_regression.pkl'):
        print("Training models...")
        analyzer.train_models()
    print("Server running at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)