from flask import request, jsonify, render_template
from flask_app import app
from settings import Settings

from .news_service import NewsService

settings_instance = Settings.build_from_file('./settings.toml')
news_service = NewsService(settings_instance)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/v1/ping')
def ping() -> str:
    return 'pong'


@app.route('/api/v1/news/predict', methods=['POST'])
def predict_news():
    news_texts = request.json
    predictions = news_service.predict(news_texts)

    return jsonify(predictions)
