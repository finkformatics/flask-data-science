from news_model import NewsPredictor
from settings import Settings
from typing import Any, Dict, List


class NewsService:
    def __init__(self, settings_instance: Settings):
        self.settings = settings_instance
        self.predictor = NewsPredictor(self.settings)

        self.predictor.load_model(self.settings.model_path, self.settings.tf_idf_path)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        return self.predictor.predict(texts)
