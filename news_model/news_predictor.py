import numpy as np
import pickle

from nlp import TextProcessing
from settings import Settings
from typing import Any, Dict, List


class NewsPredictor:
    def __init__(self, settings_instance: Settings):
        self.settings = settings_instance
        self.category_codes = {
            'business': 0,
            'entertainment': 1,
            'politics': 2,
            'sport': 3,
            'tech': 4,
        }

        self.category_names = {v: k for k, v in self.category_codes.items()}

        self.model = None
        self.tf_idf = None

        self.text_processing = TextProcessing()

    def load_model(self, path_to_model: str, path_to_tf_idf: str) -> None:
        with open(path_to_model, 'rb') as f_model, open(path_to_tf_idf, 'rb') as f_tf_idf:
            self.model = pickle.load(f_model)
            self.tf_idf = pickle.load(f_tf_idf)

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        df = self.text_processing.pre_process_text_list(texts)
        features = self.tf_idf.transform(df['Text_Processed']).toarray()
        prediction_probabilities = self.model.predict_proba(features)

        predictions = []
        for prob in prediction_probabilities:
            max_index = np.argmax(prob)
            max_prob = prob[max_index]
            prediction = {
                'prediction': self.category_names[max_index],
                'probability': round(max_prob, 6),
            }
            predictions.append(prediction)

        return predictions
