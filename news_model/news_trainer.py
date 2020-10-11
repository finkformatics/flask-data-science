import pandas as pd
import pickle

from dataclasses import dataclass
from nlp import TextProcessing
from settings import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from typing import Any, Dict


@dataclass
class TrainingStats:
    train_accuracy: float
    test_accuracy: float
    classification_report: Dict[str, Any]


class NewsTrainer:
    def __init__(self, settings_instance: Settings):
        self.settings = settings_instance
        self.category_codes = {
            'business': 0,
            'entertainment': 1,
            'politics': 2,
            'sport': 3,
            'tech': 4,
        }

        self.model = None
        self.tf_idf = None
        self.training_stats = None

    def get_training_stats(self) -> TrainingStats:
        return self.training_stats

    def save_model(self, path_model: str, path_tf_idf: str) -> None:
        with open(path_model, 'wb') as f_model, open(path_tf_idf, 'wb') as f_tf_idf:
            pickle.dump(self.model, f_model)
            pickle.dump(self.tf_idf, f_tf_idf)

    def train(self, path_to_data_file: str) -> None:
        # Load data
        df = pd.read_csv(path_to_data_file)

        # Pre process texts
        text_processing = TextProcessing()
        text_processing.initialize_libraries()
        df = text_processing.pre_process_text_df(df)

        # Label coding
        df['Category_Code'] = df['Category']
        df = df.replace({'Category_Code': self.category_codes})

        # Train/Test split
        x_train, x_test, y_train, y_test = train_test_split(
            df['Text_Processed'],
            df['Category_Code'],
            test_size=0.15
        )

        # Feature Engineering
        self.tf_idf = TfidfVectorizer(
            encoding='utf-8',
            ngram_range=(1, 1),
            stop_words=None,
            lowercase=False,
            max_df=1.0,
            min_df=10,
            max_features=300,
            norm='l2',
            sublinear_tf=True
        )

        self.tf_idf.fit(df['Text_Processed'])

        features_train = self.tf_idf.transform(x_train).toarray()
        labels_train = y_train

        features_test = self.tf_idf.transform(x_test).toarray()
        labels_test = y_test

        # Modelling
        self.model = LogisticRegression(
            C=self.settings.model_params.C,
            penalty=self.settings.model_params.penalty,
            class_weight=self.settings.model_params.class_weight,
            solver=self.settings.model_params.solver,
            multi_class=self.settings.model_params.multi_class
        )

        self.model.fit(features_train, labels_train)

        # Store model performance
        train_predictions = self.model.predict(features_train)
        test_predictions = self.model.predict(features_test)

        self.training_stats = TrainingStats(
            train_accuracy=accuracy_score(labels_train, train_predictions),
            test_accuracy=accuracy_score(labels_test, test_predictions),
            classification_report=classification_report(
                labels_test,
                test_predictions,
                target_names=list(self.category_codes.keys()),
                output_dict=True
            )
        )
