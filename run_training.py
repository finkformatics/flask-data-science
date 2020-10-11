from news_model import NewsTrainer
from pprint import pprint
from settings import Settings


def main() -> None:
    print('Loading settings')
    settings_instance = Settings.build_from_file('./settings.toml')

    print('Training model')
    news_trainer = NewsTrainer(settings_instance)
    news_trainer.train(settings_instance.data_path)

    print('Training stats:')
    stats = news_trainer.get_training_stats()
    print('Train Accuracy:', stats.train_accuracy)
    print('Test Accuracy:', stats.test_accuracy)
    print('Classification Report:')
    pprint(stats.classification_report)

    print('Saving model')
    news_trainer.save_model(settings_instance.model_path, settings_instance.tf_idf_path)


if __name__ == '__main__':
    main()
