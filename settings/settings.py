import toml

from dataclasses import dataclass


@dataclass
class ModelParams:
    C: float
    class_weight: str
    multi_class: str
    penalty: str
    solver: str


@dataclass
class Settings:
    # General
    model_path: str
    tf_idf_path: str
    data_path: str

    # Training
    n_jobs: int

    # Training - Model params
    model_params: ModelParams

    @staticmethod
    def build_from_file(path_to_settings: str) -> 'Settings':
        with open(path_to_settings) as f:
            settings_data = toml.load(f)

        model_params = ModelParams(**settings_data['training']['model_params'])

        return Settings(
            model_path=settings_data['general']['model_path'],
            tf_idf_path=settings_data['general']['tf_idf_path'],
            data_path=settings_data['general']['data_path'],
            n_jobs=settings_data['training']['n_jobs'],
            model_params=model_params
        )
