from src.nlp_project.config.configuration import ConfigurationManager
from src.nlp_project.components.model_evalution import ModelEvaluation
from src.nlp_project.logging import logger


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()