from src.nlp_project.config.configuration import ConfigurationManager
from src.nlp_project.components.data_transformation import DataTransformation
from src.nlp_project.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self) -> None:
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        data_transformation.convert()