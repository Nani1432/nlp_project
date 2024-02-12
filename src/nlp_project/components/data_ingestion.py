import os
from urllib import request, error
import zipfile
from src.nlp_project.logging import logger
from src.nlp_project.utils.common import get_size
from pathlib import Path
from src.nlp_project.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        try:
            if not os.path.exists(self.config.local_data_file):
                filename, headers = request.urlretrieve(
                    url=self.config.source_url,
                    filename=self.config.local_data_file
                )
                logger.info(f"{filename} downloaded with the following info:\n{headers}")
            else:
                logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")

        except error.URLError as e:
            logger.error(f"URLError: {e.reason}")
            # Handle specific URLError cases if needed

        except error.HTTPError as e:
            logger.error(f"HTTPError: {e.code} - {e.reason}")
            # Handle specific HTTPError cases if needed

        except PermissionError:
            logger.error(f"PermissionError: File already exists and cannot be overwritten.")
            # Handle PermissionError if needed

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            # Handle other exceptions if needed

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)