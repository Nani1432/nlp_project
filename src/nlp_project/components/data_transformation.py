import os
from src.nlp_project.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from src.nlp_project.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        try:
            input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)

            with self.tokenizer.as_target_tokenizer():
                target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)

            return {
                'input_ids': input_encodings['input_ids'],
                'attention_mask': input_encodings['attention_mask'],
                'labels': target_encodings['input_ids']
            }
        except Exception as e:
            print(f"Error in convert_examples_to_features: {e}")
            return None  # You might want to handle this differently based on your use case

    def convert(self):
        try:
            dataset_samsum = load_from_disk(self.config.data_path)

            # Specify a different output directory
            output_dir = os.path.join(self.config.root_dir, "transformed_samsum_dataset")
            
            # Check if the output directory already exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
            dataset_samsum_pt.save_to_disk(output_dir)
        except Exception as e:
            print(f"Error in convert: {e}")