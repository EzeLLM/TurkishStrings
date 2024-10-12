"""
This class is desinged to extract a random text from each data member in the dataset and store it in a new dataset.
The purpose is to create a labeled dataset for modernity scorer model.
"""

import pandas as pd
import random
import fastparquet
import FuncHub
import os
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import tiktoken
from torch import Tensor
import tqdm
import json

SAMPLE_CONSTANT = 5
def debug_(inp:str):
    print(inp)
    exit()
class RandomizedDatasetExtractor:
    def __init__(self,config) -> None:
        self.config = FuncHub.open_yaml(config,'RandomizedDatasetExtractor')
        self.te_config = FuncHub.open_yaml(config,'TextExtractor')
        self.export_path = os.path.join(self.te_config['export_path'], f'{self.te_config['export_file_name']}.{self.te_config['export_format']}')
        self.ds = fastparquet.ParquetFile(self.export_path).to_pandas()

        # sample rules
        self.sample_size = self.config['sample_size']
        self.sample_count = self.config['sample_count'] * SAMPLE_CONSTANT # extract SAMPLE_CONSTANT times the sample count but select the samples with the least similarity score to project back to the sample count
        self.sample_start = self.config['sampling_start']
        self.sample_end = self.config['sampling_end']

        # similarity score
        self.similarity_threshold = self.config['similarity_threshold']

        self.tfidf = TfidfVectorizer(min_df=1, stop_words="english") # min_df=1 means that the word must appear at least once in the dataset

        # tokenizer
        self.tokenizer = tiktoken.get_encoding('o200k_base')

        # Initialize chunk reader instead of loading the entire DataFrame
        self.parquet_file = fastparquet.ParquetFile(self.export_path)
        self.chunk_size = self.config.get('chunk_size', 1000)  # Set chunk size


    def similrity_score(self,docs: list):
        vectorized = self.tfidf.fit_transform(docs)
        return torch.tensor((vectorized*vectorized.T).toarray(), dtype=torch.float32)
    
    def purify_samples(self, samples, k=None):
        if k is None:
            k = self.sample_count//SAMPLE_CONSTANT
        """
        Purify the samples by selecting the samples with the least similarity score
        """
        sim_score: Tensor = self.similrity_score(samples)
        sim_score = sim_score.fill_diagonal_(0)
        sim_score_summed = sim_score.sum(dim=1)
        min_sim_score_indices = sim_score_summed.topk(k, largest=False)  # Get the top k smallest elements
        # debug_(min_sim_score_indices)
        samples = [samples[i] for i in min_sim_score_indices.indices]
        return samples

            
    
    def extract_sample(self,ds_record):
        """
        extract randomly sampled text chunks from the text field in the dataset according to the sample rules
        
        static_rule: extract 5 times the sample count and select the samples with least 
        """
        ds_record_text = ds_record['text']
        ds_record_text_len = len(ds_record_text.split())
        sample_start = int(self.sample_start*ds_record_text_len)
        sample_end = int(self.sample_end*ds_record_text_len)
        random_indices = random.sample(range(sample_start,sample_end),self.sample_count)
        samples = self.purify_samples([' '.join(ds_record_text.split()[i:i+self.sample_size]) for i in random_indices])
        return samples
    
    def process_chunk(self, chunk):
        # debug_([self.extract_sample(record) for _, record in chunk.iterrows()])
        return [self.extract_sample(record) for _, record in chunk.iterrows()]

    def get_ds(self):
        results = []
        with multiprocessing.Pool() as pool:
            for chunk in self.parquet_file.iter_row_groups(columns=['text']):
                chunk_results = pool.map(self.process_chunk, [chunk])
                results.extend(chunk_results)
        return results


if __name__ == '__main__':
    rde = RandomizedDatasetExtractor('config/ts.yml')
    first_record_sample = rde.get_ds()
    print(first_record_sample)
    with open('export/sample_Set.json', 'w', encoding='utf-8') as file:
        json.dump(first_record_sample, file, ensure_ascii=False, indent=4)