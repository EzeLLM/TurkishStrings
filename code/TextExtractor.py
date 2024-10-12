import FuncHub
import os
import fnmatch
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
from Data_Member import Data_Member
import csv
import tqdm
import pandas
import numpy as np
import re
import fastparquet
import nltk

class TextExtractor:
    def __init__(self,config) -> None:
        self.config = FuncHub.open_yaml(config,'TextExtractor')
        self.target_types = self.config['target_types']
        self.root_path = self.config['root_path']
        self.file_paths = self.path_finder()
        self.export_path = self.config['export_path']
        self.export_format = self.config['export_format']
        self.export_file_name = self.config['export_file_name']
        self.data_members = []
        self.ds = pandas.DataFrame(columns=['text','source','gpt4_tokens'])
        self.export_path = os.path.join(self.export_path, f'{self.export_file_name}.{self.export_format}')

    # returns paths of all files of target types in a root folder
    def path_finder(self):
        target_types = self.target_types
        root_path = self.root_path
        file_paths = []
        for root, dirs, files in os.walk(root_path):
            for file in files:
                file_extension = os.path.splitext(file)[1]
                if file_extension in target_types:
                    file_paths.append(os.path.join(root, file))
        return file_paths
    
    def extract_txt(self,filepath) -> str:
        return FuncHub.open_txt(filepath)
    
    def add_to_csv(self, data_member: Data_Member):
        csv_file = os.path.join(self.export_path, 'data.csv')
        with open(csv_file, 'a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=data_member.get_attrb().keys(), quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(data_member.get_attrb())

    def create_data_member(self,text,source):
        return Data_Member(text,source)
    
    def normalize_text(self,text):
        # for char in ['\n', '\t', '\r']:
        #     text = text.replace(char, ' ')
        
        text = text.replace('"', "'")
        return text

    def extract_epub(self,filepath) -> str:
        book = epub.read_epub(filepath)
        text_content = ''
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                text_content += soup.get_text()
        return text_content
    
    def extract_text(self,file_path):
        return FuncHub.open_text(file_path)
    
    def identify_and_remove_introduction(self,text):
        """
        abandoned for now
        """
        pass

    def extract(self, file_paths=None, export: bool = True):
        get_export_path = os.path.join(self.export_path, f'{self.export_file_name}.{self.export_format}')
        if file_paths is None:
            file_paths = self.file_paths

        new_rows = []
        for file_path in tqdm.tqdm(file_paths):
            file_extension = os.path.splitext(file_path)[1]
            if file_extension == '.txt':
                text = self.extract_txt(file_path)
            elif file_extension == '.epub':
                text = self.extract_epub(file_path)
            else:
                continue  # Skip unsupported file types

            source = file_path
            new_rows.append({
                'text': self.normalize_text(text),
                'source': file_path,
                'gpt4_tokens': FuncHub.tokenize(text).__len__(),
                'part': 'archive'
            })

        # Use concat instead of append
        self.ds = pandas.concat([self.ds, pandas.DataFrame(new_rows)], ignore_index=True)
        if export:
            if self.export_format == 'csv':
                self.ds.to_csv(self.export_path, index=False)
            elif self.export_format == 'parquet':
                self.ds.to_parquet(self.export_path, index=False)
            else:
                raise ValueError('Unsupported format')
        return self.ds



if __name__ == "__main__":
    config = 'config/ts.yml'
    TE = TextExtractor(config)
    TE.extract() #  csv, parquet

# import os

# for root, dirs, files in os.walk('data'):
#     print("Current directory:", root)
#     print("Subdirectories:", dirs)
#     print("Files:", files)




