import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join("artifacts","train.csv")
    test_data_path: str=os.path.join("artifacts","test.csv")
    raw_data_path: str=os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        df=pd.read_csv("jupyter/data.csv")
        logging.info("Data read successfully.")
        os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist=True)
        df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
        df1=pd.read_csv(self.ingestion_config.raw_data_path.pre_cleaning.csv)

        logging.info("Train Test split initiated")
        X_train,


