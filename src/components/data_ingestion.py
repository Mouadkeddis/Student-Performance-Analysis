import os
import sys
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

@dataclass

class DataIngestionConfig:
    train_set_path : str=os.path.join('artifacts','train.csv')
    test_set_path : str=os.path.join('artifacts','test.csv')
    raw_set_path : str=os.path.join('artifacts','data.csv')