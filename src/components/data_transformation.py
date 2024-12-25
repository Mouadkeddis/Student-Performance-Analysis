import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path =os.path.join('artifacts', "preprocessor.pk1")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig
    
    def get_data_transformation_object(self):
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns= [
                                  "gender",
                                  "race_ethnicity",
                                  "parental_level_of_education",
                                  "lunch",
                                  "test_preparation_course",
                                  ]
            num_pipline=Pipeline(
                steps=[
                 ("imputer",SimpleImputer(strategy="median")),
                 ("scaler",StandardScaler())

                ]
            )

            cat_pipline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipline",num_pipline,numerical_columns),
                    ("cat_pipline", cat_pipline, categorical_columns)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
           