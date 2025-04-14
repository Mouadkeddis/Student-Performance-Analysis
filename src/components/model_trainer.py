import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object
@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()

    def initiate_model_trainer(self, train_array, test_array , preprocessor_path):
        try:
            logging.info("split  training and test input data")
            X_train, y_train,X_test, y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree" : DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "adaboost regressor" : AdaBoostRegressor(),
                "Gradient Booster" : GradientBoostingRegressor(),
                "K-Nieghbors classifier" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "Catboosting classifier" : CatBoostRegressor(verbose=False),

            }
        except:
            pass