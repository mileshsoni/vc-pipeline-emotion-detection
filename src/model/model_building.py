import numpy as np
import pandas as pd
import pickle
import yaml
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import ClassifierMixin
import logging

logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params (params_path : str) -> tuple[int, int]:
    try:
        params = yaml.safe_load(open(params_path, 'r'))['model_building']
        n_estimators = params['n_estimators']
        learning_rate = params['learning_rate']
        logger.debug('parameters loaded successfully')
        return n_estimators, learning_rate
    except FileNotFoundError as e:
        logger.error('file not found at {} : {}'.format(params_path, e))
        raise
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise

def fetch_data(data_path: str) -> pd.DataFrame: 
    try:
        train_data = pd.read_csv(os.path.join(data_path, 'train_tfidf.csv'))
        logger.debug('Data fetched successfully')
        return train_data
    except FileNotFoundError as e:
        logger.error('File not found at path {} : {}'.format(data_path, e))
        raise
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise
def x_y_split (train_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    x_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    return x_train, y_train

def train_model(n_estimators: int, learning_rate: float, x_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        clf.fit(x_train, y_train)
        logger.debug('Model trained successfully')
        return clf
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise

def save_model (clf : ClassifierMixin, path:str) -> None:
    try:
        pickle.dump(clf, open(path, 'wb'))
        logger.debug('Model saved successfully')
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise

def main() -> None:
    try:
        n_estimators, learning_rate = load_params('params.yaml')
        train_data = fetch_data('./data/processed')
        x_train, y_train = x_y_split(train_data)
        clf = train_model(n_estimators, learning_rate, x_train, y_train)
        save_model(clf, 'models/model.pkl')
        logger.debug('data preprocessing step completed successfully')
    except Exception as e:
        logger.error('Failed to complete the model building process : {}'.format(e))
        
        
if __name__ == '__main__':
    main()
    

