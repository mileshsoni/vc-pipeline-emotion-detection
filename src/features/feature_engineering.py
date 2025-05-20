import numpy as np
import pandas as pd
import yaml
import os
import logging

from sklearn.feature_extraction.text import CountVectorizer
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('feature_engineering_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> int:
    max_features = yaml.safe_load(open('params.yaml', 'r'))['feature_engineering']['max_features']
    logger.debug('loaded parameters successfully')
    return max_features

def fetch_data (data_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train_data = pd.read_csv(os.path.join(data_path, 'train_processed_data.csv') )
        test_data = pd.read_csv(os.path.join(data_path, 'test_processed_data.csv'))
        logger.debug('fetched data successfully')
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error('File not found at {}'.format(data_path))
        raise
    except Exception as e:
        logger.error('Unexpected error : {}'.format(e))
        raise
    
def fill_missing_values (train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data.fillna('', inplace = True)
    test_data.fillna('', inplace=True)
    logger.debug('filled missing values successfully')
    return train_data, test_data
    
def split_data_train_test (train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        return X_train, y_train, X_test, y_test

    except KeyError as e:
        logger.error('Column not found : {}'.format(e))
        raise
    except Exception as e:
        logger.error('Unexpected error occurred : {}'.format(e))
        raise

def feature_transformation ( max_features:int, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame, y_test:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)
    train_df = pd.DataFrame(X_train_bow.toarray())
    train_df['label'] = y_train
    test_df = pd.DataFrame(X_test_bow.toarray())
    test_df['label'] = y_test
    return train_df, test_df

def save_data (data_path: str, train_df:pd.DataFrame, test_df: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path)
        train_df.to_csv(os.path.join(data_path, "train_bow.csv"))
        test_df.to_csv(os.path.join(data_path, "test_bow.csv"))
    except Exception as e:
        logger.exception('Unexpected error : {}'.format(e))
        raise

def main():
    try:
        max_features = load_params('params.yaml')
        train_data, test_data = fetch_data('./data/interim')
        train_data, test_data = fill_missing_values(train_data, test_data)
        X_train, y_train, X_test, y_test = split_data_train_test(train_data, test_data)
        train_df, test_df = feature_transformation(max_features, X_train, y_train, X_test, y_test)
        data_path = os.path.join("data", "processed")
        save_data(data_path, train_df, test_df)
        logger.debug('feature engineering step complete successfully')
    except:
        logger.error("Failed to complete the feature engineering process : {}".format(e))
        raise
        
if __name__ == '__main__':
    main()