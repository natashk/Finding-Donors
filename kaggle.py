import numpy as np
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib

def prepare_data(df):
    '''
    INPUT:
    df - DataFrame

    OUTPUT:
    features_final - DataFrame, cleaned and prepared for training
    '''

    df_clean = df.fillna(df.mean())

    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = df_clean)
    features_log_transformed[skewed] = df_clean[skewed].apply(lambda x: np.log(x + 1))

    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])


    # TODO: One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_log_minmax_transform)
    
    return features_final


def save_model(model, model_filepath):
    '''
    INPUT:
    model - machine learning pipeline
    model_filepath - the filepath of the pickle file to save the model to

    OUTPUT:
    Saves the model as a pickle file
    '''

    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    # load test data set
    kaggle_test = pd.read_csv('test_census.csv').drop(['Unnamed: 0'],axis=1)
    kaggle_test = prepare_data(kaggle_test)

    # load model
    model = joblib.load('model_gbc.pkl')

    start = time()
    predictions_kaggle = model.predict(kaggle_test)
    end = time()
    print(f'Predicting time: {end-start} sec')

    # save predictions
    output = pd.DataFrame({'id': kaggle_test.index, 'income': predictions_kaggle})
    output.to_csv('kaggle_submission.csv', index=False)
    print(output.head())


if __name__ == '__main__':
    main()
