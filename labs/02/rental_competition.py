#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import numpy as np
import os
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics

import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn) C
    - year (0: 2011, 1: 2012) C
    - month (1-12) C
    - hour (0-23) C
    - holiday (binary indicator) C
    - day of week (0: Sun, 1: Mon, ..., 6: Sat) C
    - working day (binary indicator; a day is neither weekend nor holiday) C
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain) C
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1) R
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1) R
    - relative humidity (0-1 range) R
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()   

        def pipeline(train, test, degree):
            col_int = np.all(train.astype(int) == train, axis=0)
            transformer = sklearn.compose.ColumnTransformer([("Cat", sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), col_int), ("Std", sklearn.preprocessing.StandardScaler(), ~col_int)])
            polynomial = sklearn.preprocessing.PolynomialFeatures(degree, include_bias=False)
            pipeline = sklearn.pipeline.Pipeline([("transformer", transformer), ("polynomial", polynomial)])    
            train_data = pipeline.fit_transform(train)
            test_data = pipeline.transform(test)

            return train_data, test_data

        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(train.data, train.target, test_size=0.2, random_state=args.seed)
    
        for i in range(1, 4):
            X, X2 = pipeline(X_train, X_test, i)


            lambdas = np.geomspace(0.01, 10, num=500)
            rmses = [0 for i in range(len(lambdas))]
            for count, value in enumerate(lambdas):
                model = sklearn.linear_model.Ridge(value).fit(X, y_train)
                pred = model.predict(X2)
                rmses[count] = sklearn.metrics.mean_squared_error(y_test, pred, squared=False)
            
            best_rmse = min(rmses)
            best_rmse

            best_lambda = lambdas[rmses.index(best_rmse)]
            best_lambda

            print(best_lambda)

            X_final, X_f = pipeline(train.data, train.data, i)

            model = sklearn.linear_model.Ridge(best_lambda).fit(X_final, train.target)

            # Serialize the model.
            with lzma.open('rental_competition{a}.model'.format(a = i), "wb") as model_file:
                pickle.dump(model, model_file)

    else:
        def pipeline(train, test, d):
            col_int = np.all(test.astype(int) == test, axis=0)
            transformer = sklearn.compose.ColumnTransformer([("Cat", sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), col_int), ("Std", sklearn.preprocessing.StandardScaler(), ~col_int)])
            polynomial = sklearn.preprocessing.PolynomialFeatures(d, include_bias=False)
            pipeline = sklearn.pipeline.Pipeline([("transformer", transformer), ("polynomial", polynomial)])    
            train_data = pipeline.fit_transform(train)
            test_data = pipeline.transform(test)
            

            return train_data, test_data

        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        #alpha = np.ones((len(test.data), 1))
        #test.data = np.concatenate((test.data, alpha), axis=1) 

        test1, test12 = pipeline(test.data, test.data, 1)
        test2, test22 =  pipeline(test.data, test.data, 2)
        test3, test33 = pipeline(test.data, test.data, 3)

        #alpha = np.ones((len(test12), 1))
        #test12 = np.concatenate((test12, alpha), axis=1) 

        #alpha2 = np.ones((len(test22), 1))
        #test22 = np.concatenate((test22, alpha2), axis=1) 

        #alpha3 = np.ones((len(test33), 1))
        #test33 = np.concatenate((test33, alpha3), axis=1) 


        with lzma.open('rental_competition1.model', "rb") as model_file:
            model1 = pickle.load(model_file)
        with lzma.open('rental_competition2.model', "rb") as model_file:
            model2 = pickle.load(model_file)
        with lzma.open('rental_competition3.model', "rb") as model_file:
            model3 = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        print(test12.shape)
        pred1 = model1.predict(test12)
        pred2 = model2.predict(test22)
        pred3 = model3.predict(test33)
       

        rmse1 = sklearn.metrics.mean_squared_error(test.target, pred1, squared=False)
        rmse2 = sklearn.metrics.mean_squared_error(test.target, pred2, squared=False)
        rmse3 = sklearn.metrics.mean_squared_error(test.target, pred3, squared=False)

        print(rmse1, rmse2, rmse3)
        #best_rmse = min(rmse1, rmse2, rmse3)
        #print(best_rmse)
        #if best_rmse == rmse1:
            #return pred1
        #if best_rmse == rmse2:
            #return pred2
        #else:
            #return pred3
        print(rmse2, rmse1)
        return pred3


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
