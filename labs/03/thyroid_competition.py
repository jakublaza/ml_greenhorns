#!/usr/bin/env python3
# 4c2f5d6e-10bb-4eaa-baa5-09d6adfa7ffa
# e20c7b3d-4d2c-4b16-8971-110bcf9fbeab
# 7e271e04-8848-11e7-a75c-005056020108

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics
import sklearn.decomposition


import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
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
        dataset = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        
        
        col_int = np.all(dataset.data.astype(int) == dataset.data, axis=0)
        transformer = sklearn.compose.ColumnTransformer([("Cat", sklearn.preprocessing.OneHotEncoder(sparse=False, handle_unknown="ignore"), col_int), ("Std", sklearn.preprocessing.StandardScaler(), ~col_int)])
        polynomial = sklearn.preprocessing.PolynomialFeatures(include_bias=False)
        pca = sklearn.decomposition.PCA()
        model = sklearn.linear_model.LogisticRegression(random_state=args.seed)
        pipeline = sklearn.pipeline.Pipeline([("transformer", transformer), ("polynomial", polynomial), ('pca', pca), ('logistic reg', model)])    
        
        CV = sklearn.model_selection.GridSearchCV(pipeline, {'polynomial__degree':[3], 'logistic reg__C':[700], "pca__n_components": [550]}, 
                                     refit = True)
        
        model = CV.fit(dataset.data, dataset.target)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)


        predictions = model.predict(test.data)


        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
