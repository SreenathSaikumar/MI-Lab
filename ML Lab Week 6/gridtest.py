from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import *
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np


class SVM:

    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        data = pd.read_csv(self.dataset_path)

        # X-> Contains the features
        self.X = data.iloc[:, 0:-1]
        # y-> Contains all the targets
        self.y = data.iloc[:, -1]

    def solve(self):
        """
        Build an SVM model and fit on the training data
        The data has already been loaded in from the dataset_path

        Refrain to using SVC only (with any kernel of your choice)

        You are free to use any any pre-processing you wish to use
        Note: Use sklearn Pipeline to add the pre-processing as a step in the model pipeline
        Refrain to using sklearn Pipeline only not any other custom Pipeline if you are adding preprocessing

        Returns:
            Return the model itself or the pipeline(if using preprocessing)
        """

        # TODO
        param_grid={'C':[0.1,1,2,3,4.1,10,100,1000],'gamma':['auto',1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}
        scaler=StandardScaler()
        X=scaler.fit_transform(self.X)
        model=GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
        model.fit(X,self.y)
        print(model.best_params_)
        print(model.best_estimator_)
        return model
