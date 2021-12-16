import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# from typing import List, Dict, Optional
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from Preprocessing.eda import *
from sklearn.svm import SVR

# df = read_dataset(Path('C:\\Visual_analytics\\assignments\\flask_music\\venv\Datasets\\songs_dataset_x_excel.csv'))
#
# sss = read_dataset(Path('C:\\Visual_analytics\\assignments\\flask_music\\venv\Datasets\\songs.csv'))

def Linear_Regression(x,y):
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    LR = LinearRegression()
    LR.fit(X_train,y_train)

    trainy_predict = LR.predict(X_train)
    testy_predict = LR.predict(X_test)


    print("MSE:", mean_squared_error(y_test, testy_predict))
    print("R2:", r2_score(y_test, testy_predict))
    # LR.score(X_test, y_test)

    lr_model = LR
    lr_mse = mean_squared_error(y_test, testy_predict)
    lr_mae = mean_absolute_error(y_test, testy_predict)

    plot = plot_model_learning_curves(X_train, y_train, X_test, y_test, lr_model, 'mean_squared_error')
    plot = plot_model_learning_curves(X_train, trainy_predict, X_test, testy_predict, lr_model, 'mean_squared_error')


    # filename = 'LinearRegression.sav'
    # pickle.dump(lr_model, open(filename, 'wb'))

    return dict(model=lr_model, mse=lr_mse, mae=lr_mae)



def SVRegressor(x,y):
    # gridsearch for choosing the parameter for SVM
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    param_grid = {'C': [10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf']}
    sv_m = SVR()
    svr = GridSearchCV(estimator=sv_m, param_grid=param_grid, refit=True, verbose=3, cv=2)
    svr.fit(X_train, y_train)


    svr_model = svr.best_estimator_

    trainy_predict = svr_model.predict(X_train)
    testy_predict = svr_model.predict(X_test)



    svr_mse = mean_squared_error(y_test, testy_predict)
    svr_mae = mean_absolute_error(y_test, testy_predict)

    plot = plot_model_learning_curves(X_train, y_train, X_test, y_test, svr_model, 'mean_squared_error')
    plot = plot_model_learning_curves(X_train, trainy_predict, X_test, testy_predict, svr_model, 'mean_squared_error')


    # filename = 'SVRegressor.sav'
    # pickle.dump(svr_model, open(filename, 'wb'))

    return dict(model=svr_model, mse=svr_mse, mae=svr_mae)


def random_forest_regressor(x: pd.DataFrame, y: pd.Series) :


    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20,  random_state=42)

    # Train your model using train set.
    rf_regressor = RandomForestRegressor(n_jobs=-1)
    rf_regressor.fit(trainx, trainy)

    # Predict test labels/classes for test set.
    # Predicting the model on both train and test data
    trainy_predict = rf_regressor.predict(trainx)
    testy_predict = rf_regressor.predict(testx)


    # Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    rf_model = rf_regressor
    rf_mse = mean_squared_error(testy, testy_predict)
    rf_mae = mean_absolute_error(testy, testy_predict)

    plot = plot_model_learning_curves(trainx, trainy, testx, testy, rf_model, 'mean_squared_error')
    plot = plot_model_learning_curves(trainx, trainy_predict, testx, testy_predict, rf_model, 'mean_squared_error')

    # print("Random Forest _> Train accuracy score : ", accuracy_score(trainy, trainy_predict) * 100)
    # print("Random Forest _> Test accuracy score : ", accuracy_score(testy, testy_predict) * 100)
    #
    # print("Random Forest _> Train recall score : ", recall_score(trainy, trainy_predict, average='macro') * 100)
    # print("Random Forest _> Test recall score : ", recall_score(testy, testy_predict, average='macro') * 100)
    #
    # print("Random Forest _> Train precision score : ", precision_score(trainy, trainy_predict, average='macro') * 100)
    # print("Random Forest _> Test precision score : ", precision_score(testy, testy_predict, average='macro') * 100)
    #
    # print("Random Forest _> Train f1 score : ", f1_score(trainy, trainy_predict, average='macro') * 100)
    # print("Random Forest _> Test f1 score : ", f1_score(testy, testy_predict, average='macro') * 100)
    #
    # print("Random Forest _> Train confusion_matrix : \n", confusion_matrix(trainy, trainy_predict))
    # print("Random Forest _> Test confusion_matrix : \n", confusion_matrix(testy, testy_predict))

    return dict(model=rf_model, mse=rf_mse, mae=rf_mae)


def rfr_grid_cv(x: pd.DataFrame, y: pd.Series):

    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20,  random_state=42)


    rf_grid = RandomForestRegressor(n_jobs=-1)
    param_grid = {"n_estimators": [100, 200],
                  "max_depth": [7, 12],
                  "min_samples_leaf": [20, 40]}

    rf_cv_grid = GridSearchCV(estimator=rf_grid, param_grid=param_grid)
    rf_cv_grid.fit(trainx, trainy)

    rf_model = rf_cv_grid.best_estimator_

    trainy_predict = rf_model.predict(trainx)
    testy_predict = rf_model.predict(testx)

    rf_mse = mean_squared_error(testy, testy_predict)
    rf_mae = mean_absolute_error(testy, testy_predict)

    # pickle.dump(rf_model, open("Saved_models//random_forest_best", 'wb'))

    return dict(model=rf_model, mse=rf_mse, mae=rf_mae)


def final_random_forest_regressor(x: pd.DataFrame, y: pd.Series) :


    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20,  random_state=42)

    # Train your model using train set.
    rf_regressor = RandomForestRegressor(max_depth=12, min_samples_leaf=20, n_estimators=200,n_jobs=-1)
    rf_regressor.fit(trainx, trainy)

    # Predict test labels/classes for test set.
    # Predicting the model on both train and test data
    trainy_predict = rf_regressor.predict(trainx)
    testy_predict = rf_regressor.predict(testx)


    # Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    rf_model = rf_regressor
    rf_mse = mean_squared_error(testy, testy_predict)
    rf_mae = mean_absolute_error(testy, testy_predict)

    return dict(model=rf_model, mse=rf_mse, mae=rf_mae)