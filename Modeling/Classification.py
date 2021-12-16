from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def random_forest_classifier(x: pd.DataFrame, y: pd.Series) :
    """

    """

    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=42)

    # Train your model using train set.
    rf_classifier = RandomForestClassifier(n_jobs=-1)
    rf_classifier.fit(trainx, trainy)

    # Predict test labels/classes for test set.
    # Predicting the model on both train and test data
    trainy_predict = rf_classifier.predict(trainx)
    testy_predict = rf_classifier.predict(testx)

    # Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    rf_model = rf_classifier
    rf_confusion_matrix = confusion_matrix(testy, testy_predict)
    rf_accuracy = accuracy_score(testy, testy_predict) * 100
    rf_precision = precision_score(testy, testy_predict, average='macro') * 100
    rf_recall = recall_score(testy, testy_predict, average='macro') * 100
    rf_f1_score = f1_score(testy, testy_predict, average='macro') * 100

    # printing the error metrics for both train and test data and comparing them as we have to make sure we are generalizing the model not overfitting it
    # We can also plot the learning curves to view the errors
    #
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

    return dict(model=rf_model, confusion_matrix=rf_confusion_matrix, accuracy=rf_accuracy, precision=rf_precision,
                recall=rf_recall, f1_score=rf_f1_score)


def rfc_grid_cv(x: pd.DataFrame, y: pd.Series):
    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=42)

    rf_grid = RandomForestClassifier(n_jobs=-1)
    param_grid = {"n_estimators": [50, 100,200],
                  "max_depth": [7, 10, 20],
                  "min_samples_leaf": [20, 45, 60]}

    rf_cv_grid = GridSearchCV(estimator=rf_grid, param_grid=param_grid, cv=5)
    rf_cv_grid.fit(trainx, trainy)

    rf_model = rf_cv_grid.best_estimator_

    trainy_predict = rf_model.predict(trainx)
    testy_predict = rf_model.predict(testx)

    rf_model = rf_model
    rf_confusion_matrix = confusion_matrix(testy, testy_predict)
    rf_accuracy = accuracy_score(testy, testy_predict) * 100
    rf_precision = precision_score(testy, testy_predict, average='macro') * 100
    rf_recall = recall_score(testy, testy_predict, average='macro') * 100
    rf_f1_score = f1_score(testy, testy_predict, average='macro') * 100


    # pickle.dump(rf_model, open("Models//random_forest_best", 'wb'))

    # loaded_model2 = pickle.load(open("optics_clustering_03", 'rb'))


    return dict(model=rf_model, confusion_matrix=rf_confusion_matrix, accuracy=rf_accuracy, precision=rf_precision,
                recall=rf_recall, f1_score=rf_f1_score)


def final_random_forest_classifier(x: pd.DataFrame, y: pd.Series) :
    """

    """

    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=42)

    # Train your model using train set.
    rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=7, max_leaf_nodes=20, n_jobs=-1)
    rf_classifier.fit(trainx, trainy)

    # Predict test labels/classes for test set.
    # Predicting the model on both train and test data
    trainy_predict = rf_classifier.predict(trainx)
    testy_predict = rf_classifier.predict(testx)

    # Measure the below given performance measures on test predictions.
    # Use methods provided by sklearn to perform train-test split and measure below asked model performance scores.

    rf_model = rf_classifier
    rf_confusion_matrix = confusion_matrix(testy, testy_predict)
    rf_accuracy = accuracy_score(testy, testy_predict) * 100
    rf_precision = precision_score(testy, testy_predict, average='macro') * 100
    rf_recall = recall_score(testy, testy_predict, average='macro') * 100
    rf_f1_score = f1_score(testy, testy_predict, average='macro') * 100

    return dict(model=rf_model, confusion_matrix=rf_confusion_matrix, accuracy=rf_accuracy, precision=rf_precision,
                recall=rf_recall, f1_score=rf_f1_score)

def KNN_classifier(x,y):
    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=42)

    # traing KNN
    knn_model = KNeighborsClassifier(n_neighbors=13).fit(trainx, trainy)
    y_pred = knn_model.predict(testx)
    accuracy_score(testy, y_pred)
    # return knn_model
    return dict(model = knn_model)
def SVM_classifier(x,y):
    # Spliting data using train_test split with test size as 20% and shuffle = True(indicating randomly selecting)
    trainx, testx, trainy, testy = train_test_split(x, y, test_size=0.20, shuffle=True, random_state=42)

    param_grid = {'C': [10, 100, 1000], 'gamma': [1, 0.1, 0.01], 'kernel': ['rbf', 'poly', 'linear']}
    sv_c = SVC()
    SVM_genre = GridSearchCV(estimator=sv_c , param_grid = param_grid, refit=True, verbose=3, cv=2)

    SVM_model = sv_c
    SVM_model.fit(trainx, trainy)

    # svm_model = SVC(C=10, gamma=1, kernel='linear').fit(X_train, y_train)
    y_pred_train = SVM_model.predict(trainx)
    y_pred = SVM_model.predict(testx)
    accuracy_score(testy, y_pred)

    svm_confusion_matrix = confusion_matrix(testy, y_pred)
    svm_accuracy = accuracy_score(testy, y_pred) * 100
    svm_precision = precision_score(testy, y_pred, average='macro') * 100
    svm_recall = recall_score(testy, y_pred, average='macro') * 100
    svm_f1_score = f1_score(testy, y_pred, average='macro') * 100

    return dict(model=SVM_model, confusion_matrix=svm_confusion_matrix, accuracy=svm_accuracy, precision=svm_precision,
                recall=svm_recall, f1_score=svm_f1_score)