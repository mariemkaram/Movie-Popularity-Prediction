# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import json
#import langid
#from translate import Translator
#from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from preprocessing import preprocessing
from multipleregression import multilinear
from RidgeRegressionModel import ridgeregress
from RandomForestModel import randomforest
from SupportVectorRegressor import SVRegressor
from SupportVectorClassifier import SVClassifier
from KNeighborsClassifier import KNClassifier
from RandomForestClassification import randomforest_calss
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
if __name__ == '__main__':
    mydata = pd.read_csv('movies-regression-dataset.csv')
    mydata.info()
    X = mydata.iloc[:, 0:-1]  # Features
    Y = mydata['vote_average']  # Label
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)
    # pd.set_option("max_rows", None)


    #preprocessing
    pre= preprocessing()
    X_train,X_test=pre.process(X_train,X_test)

    features=pre.feature_selection(X_train,y_train)
    #here features
    X_train=pre.feature_selection_transform(X_train,features)
    X_test=pre.feature_selection_transform(X_test,features)

    #regression models
    multi_model=multilinear()
    multi_model.multi(X_train,X_test,y_train,y_test)
    #print(X_train.shape,X_test.shape)
    ridge_model=ridgeregress()
    ridge_model.ridgereg(X_train,X_test,y_train,y_test)
    randomf_model=randomforest()
    randomf_model.randomf(X_train,X_test,y_train,y_test)
    SVR_model = SVRegressor()
    SVR_model.S_V_R(X_train, X_test, y_train, y_test)

#----------------------------------------------------------------------------------------------------------

    #classification
    mydata = pd.read_csv('movies-classification-dataset.csv')
    mydata.info()
    X = mydata.iloc[:, 0:-1]  # Features
    Y = mydata['Rate']  # Label
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True, random_state=10)
    # pd.set_option("max_rows", None)
    # enc = OrdinalEncoder()
    # scale_mapper = {"Low": 1, "Intermediate": 2, "High": 3}
    # y_train = y_train.replace(scale_mapper)
    # y_test = y_test.replace(scale_mapper)
    # preprocessing
    pre = preprocessing()
    X_train, X_test = pre.process(X_train, X_test)

    sfs = pre.wrapper_feature_selection(X_train, y_train)
    #here sfs
    X_train=pre.wrapper_feature_selection_transform(X_train,sfs)
    X_test=pre.wrapper_feature_selection_transform(X_test,sfs)

    #classificaton models
    SVC_model = SVClassifier()
    acc1,train_time1,test_time1=SVC_model.S_V_C(X_train, X_test, y_train, y_test)
    KNN_model = KNClassifier()
    acc2,train_time2,test_time2=KNN_model.KNN(X_train, X_test, y_train, y_test)
    randomfc_model=randomforest_calss()
    acc3,train_time3,test_time3=randomfc_model.RFClassification(X_train,X_test,y_train,y_test)
    accuracy = {'models': ['SVM_linear','SVM_rbf','SVM_poly', 'KNN', 'Random forest'],
                'accuracy': [acc1[0],acc1[1],acc1[2],acc2,acc3]}
    acc = pd.DataFrame.from_dict(accuracy)
    # df = sns.load_dataset('tips')
    sns.barplot(x='models', y='accuracy', data=acc)
    plt.show()


    training_time = {'models': ['SVM_linear', 'SVM_rbf', 'SVM_poly', 'KNN', 'Random forest'],
                'train_time': [train_time1[0], train_time1[1], train_time1[2], train_time2, train_time3]}
    train = pd.DataFrame.from_dict(training_time)
    # df = sns.load_dataset('tips')
    sns.barplot(x='models', y='train_time', data=train)
    plt.show()

    testing_time = {'models': ['SVM_linear', 'SVM_rbf', 'SVM_poly', 'KNN', 'Random forest'],
                'test_time': [test_time1[0], test_time1[1], test_time1[2], test_time2, test_time3]}
    test = pd.DataFrame.from_dict(testing_time)
    # df = sns.load_dataset('tips')
    sns.barplot(x='models', y='test_time', data=test)
    plt.show()




