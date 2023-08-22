from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import pandas as pd
import numpy as np
import time
import pickle
class randomforest_calss:
    def RFClassification(self,X_train,X_test,Y_train,Y_test):
        dict={'accuracy':0,'train_time':0,'test_time':0}
        maxdepth=[5,10,15,20]
        for i in maxdepth:
            rf = RandomForestClassifier(max_depth=i,n_estimators=200,random_state=10)

            start_time=time.time()
            rf.fit(X_train, Y_train)
            end_time=time.time()
            training_time=end_time-start_time
            print("training time for Random Forest Classification for maxdepth = "+str(i)+" ",training_time)
            start_time=time.time()
            y_pred = rf.predict(X_test)
            end_time=time.time()
            testing_time=end_time-start_time
            print("testing time for Random Forest Classification for maxdepth = "+str(i)+" ",testing_time)
            accuracy = accuracy_score(Y_test, y_pred)
            if(accuracy>dict['accuracy']):
                pickle.dump(rf, open('SavedData/classification_RFClassification_model.sav', 'wb'))
                dict['accuracy']=accuracy
                dict['train_time']=training_time
                dict['test_time']=testing_time
            print("Accuracy for maxdepth = "+str(i)+" ", accuracy)
        return dict['accuracy'],dict['train_time'],dict['test_time']