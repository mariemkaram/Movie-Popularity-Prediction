import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

import pickle

# preprocessing(X_train,X_test,Y_train,Y_test)
class SVRegressor:
    def S_V_R(self,X_train,X_test,Y_train,Y_test):
        svr_regressor = SVR(kernel = 'rbf')
        svr_regressor.fit(X_train, Y_train)
        pickle.dump(svr_regressor, open('SavedData/svrModel.sav', 'wb'))
        # p=open('ridgemodel.picle','wb')
        # pickle.dump(model2,p)
        prediction=svr_regressor.predict(X_test)
        prediction_train=svr_regressor.predict(X_train)
        print("Support Vector regression model")
        print("Mean square error of train:",metrics.mean_squared_error(np.asarray(Y_train),prediction_train))
        print("Mean square error of test :",metrics.mean_squared_error(np.asarray(Y_test),prediction))
        print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction) * 100) + "%")
        true_player_value = np.asarray(Y_test)[0]
        predicted_player_value = prediction[0]
        print('True value for the first movie in the test set is : ' + str(true_player_value))
        print('Predicted value for the first movie in the test set  is : ' + str(predicted_player_value))
        print(X_train)
        plt.scatter(Y_test, prediction)
        plt.xlabel('Actual')
        plt.ylabel('PredictedSVR')
        sns.regplot(x=Y_test, y=prediction, ci=None, color='red')
        plt.show()

