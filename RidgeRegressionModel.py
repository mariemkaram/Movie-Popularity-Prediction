import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# preprocessing(X_train,X_test,Y_train,Y_test)
class ridgeregress:
    def ridgereg(self,X_train,X_test,Y_train,Y_test):
        model2=RidgeCV()
        model2.fit(X_train,Y_train)
        pickle.dump(model2, open('SavedData/regression_ridge_model.sav', 'wb'))
        # p=open('ridgemodel.picle','wb')
        # pickle.dump(model2,p)
        prediction=model2.predict(X_test)
        prediction_train=model2.predict(X_train)
        print("Ridge regression model")
        print("Mean square error of train:",metrics.mean_squared_error(np.asarray(Y_train),prediction_train))
        print("Mean square error of test :",metrics.mean_squared_error(np.asarray(Y_test),prediction))
        print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction) * 100) + "%")
        true_player_value = np.asarray(Y_test)[0]
        predicted_player_value = prediction[0]
        print('True value for the first movie in the test set is : ' + str(true_player_value))
        print('Predicted value for the first movie in the test set  is : ' + str(predicted_player_value))
        plt.scatter(Y_test, prediction)
        plt.xlabel('Actual')
        plt.ylabel('Predicted Ridge')
        sns.regplot(x=Y_test, y=prediction, ci=None, color='red')
        plt.show()

