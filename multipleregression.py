import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class multilinear:
    def multi(self,X_train,X_test,Y_train,Y_test):
        cls = linear_model.LinearRegression()
        cls.fit(X_train,Y_train)
        pickle.dump(cls, open('SavedData/regression_multipleRegression_model.sav', 'wb'))
        y_train_predicted=cls.predict(X_train)
        prediction= cls.predict(X_test)
        print("MultiLinear model")
        #print('Co-efficient of linear regression',cls.coef_)
        print('Intercept of linear regression model',cls.intercept_)
        print('Mean Square Error of train', metrics.mean_squared_error(Y_train,y_train_predicted))
        print('Mean Square Error of test', metrics.mean_squared_error(np.asarray(Y_test), prediction))
        print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction) * 100) + "%")
        true_player_value = np.asarray(Y_test)[0]
        predicted_player_value = prediction[0]
        print('True value for the first movie in the test set is : ' + str(true_player_value))
        print('Predicted value for the first movie in the test set  is : ' + str(predicted_player_value))
        plt.scatter(Y_test, prediction)
        plt.xlabel('Actual')
        plt.ylabel('Predicted Multiple')
        sns.regplot(x=Y_test, y=prediction, ci=None, color='red')
        plt.show()

