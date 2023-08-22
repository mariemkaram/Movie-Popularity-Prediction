from sklearn.ensemble import RandomForestRegressor
import pickle
import numpy as np
from sklearn import metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

class randomforest:
    def randomf(self,X_train,X_test,Y_train,Y_test):
        model1=RandomForestRegressor()
        model1.fit(X_train,Y_train)
        # pickle.dump(model1, open('SavedData/regression_randomForest_model.sav', 'wb'))
        pickle.dump(model1, open('SavedData/regression_RandomForest_model.sav', 'wb'))
        #p=open('randomforest.picle','wb')
        #pickle.dump(model1,p)
        prediction=model1.predict(X_test)
        prediction_train=model1.predict(X_train)

        print("Random forest model")
        print("Mean square error of train:", metrics.mean_squared_error(np.asarray(Y_train), prediction_train))
        print("Mean square error of test :", metrics.mean_squared_error(np.asarray(Y_test), prediction))
        print("Model Accuracy(%): \t" + str(r2_score(Y_test, prediction) * 100) + "%")
        true_player_value = np.asarray(Y_test)[0]
        predicted_player_value = prediction[0]
        print('True value for the first movie in the test set is : ' + str(true_player_value))
        print('Predicted value for the first movie in the test set  is : ' + str(predicted_player_value))
        plt.scatter(Y_test, prediction)
        plt.xlabel('Actual')
        plt.ylabel('Predicted RandomForest')
        sns.regplot(x=Y_test, y=prediction, ci=None, color='red')
        plt.show()
