import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
# import some data to play with
import time
import pickle

class SVClassifier:
    def S_V_C(self,X_train,X_test,Y_train,Y_test):
# we create an instance of SVM and fit out data.
        C = 100  # SVM regularization parameter
        train_time=list()
        stat_time=time.time()
        svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)


        pickle.dump(svc, open('SavedData/classification_SVCLinear_model.sav', 'wb'))
        end_time=time.time()
        training_time = end_time - stat_time
        train_time.append(training_time)
        print("trainig time for linear SVC",training_time)
        stat_time = time.time()
        rbf_svc = svm.SVC(kernel='rbf', gamma=0.1, C=C).fit(X_train, Y_train)

        pickle.dump(rbf_svc, open('SavedData/classification_rbf_Model.sav', 'wb'))

        end_time=time.time()
        training_time = end_time - stat_time
        train_time.append(training_time)
        print("trainig time for rbf SVC",training_time)
        stat_time = time.time()
        poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, Y_train)

        pickle.dump(poly_svc, open('SavedData/classification_poly_svc_Model.sav', 'wb'))

        end_time=time.time()
        training_time = end_time - stat_time
        train_time.append(training_time)
        print("trainig time for poly SVC",training_time)
        # create a mesh to plot in


        # title for the plots
        titles = ['SVC with linear kernel',
                  'SVC with RBF kernel',
                  'SVC with polynomial (degree 3) kernel']

        accuracy=list()
        test_time=list()
        for i, clf in enumerate((svc, rbf_svc, poly_svc)):
            print(clf)
            stat_time = time.time()
            predictions = clf.predict(X_test)
            end_time=time.time()
            testing_time=end_time-stat_time
            test_time.append(testing_time)
            print("testing time for "+clf.kernel, end_time - stat_time)
            acc = accuracy_score(Y_test, predictions)
            # if clf.kernel == 'linear':
            #     accuracy=acc
            accuracy.append(acc)
            print("accuracy: ", acc)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
        return accuracy,train_time,test_time