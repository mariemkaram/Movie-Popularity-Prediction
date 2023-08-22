from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
class KNClassifier:
    def KNN(self,X_train,X_test,Y_train,Y_test):
        dict={'accuracy':0,'train_time':0,'test_time':0}
        p=[2,3,4]
        for i in p:
            classifier= KNeighborsClassifier(n_neighbors=10, metric='minkowski', p=i )
            start_time=time.time()
            classifier.fit(X_train, Y_train)
            end_time = time.time()
            training_time=end_time-start_time
            print("training time for Kneighbor Classification for p = " + str(i) + " ", training_time)
            start_time=time.time()
            ypred=classifier.predict(X_test)
            end_time = time.time()
            tesing_time=end_time-start_time
            print("testing time for Kneighbor Classification for p = " + str(i) + " ", tesing_time)
            acc = accuracy_score(Y_test, ypred)
            if (acc > dict['accuracy']):
                dict['accuracy'] = acc
                dict['train_time'] = training_time
                dict['test_time'] = tesing_time
                pickle.dump(classifier, open('SavedData/classification_KNN_model.sav', 'wb'))

            print("Accuracy for p = " + str(i) + " ", acc)
        return dict['accuracy'], dict['train_time'], dict['test_time']