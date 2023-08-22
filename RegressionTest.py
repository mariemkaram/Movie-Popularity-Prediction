import pandas as pd
import pickle
from preprocessing import *
from RandomForestModel import *
from RidgeRegressionModel import *
from multipleregression import *
mydata = pd.read_csv('movies-reg-test day 2.csv')


def preprocessingg(mydata):

    mydata.info()
    # X = mydata.iloc[:, 0:-1]  # Features
    # Y = mydata['Rate']  # Label
    X_test = mydata.iloc[:, 0:-1]
    Y_test = mydata['vote_average']

    X_test=preprocessing().handle_hompage(X_test)
    X_test=preprocessing().removedupli(X_test)
    X_test =preprocessing().handle_date(X_test)


    dol = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages']
    selected_list = ['name', 'name', 'name', 'name', 'iso_639_1']

    for i in range(len(dol)):
        X_test = preprocessing().Loctolist(X_test, dol[i], selected_list[i])
        if i == 0:
            ls=pickle.load(open('SavedData/ListOfDicgeners.sav', 'rb'))
        if i == 1:
            ls=pickle.load(open('SavedData/ListOfDickeywords.sav', 'rb'))
        if i == 2:
            ls=pickle.load(open('SavedData/ListOfDicproductioComp.sav', 'rb'))
        if i == 3:
            ls=pickle.load(open('SavedData/ListOfDicproductionCountries.sav', 'rb'))
        if i == 4:
            ls = pickle.load(open('SavedData/ListOfDicSpoken.sav', 'rb'))
        print(ls)
        X_test =preprocessing(). transform_List(X_test, dol[i], ls)


    col = pickle.load(open('SavedData/drop_coloums.sav', 'rb'))
    X_test=preprocessing().drop_columns_test(X_test,col)

    missingValues=pickle.load(open('SavedData/handleNull.sav', 'rb'))
    X_test=preprocessing().fill_null(X_test,missingValues)

    X_test=preprocessing().convert_transform(X_test)

    X_test=preprocessing().one_hot_encode(X_test)
    # print(X_test['Action'])
    featureScale = pickle.load(open('SavedData/featureScaling.sav', 'rb'))
    # print('scallllllle :  ', featureScaleee)
    X_test = pd.DataFrame(featureScale.transform(X_test), columns=X_test.columns, index=X_test.index)
    # X_test=preprocessing().transform_scaling(X_test,featureScale)

    featureSelection=pickle.load(open('SavedData/featureselection.sav','rb'))
    X_test=preprocessing().feature_selection_transform(X_test,featureSelection)




    # Random=pickle.load(open('SavedData/regression_randomForest_model.sav', 'rb'))


    RandomForestModel=pickle.load(open('SavedData/regression_RandomForest_model.sav', 'rb'))
    Random=RandomForestModel.predict(X_test)
    print("Random" ,Random)
    print("Mean square error of test of RandomForest :", metrics.mean_squared_error(np.asarray(Y_test), Random))
    print("Model Accuracy(%) of Random: \t" + str(r2_score(Y_test, Random) * 100) + "%")


    Ridge = pickle.load(open('SavedData/regression_ridge_model.sav', 'rb'))
    RidgePrediction=Ridge.predict(X_test)
    print("Mean square error of test Ridge :", metrics.mean_squared_error(np.asarray(Y_test), RidgePrediction))
    print("Model Accuracy(%) of Ridge: \t" + str(r2_score(Y_test, RidgePrediction) * 100) + "%")

    multipleRegression = pickle.load(open('SavedData/regression_multipleRegression_model.sav', 'rb'))
    multipleRegressionPrediction=multipleRegression.predict(X_test)
    print("Mean square error of test multipleRegression :", metrics.mean_squared_error(np.asarray(Y_test), multipleRegressionPrediction))
    print("Model Accuracy(%) of multipleRegression : \t" + str(r2_score(Y_test,multipleRegressionPrediction ) * 100) + "%")


    svr = pickle.load(open('SavedData/svrModel.sav', 'rb'))
    svr_Prediction=svr.predict(X_test)
    print("Mean square error of test SupportVectorRegression :",metrics.mean_squared_error(np.asarray(Y_test), svr_Prediction))
    print("Model Accuracy(%) of SupportVectorRegression : \t" + str( r2_score(Y_test, svr_Prediction) * 100) + "%")


preprocessingg(mydata)



# def handle_hompage(self,X):
#     print("precentage of null in homepage :"+str((X.homepage.isnull().sum().sum()/len(X.homepage))*100))
#     X['is_homepage'] = np.where(X['homepage'].isnull(), 0, 1)
#     # X = X.drop(['homepage'], axis=1)
#     return X
#
# def removedupli(self,X):
#     if(X.duplicated().sum()>0):
#         X=X.drop_duplicates()
#     return X