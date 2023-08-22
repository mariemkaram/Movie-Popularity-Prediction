import pandas as pd
import pickle
from preprocessing import *
from RandomForestClassification import *
from SupportVectorClassifier import *
from KNeighborsClassifier import *

mydata = pd.read_csv('movies-tas-test day 2.csv')


def preprocessingg(mydata):

    mydata.info()
    # X = mydata.iloc[:, 0:-1]  # Features
    # Y = mydata['Rate']  # Label
    X_test = mydata.iloc[:, 0:-1]
    Y_test = mydata['Rate']

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

    sfs=pickle.load(open('SavedData/classfeatureselection.sav', 'rb'))
    X_test=preprocessing().wrapper_feature_selection_transform(X_test,sfs)



     #Models
    KNN=pickle.load(open('SavedData/classification_KNN_model.sav', 'rb'))
    kNNPred = KNN.predict(X_test)
    accuracy1 = accuracy_score(Y_test, kNNPred)
    print("Accuracy KNN:", accuracy1)

    RandomForest=pickle.load(open('SavedData/classification_RFClassification_model.sav', 'rb'))
    RandomForestPred=RandomForest.predict(X_test)
    accuracy2 = accuracy_score(Y_test, RandomForestPred)
    print("Accuracy RandomForest:", accuracy2)

    Svc=pickle.load(open('SavedData/classification_SVCLinear_model.sav', 'rb'))
    svcpred=Svc.predict(X_test)
    accuracy3 = accuracy_score(Y_test, svcpred)
    print("Accuracy SVCLinear:", accuracy3)


    rbf=pickle.load(open('SavedData/classification_rbf_Model.sav', 'rb'))
    rbfpred = rbf.predict(X_test)
    accuracy4 = accuracy_score(Y_test, rbfpred)
    print("Accuracy rbf:", accuracy4)


    poly=pickle.load(open('SavedData/classification_poly_svc_Model.sav', 'rb'))
    polypred = poly.predict(X_test)
    accuracy5 = accuracy_score(Y_test,polypred)
    print("Accuracy poly:", accuracy5)





preprocessingg(mydata)