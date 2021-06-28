import csv
import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.svm import SVR
from scipy.stats import pearsonr
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix

count = 40

for x in range(6):
    # a legjobb c paraméter beállítása, amelyel az SVR számol
    c = 0.001
    if x >1 and x<4:
        c = 0.0001

    print("start",count)
    #változók deklarálása és inicializálása
    scaler_train = StandardScaler()

    features_train = []
    features_dev = []
    features_test = []
    X_features = []
    y_features = []
    round_dev = []
    round_test = []

# beolvasásra és kiíratásra tömbök
    trains = ['features/train-40.csv' , 'features/train-50.csv' , 'features/train-60.csv' , 'features/train-70.csv' , 'features/train-80.csv' , 'features/train-90.csv']
    developments = ['features/devel-40.csv','features/devel-50.csv','features/devel-60.csv','features/devel-70.csv' , 'features/devel-80.csv' , 'features/devel-90.csv']
    tests = ['features/test-40.csv','features/test-50.csv','features/test-60.csv','features/test-70.csv' , 'features/test-80.csv' , 'features/test-90.csv']
    results = ['print/result-40.csv','print/result-50.csv','print/result-60.csv','print/result-70.csv' , 'print/result-80.csv' , 'print/result-90.csv']
    results_dev = ['print/result_dev-40.csv','print/result_dev-50.csv','print/result_dev-60.csv','print/result_dev-70.csv' , 'print/result_dev-80.csv' , 'print/result_dev-90.csv']
    results_test = ['print/result_test-40.csv','print/result_test-50.csv','print/result_test-60.csv','print/result_test-70.csv' , 'print/result_test-80.csv' , 'print/result_test-90.csv']


# CSV beolvasás -> jellemzővektorok

    with open(trains[x], encoding='utf-8-sig') as csvfile_train:
        spamreader = csv.reader(csvfile_train, delimiter=';', quotechar='|')
        for row in spamreader:
            features_train.append(row)


    with open(developments[x], encoding='utf-8-sig') as csvfile_dev:
        spamreader = csv.reader(csvfile_dev, delimiter=';', quotechar='|')
        for row in spamreader:
            features_dev.append(row)


    with open(tests[x], encoding='utf-8-sig') as csvfile_test:
        spamreader = csv.reader(csvfile_test, delimiter=';', quotechar='|')
        for row in spamreader:
            features_test.append(row)

# CSV beolvasása -> célértékek
    dataset_train = pd.read_csv('label/label_train.csv')
    y_train = dataset_train.iloc[:, 0].values

    dataset_dev = pd.read_csv('label/label_dev.csv')
    y_dev = dataset_dev.iloc[:, 0].values

    dataset_test = pd.read_csv('label/label_test.csv')
    y_test = dataset_test.iloc[:, 0].values

# training

    SF_train = scaler_train.fit(features_train)
    X_train = SF_train.transform(features_train)

    X_dev = features_dev
    X_test = features_test

    # #forciklusba melyika legjobb C érték, ezekkel az értékekkel számolunk
    # complexes = [10 , 1 , 0.1 , 0.01 , 0.001 , 0.0001 , 0.00001]
    # for c in complexes:
    #     svr_lin = SVR(kernel='linear', C=c, gamma='auto')
    #     svr_lin.fit(X_train, y_train)
    #     pred_dev = svr_lin.predict(X_dev)
    #     corr = pearsonr(pred_dev, y_dev)
    #     print("C:",c," korreláció: ",corr)


    svr_lin = SVR(kernel='linear', C=c, gamma='auto')
    svr_lin.fit(X_train, y_train)
    pred_dev = svr_lin.predict(X_dev)

# átalakítás egész számokká a dev tömböt
    fact = np.std(y_dev) / np.std(pred_dev)
    preds_dev_res = ((pred_dev - statistics.mean(pred_dev)) * fact + statistics.mean(y_dev));
    r = np.round(preds_dev_res)
    for i in r:
        if i > 9:
            i = 9.0
            round_dev.append(i)
        elif i < 1:
            i = 1.0
            round_dev.append(i)
        else:
            round_dev.append(i)
# korreláció a dev-re
    #print("dev egészek:", round_dev)
    pearson_corr = pearsonr(round_dev, y_dev)[0]
    spear_corr=stats.spearmanr(round_dev, y_dev)[0]
    print("dev C:",c,"pearson korreláció: ",pearson_corr)
    print("dev C:",c,"spearman korreláció: ",spear_corr)
# train és dev összefűzése
    for i in X_train:
        X_features.append(i)
    for j in X_dev:
        X_features.append(j)

    for ii in y_train:
        y_features.append(ii)
    for jj in y_dev:
        y_features.append(jj)

    svr_lin.fit(X_features, y_features)
    pred_test = svr_lin.predict(X_test)

# átalakítás egész számokká a test tömböt
    fact = np.std(y_test) / np.std(pred_test)
    preds_test_res = ((pred_test - statistics.mean(pred_test)) * fact + statistics.mean(y_test));
    r = np.round(preds_test_res)
    for i in r:
        if i > 9:
            i = 9.0
            round_test.append(int(i))
        elif i < 1:
            i = 1.0
            round_test.append(int(i))
        else:
            round_test.append(int(i))
# korreláció a test-re
    #print("test egészek:",round_test)
   # print(round_test)
   # print(y_test)
    pearson_corr2 = pearsonr(round_test, y_test)[0]
    spear_corr2 = stats.spearmanr(round_test, y_test)[0]
    print("test C:",c, "pearson korreláció: ",pearson_corr2)
    print("test C:", c, "spearman korreláció: ", spear_corr2)

# count változó növelése
    count=count+10
    
# tévesztési mátrix
    mat = confusion_matrix(y_test,round_test)
    class_name = ['1','2','3','4','5','6','7','8','9']
    norm='develop'
    plot_confusion_matrix(conf_mat=mat,figsize=(9,9),class_names=class_name)
    plt.show()

    # eredmények kiírása CSV-be
    with open(results[x], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["pearson korrelacio dev:",pearson_corr])
        writer.writerow(["spearman korrelacio dev:",spear_corr])
        writer.writerow(["pearson korrelacio test:",pearson_corr2])
        writer.writerow(["spearman korrelacio test:",spear_corr2])

    with open(results_dev[x], 'w', newline='') as file:
        writer = csv.writer(file)
        for i in round_dev:
            writer.writerow([int(i)])

    with open(results_test[x], 'w', newline='') as file:
        writer = csv.writer(file)
        for i in round_test:
            writer.writerow([int(i)])

