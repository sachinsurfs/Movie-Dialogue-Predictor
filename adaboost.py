import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_hastie_10_2
import matplotlib.pyplot as plt
import math
import csv
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import pairwise
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y):
    y_len = float(len(Y))
    err  = 0.0
    for i in range(len(Y)):
        err += int(int(pred[i]) != int(Y[i]))

    return  100*(err/y_len)   
    #return sum(int(pred != Y)) / y_len

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err):
    print 'Error rate: Training: %.4f - Test: %.4f' % err

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

def calculateScore(copy, pred, Y) :

    y_len = float(len(Y))
    theta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    sums = [0.0] * 10
    correct = 0
    err  = 0.0
    for i in range(len(Y)):
        if(int(int(pred[i]) == int(Y[i]))) :
            correct += 1
            for j in range(10):
                if(abs(copy[i]) < theta[j]):
                    sums[j] += 1
    for j in range(10):
        sums[j] = sums[j]/correct

    print sums
    
def adaboost_clf(Y_train, X_train, Y_test, X_test, M, clf, rho=0, theta=0.1):
    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]
    
    #Create Model
    for i in range(M):
        # Train base classifier with W
        clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = clf.predict(X_train)
        pred_test_i = clf.predict(X_test)

        #h(x) expressed in terms of {1,-1}
        pred_train_i1 = [1 if x=='1' else -1 for x in pred_train_i]
        pred_test_i1 = [1 if x=='1' else -1 for x in pred_test_i]

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
         
        #h(x)y in terms of {1,-1}
        miss2 = [x if x==1 else -1 for x in miss]
        

        # Error
        err_m = np.dot(w,miss) / sum(w)
        #err_m = float(sum(miss)) / n_train

        err_thr = (1.0-rho)/2.0
        print err_m

        if(err_m > err_thr) :
            continue

        #print err_m
        # Alpha
        alpha_m = 0.5 * np.log(((1.0-rho) *(1.0 - err_m) )/ ((1.0+rho)*err_m))

        #print alpha_m
        #Normalization Factor 
        Z_t = 2*( math.sqrt((err_m*(1-err_m)) / (1-rho**2) ) )
        
        #w, np.exp([float(x) * alpha_m for x in miss])

        # Update Distribution        
        w = np.divide(np.multiply(w, np.exp([float(x) * alpha_m for x in miss2])), Z_t)

        #print sum(w)
        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, 
                                          [float(x) * alpha_m for x in pred_train_i1])]
        pred_test = [sum(x) for x in zip(pred_test, 
                                         [float(x) * alpha_m for x in pred_test_i1])]
    


    pred_test_copy = pred_test

    pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)
    
    pred_train = [x if x==1 else 0 for x in pred_train]
    pred_test = [x if x==1 else 0 for x in pred_test]


    #calculateScore(pred_test_copy,pred_test, Y_test)


    # Return error rate in train and test set
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)




""" PLOT FUNCTION ==========================================================="""
def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('Error rate vs number of iterations', fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')

""" MAIN SCRIPT ============================================================="""
if __name__ == '__main__':
    

    reader = csv.reader(open("featureSet.csv", "rb"), delimiter=",")
    data = list(reader)

    m = 1000

    T = {100, 200}
    #T = {500}

    #rho = {0,0.00390625}
    rho = {0}
    #rho = { 0.0009765625,0.001953125,0.00390625,0.0078125, 0.015625,0.03125,0.0625, 0.125, 0.25, 0.5 }
   
    X_train = []
    Y_train = []

    t_cross_error = []
    t_fit_error  = []

    training = data[:m]
    testing = data[m:]

    D = [[]] * m

    for i in range(len(training)):
        X_train.append(list(training[i][:-1]))
        Y_train.append(training[i][-1])

    X_test = []
    Y_test = []

    for i in range(len(testing)):
        X_test.append(list(testing[i][:-1]))
        Y_test.append(testing[i][-1])


    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)   

    
    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = 1, random_state = 1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    
    # Fit Adaboost classifier using a decision tree as base estimator
    # Test with different number of iterations
#    er_train, er_test = [er_tree[0]], [er_tree[1]]

    for j in rho: 
        er_train, er_test = [], []
        for i in T:   
            er_i = adaboost_clf(Y_train, X_train, Y_test, X_test, i, clf_tree,j)
            er_train.append(er_i[0])
            er_test.append(er_i[1])
        print "----"
        print " rho no : "+str(j)
        print er_train
        print er_test