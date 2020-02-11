'''FeatureSelex - Feature Selection Module'''
# If you using a for research, please cite our research paper.
# If you are using this code for any other project, please contact Thejas Gubbi Sadashiva <tgs001@fiu.edu> or Daniel Jimenez <djime072@fiu.edu>.

# imports
import numpy as np 
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def COMB( X,Y):

    # imports
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import homogeneity_completeness_v_measure as hcv

    score = []
    for i in range(len(X.columns)): # loop number of features
        K = MiniBatchKMeans(n_clusters=len(Y[Y.columns[0]].unique()), random_state=0)
        pred = K.fit_predict(X.iloc[:, [i]].values)
        h,c,v = hcv(Y[Y.columns[0]].values,pred)
        score.append(np.mean([h,c,v]))
    

    # Rank the features and sort

    s2 = score
    np.asarray(s2)

    s1 = []
    for i in range(len(X.columns)):
        s1.append(i)
    
    np.asarray(s1)

    li = list(zip(s1, s2))

    def sortSecond(val): 
        return val[1] 
    
    li.sort(key = sortSecond, reverse=True) 

    # Create a copy of X dataframe with columns sorted by score

    titles = []

    for i in range(len(X.columns)):
        p = X.columns[li[i][0]]
        titles.append(p)


    X1 = pd.DataFrame(columns=titles)

    for i in range(len(X.columns)):
        X1[X1.columns[i]] = X[X.columns[li[i][0]]]
    

    # Recursive Feature Elemination from # of features to 0 and keep the accuracy score of each

    accuracy = []
    X2 = X1.copy()

    # for i in range(len(X1.columns)-1,-1,-1):
    for i in range(len(X1.columns)-1): 
        # remove lowest scored column
        X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
        # begin cross-validation
        cv_accuracy =[]
        from sklearn.model_selection import StratifiedKFold
        kf = StratifiedKFold(n_splits=5)
        for train, test in kf.split(X1,Y):
            # classifyer
            r = rf(random_state=0, n_jobs=-1) 
            # train test split
            x_train = X.iloc[train]
            x_test = X.iloc[test]
            y_train = Y.iloc[train]
            y_test = Y.iloc[test]
            r.fit(x_train, y_train)
            y_pred = r.predict(x_test)
            cv_accuracy.append(accuracy_score(y_test, y_pred)) 
        accuracy.append(np.mean(cv_accuracy))
    
    # best score calcuation
    index = accuracy.index(max(accuracy))
    X = X2.iloc[:,0:len(X.columns)-index]
    
    return X