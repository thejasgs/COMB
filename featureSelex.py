'''FeatureSelex - Feature Selection Module'''
# If you using a for research, please cite our research paper.
# If you are using this code for any other project, please contact Thejas Gubbi Sadashiva <tgs001@fiu.edu> or Daniel Jimenez <djime072@fiu.edu>.

# imports
import numpy as np 
import pandas as pd 


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.metrics import accuracy_score


def COMB( X,Y):

    # imports
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.cluster import completeness_score

    score = []
    for i in range(len(X.columns)): # loop number of features
        K = MiniBatchKMeans(n_clusters=len(Y[Y.columns[0]].unique()), random_state=0)
        pred = K.fit_predict(X.iloc[:, [i]].values)
        s = completeness_score(Y[Y.columns[0]].values,pred)
        score.append(s)
    

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
        x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=11)
        X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
        clf = rf(random_state=0, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred)) 
    

    
    # best score calcuation
    index = accuracy.index(max(accuracy))
    X = X2.iloc[:,0:len(X.columns)-index]

def VMMB( X,Y):
    import numpy as np 
    import pandas as pd 
        # cluster and score
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.cluster import v_measure_score as v
    score = []
    for i in range(len(X.columns)): # loop number of features
        K = MiniBatchKMeans(n_clusters=len(Y[Y.columns[0]].unique()), random_state=0)
        pred = K.fit_predict(X.iloc[:, [i]].values)
        s = v(Y[Y.columns[0]].values,pred)
        score.append(s)
    

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
        x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=11)
        X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
        clf = rf(random_state=0, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred)) 
    

    
    # best score calcuation
    index = accuracy.index(max(accuracy))

    X = X2.iloc[:,0:len(X.columns)-index]


def ARMB( X, Y):
    # cluster and score
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.cluster import adjusted_rand_score

    score = []
    for i in range(len(X.columns)): # loop number of features
        K = MiniBatchKMeans(n_clusters=len(Y[Y.columns[0]].unique()), random_state=0)
        pred = K.fit_predict(X.iloc[:, [i]].values)
        s = adjusted_rand_score(Y[Y.columns[0]].values,pred)
        score.append(s)
    

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
        x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=11)
        X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
        clf = rf(random_state=0, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred)) 
    

    
    # best score calcuation
    index = accuracy.index(max(accuracy))

    X = X2.iloc[:,0:len(X.columns)-index]


def FIMB(X,Y):
    import numpy as np 
    import pandas as pd 
        # cluster and score
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics.cluster import fowlkes_mallows_score as scorer
    score = []
    for i in range(len(X.columns)): # loop number of features
        K = MiniBatchKMeans(n_clusters=len(Y[Y.columns[0]].unique()), random_state=0)
        pred = K.fit_predict(X.iloc[:, [i]].values)
        s = scorer(Y[Y.columns[0]].values,pred)
        score.append(s)
    

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
        x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=11)
        X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
        clf = rf(random_state=0, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred)) 
    

    
    # best score calcuation
    index = accuracy.index(max(accuracy))

    X = X2.iloc[:,0:len(X.columns)-index]

def BF(X,Y):
    import numpy as np 
    import pandas as pd 
        # cluster and score
    from sklearn.cluster import Birch
    from sklearn.metrics.cluster import fowlkes_mallows_score as scorer
    score = []
    for i in range(len(X.columns)): # loop number of features
        K = Birch(n_clusters=len(Y[Y.columns[0]].unique()))
        pred = K.fit_predict(X.iloc[:, [i]].values)
        s = scorer(Y[Y.columns[0]].values,pred)
        score.append(s)
    

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
        x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state=11)
        X1.drop(X1.columns[len(X1.columns)-1], axis=1, inplace=True)
        clf = rf(random_state=0, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy.append(accuracy_score(y_test, y_pred)) 
    

    
    # best score calcuation
    index = accuracy.index(max(accuracy))

    X = X2.iloc[:,0:len(X.columns)-index]