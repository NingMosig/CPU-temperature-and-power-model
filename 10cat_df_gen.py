import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
import os


cly = 'power_instant_main'
clx = 'cpu_frequency'
clu = 'cpu_usage'

STANDBY_THRESHOLD = 12.0

for filename in os.listdir('/media/sf_shared_VB/'):
    if filename.startswith("cat-"):
        filepath = '/media/sf_shared_VB/' + filename
        #print(filepath)
        df = pd.read_csv(filepath, usecols = [clx, cly, clu], sep = '\t')
        #print(df)

        #print('before',df.shape[0])
        strongFiltered_df = df.loc[df['power_instant_main']>STANDBY_THRESHOLD]
        strongFiltered_df = strongFiltered_df.loc[(strongFiltered_df != 0).all(1)]
        df = strongFiltered_df
        #print('sfter',df.shape[0])
        #print(df.cpu_frequency.unique())
        #continue
        #evaluate the trained model for this df
        
        if df.shape[0] == 0:
            print('--------------------------------------------------------------')
            print('filename: ', filename)
            print('This file does not have samples after strong filter')
            print('--------------------------------------------------------------')
            continue
                    

        df[clx+'_2'] = df[clx]**2
        df[clx+'_3'] = df[clx]**3
        df['ones'] = 1

        df = df.loc[df[clu]==100]

        #poly = PolynomialFeatures(4)
        #X = poly.fit_transform(df[[clx,clu]].as_matrix())

        X = df[['ones', clx, clx+'_2', clx+'_3']]
        X = X.as_matrix()
        
        #print(X)
        #quit()

        #clx = clx+'_3'
        #X = df[[clx]]

        Y = df[cly]
        Y = Y.as_matrix()
        #print(Y)
        #continue

        lm = Lasso(max_iter=10000)
        
        lm = LinearRegression()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 5)
        lm.fit(X_train, Y_train)        
        print('--------------------------------------------------------------')
        print('filename: ', filename)
        print('lm_coeffients:', lm.coef_)
        print('lm train score: ', lm.score(X_train, Y_train))
        print('lm test score: ', lm.score(X_test, Y_test))       
        print('--------------------------------------------------------------')
        continue
        
        cv = ShuffleSplit(n_splits=3, test_size=0.33)
        scores_cv = cross_val_score(lm, X, Y, cv=cv)
        scores_3fold = cross_val_score(lm, X, Y, cv=3)

        #print scores for this df
        print('--------------------------------------------------------------')
        print('filename: ', filename)
        print("scores_cv: %0.2f(+/_ %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
        print("scores_3fold: %0.2f(+/_ %0.2f)" % (scores_3fold.mean(), scores_3fold.std() * 2))
        print('--------------------------------------------------------------')
        

