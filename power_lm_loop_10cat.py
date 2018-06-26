#########StrongFilter###############################################################
#This script applies a strong filtering algorithm to the raw QRad/QMobo log data
#These filters are based on the following observations:
#	- There were many rows where values such as cpu_temperature or cpu_frequency (measured in the mobo)
#	  were zero, while other values such as power_instant_main and heatink_temperature (measured in the rad)
#	  were nonzero. This is probably due to a measurement problem in the sensors. Every row where some value
#	  was zero is removed.
#
#	- It seems that when the rad is put on standby mode (power less than around 12 watts) the data measured 
#	  in the mobo (notably cpu_frequency and cpu_temperature) is not updated. It is instead assigned to the last value 
#	  measured when the rad was on. This lead to many lines where the cpu_temperature was very high for a long time
#	  with a very low power, this phisically makes very little sense. It is then removed the lines where the power_instant_main 
#	  is less than 12 watts

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

cly = 'power_instant_main'
clx = 'cpu_frequency'
clu = 'cpu_usage'

STANDBY_THRESHOLD = 12.0

for 

#df = pd.read_csv('/media/sf_shared_VB/cat-MOBO-0400-9ea3-a5da-94de80a6fc53.csv', usecols = [clx, cly, clu],sep = '\t')

df = pd.read_csv('/media/sf_shared_VB/cleaned_test_data_2.csv', usecols = [clx, cly, clu],sep = '\t')
print(df[:3])

strongFiltered_df = df.loc[df['power_instant_main']>STANDBY_THRESHOLD]
strongFiltered_df = strongFiltered_df.loc[(strongFiltered_df != 0).all(1)]

#df = strongFiltered_df
print(strongFiltered_df[:3])

df[clx+'_2'] = df[clx]**2
df[clx+'_3'] = df[clx]**3
df['ones'] = 1

df = df.loc[df[clu]==100]

#poly = PolynomialFeatures(4)
#X = poly.fit_transform(df[[clx,clu]].as_matrix())

X = df[['ones', clx, clx+'_2', clx+'_3']]

#clx = clx+'_3'
#X = df[[clx]]

Y = df[cly]

lm = Lasso()
cv = ShuffleSplit(n_splits=3, test_size=0.33)
scores = cross_val_score(lm, X, Y, cv=cv)
#scores = cross_val_score(lm, X, Y, cv=2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state = 5)
X_train, Y_train = shuffle(X_train, Y_train)



print("CV_score: %0.2f(+/_ %0.2f)" % (scores.mean(), scores.std() * 2))


lm.fit(X_train, Y_train)
print('lm_coeffients:', lm.coef_)
print('lm train score: ', lm.score(X_train, Y_train))
print('lm test score: ', lm.score(X_test, Y_test))

import matplotlib.pyplot as plt

plt.figure(figsize=(15,15))
plt.subplot(221)
plt.scatter(Y_test, lm.predict(X_test))
plt.xlabel(clx)
plt.ylabel(cly+":$/hat{T}_i$")
plt.title('LM TEST DATA: '+clx+ " vs "+ cly)

plt.subplot(222)
plt.scatter(X_test[clx], lm.predict(X_test), s=2.3, c='r')
plt.scatter(X_test[clx], Y_test, s=0.6, c='b')
plt.ylabel(clx)
plt.xlabel(cly)
plt.title('LM TEST DATA: '+clx+' vs '+cly)

plt.subplot(223)
plt.scatter(Y_train, lm.predict(X_train))
plt.xlabel(clx)
plt.ylabel(cly)
plt.title('LM TRAIN DATA: '+clx+' vs '+cly)

plt.subplot(224)
plt.scatter(X_train[clx], lm.predict(X_train), s=2.3, c='r')
plt.scatter(X_train[clx], Y_train, s=0.6, c='b')
plt.ylabel(cly)
plt.xlabel(clx)
plt.title('LM TRAIN DATA: '+clx+' vs '+cly)


plt.show()
