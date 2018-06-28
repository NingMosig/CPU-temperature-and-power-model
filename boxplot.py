import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

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
        ax = df.boxplot(by = clx, column = cly)
        plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
        filename = filename[:23]
        plt.savefig('boxplot-'+filename+'.jpg', bbox_inches = 'tight')
