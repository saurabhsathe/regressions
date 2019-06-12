# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:38:13 2018

@author: therock
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as mp


dataset=pd.read_csv('Position_Salaries.csv')

#lets create our dependent and independent vectors
x=dataset.iloc[:,[1]]
y=dataset.iloc[:,[2]]

from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
scy=StandardScaler()
x=scx.fit_transform(x)
y=scy.fit_transform(y)

from sklearn.svm import SVR
regres=SVR(kernel='rbf')
regres.fit(x,y)

ypred=scy.inverse_transform(regres.predict(scx.transform(np.array([[6.5]]))))

