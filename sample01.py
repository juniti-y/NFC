# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import seaborn as sns
from ncupy import glm

#
#
#

ndata = 100
xdim = 2
sgm = 0.5
mu_diff = 1.0

x1 = sgm * np.random.randn(ndata, xdim)
y1 = np.ones((ndata,1))
x0 = sgm * np.random.randn(ndata, xdim) + mu_diff
y0 = np.zeros((ndata, 1))

x = np.r_[x0, x1]
y = np.r_[y0, y1]

df = pd.DataFrame(data = np.c_[x,y])
sns.pairplot(df, hue=2)

model = glm.Logit()
model.fit(x, y)
#model.status()