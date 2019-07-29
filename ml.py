import numpy as np
import pandas as pd
import sklearn
from sklearn.utils import shuffle
from sklearn import linear_model

data = pd.read_csv("cmc.data")

data = data[["age", "f_education", "m_education", "children", "religion", "working", "occupation", "soli", "media",
             "contraception"]]

#  variable
predict = "contraception"
#  arrays for attributes and labels
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

#  split / train / testsize
x_train, y_train, x_test, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

print(data.head())

