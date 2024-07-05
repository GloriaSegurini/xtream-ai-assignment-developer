# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import joblib
import numpy as np



# create a function do automatically drop the desired features and generates the get_dummies dataset
def data_preparation_func(data_url, to_drop, to_get_dummies, tg):
  data = pd.read_csv(data_url, delimiter = ",")

  data = data[(data.x * data.y * data.z != 0) & (data.price > 0)]
  data_processed = data.drop(columns= to_drop)
  data_dummy = pd.get_dummies(data_processed, columns=to_get_dummies, drop_first=True)

  x = data_dummy.drop(columns = tg)
  y = data_dummy[tg]

  x.to_csv('x.csv', index = False)
  y.to_pickle('y')





