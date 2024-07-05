# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import joblib
import numpy as np



# create a function do automatically drop the desired features and generates the get_dummies dataset
def data_preparation(df, to_drop = None, to_get_dummies = None):
  if to_drop == None and to_get_dummies != None: # it means there are some cols to drop, but there's no need to create dummy variables
    df_prepared = pd.get_dummies(df, columns = to_get_dummies, drop_first=True, dtype = int) 

  elif to_drop != None and to_get_dummies == None: #it means there are no cols to drop, but we want to create dummy variables
    df_prepared = df.drop(columns = to_drop)

  elif to_drop != None and to_get_dummies != None: #it means we both want to drop some cols and create dummies
    df_drop = df.drop(columns = to_drop)
    df_prepared = pd.get_dummies(df_drop, columns = to_get_dummies, drop_first=True, dtype = int)

  else: # if both to_drop and to_get_dummies are None, this function is useless
    raise Exception("If both to_drop and to_get_dummies are None, tehe function is useless.")


  return df_prepared






