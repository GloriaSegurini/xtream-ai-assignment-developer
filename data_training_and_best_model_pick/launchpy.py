from full_pip import *
from full_pip import full_pipeline
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

x = pd.read_csv("https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data_training_and_best_model_pick/x.csv")
y = pd.read_csv("https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data_training_and_best_model_pick/y.csv")
y = y['price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = [LinearRegression(), SVR(), DecisionTreeRegressor(random_state=0) ]
metrics = [r2_score, mean_absolute_error]
scores_file = 'scores_file_pkl'
weights = [1,2]
metric_tomin = ['mean_absolute_error']
best_model_file = 'best_model_file.pkl'

best_model = full_pipeline(models, x_train, y_train, x_test, metrics, scores_file, y_test, weights, metric_tomin, best_model_file)

with open(best_model_file, 'rb') as file:
    temp = pickle.load(file)

print(temp)