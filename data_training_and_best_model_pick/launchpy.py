from full_pip import *
from full_pip import full_pipeline
import sklearn.metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

url = "https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data/diamonds.csv"
data = pd.read_csv(url, delimiter = ",")
data = data[(data.x * data.y * data.z != 0) & (data.price > 0)]
data_processed = data.drop(columns=['depth', 'table', 'y', 'z'])
data_dummy = pd.get_dummies(data_processed, columns=['cut', 'color', 'clarity'], drop_first=True)
x = data_dummy.drop(columns='price')
y = data_dummy.price

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