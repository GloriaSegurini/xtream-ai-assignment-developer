import argparse
import pandas as pd
import pickle
import importlib.util
from full_pip import full_pipeline
from sklearn.model_selection import train_test_split

DEFAULT_IMPORTS = [
    "from sklearn.svm import SVR",
    "from sklearn.linear_model import LinearRegression",
    "from sklearn.metrics import r2_score, mean_absolute_error",
    "from sklearn.tree import DecisionTreeRegressor"
]

DEFAULT_MODELS = ["LinearRegression()", "SVR()", "DecisionTreeRegressor(random_state=0)"]
DEFAULT_METRICS = ["r2_score", "mean_absolute_error"]
DEFAULT_WEIGHTS = [1,2]
DEFAULT_METRICS_TOMIN = ["mean_absolute_error"]


# Define arguments from command line
parser = argparse.ArgumentParser(description='Run machine learning models')
parser.add_argument('--url', type=str, default="https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data/diamonds.csv", help='URL of the dataset')
parser.add_argument('--imports', type=str, default = DEFAULT_IMPORTS, nargs='+', help='List of import statements')



parser.add_argument('--models', nargs='+', default = DEFAULT_MODELS, help='List of models to use (as a string to be evaluated)')
parser.add_argument('--metrics', type=str, nargs='+', default = DEFAULT_METRICS, help='List of metrics to evaluate (as a string to be evaluated)')
parser.add_argument('--scores_file', type=str, default='scores_file_pkl', help='File to save the scores')
parser.add_argument('--weights', type=int, nargs='+', default= DEFAULT_WEIGHTS, help='List of weights for the metrics')
parser.add_argument('--metric_tomin', type=str, nargs='+', default= DEFAULT_METRICS_TOMIN, help='List of metrics to minimize')
parser.add_argument('--best_model_file', type=str, default='best_model_file.pkl', help='File to save the best model')

args = parser.parse_args()

# Import user's modules
for import_statement in args.imports:
    exec(import_statement)

models = []
for model_str in args.models:
    model = eval(model_str)
    models.append(model)

metrics = []
for metric_str in args.metrics:
    metric = eval(metric_str)
    metrics.append(metric)

metric_tomin = []
for metric in args.metric_tomin:
    metric_tomin.append(metric)

weights = []
for weight in args.weights:
    weights.append(weight)

# Carica e preprocessa il dataset
data = pd.read_csv(args.url, delimiter=",")
data = data[(data.x * data.y * data.z != 0) & (data.price > 0)]
data_processed = data.drop(columns=['depth', 'table', 'y', 'z'])
data_dummy = pd.get_dummies(data_processed, columns=['cut', 'color', 'clarity'], drop_first=True)
x = data_dummy.drop(columns='price')
y = data_dummy.price

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


best_model = full_pipeline(models, x_train, y_train, x_test, metrics, args.scores_file, y_test, args.weights, args.metric_tomin, args.best_model_file)

with open(args.best_model_file, 'rb') as file:
    temp = pickle.load(file)

print(temp)
