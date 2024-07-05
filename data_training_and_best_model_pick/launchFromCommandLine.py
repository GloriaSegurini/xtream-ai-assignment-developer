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

DEFAULT_XURL = "https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data_training_and_best_model_pick/x.csv"
DEFAULT_YURL = "https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data_training_and_best_model_pick/y.csv"
DEFAULT_MODELS = ["LinearRegression()", "SVR()", "DecisionTreeRegressor(random_state=0)"]
DEFAULT_METRICS = ["r2_score", "mean_absolute_error"]
DEFAULT_WEIGHTS = [1,2]
DEFAULT_METRICS_TOMIN = ["mean_absolute_error"]


# Define arguments from command line
parser = argparse.ArgumentParser(description='Run machine learning models')
parser.add_argument('--xurl', type=str, default= DEFAULT_XURL, help='URL of dataset')
parser.add_argument('--yurl', type=str, default= DEFAULT_YURL, help='URL of dataset - target vals')
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

x = pd.read_csv(args.xurl)
y = pd.read_csv(args.yurl)
y = y['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


best_model = full_pipeline(models, x_train, y_train, x_test, metrics, args.scores_file, y_test, args.weights, args.metric_tomin, args.best_model_file)

with open(args.best_model_file, 'rb') as file:
    temp = pickle.load(file)

print(temp)
