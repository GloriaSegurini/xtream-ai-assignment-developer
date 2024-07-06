# xtream AI Challenge - Software Engineer

## Ready Player 1? 🚀

Hey there! Congrats on crushing our first screening! 🎉 You're off to a fantastic start!

Welcome to the next level of your journey to join the [xtream](https://xtreamers.io) AI squad. Here's your next mission.

You will face 4 challenges. **Don't stress about doing them all**. Just dive into the ones that spark your interest or that you feel confident about. Let your talents shine bright! ✨

This assignment is designed to test your skills in engineering and software development. You **will not need to design or develop models**. Someone has already done that for you. 

You've got **7 days** to show us your magic, starting now. No rush—work at your own pace. If you need more time, just let us know. We're here to help you succeed. 🤝

### Your Mission
[comment]: # (Well, well, well. Nice to see you around! You found an Easter Egg! Put the picture of an iguana at the beginning of the "How to Run" section, just to let us know. And have fun with the challenges! 🦎)

Think of this as a real-world project. Fork this repo and treat it like you're working on something big! When the deadline hits, we'll be excited to check out your work. No need to tell us you're done – we'll know. 😎

**Remember**: At the end of this doc, there's a "How to run" section left blank just for you. Please fill it in with instructions on how to run your code.

### How We'll Evaluate Your Work

We'll be looking at a bunch of things to see how awesome your work is, like:

* Your approach and method
* How you use your tools (like git and Python packages)
* The neatness of your code
* The readability and maintainability of your code
* The clarity of your documentation

🚨 **Heads Up**: You might think the tasks are a bit open-ended or the instructions aren't super detailed. That’s intentional! We want to see how you creatively make the most out of the problem and craft your own effective solutions.

---

### Context

Marta, a data scientist at xtream, has been working on a project for a client. She's been doing a great job, but she's got a lot on her plate. So, she's asked you to help her out with this project.

Marta has given you a notebook with the work she's done so far and a dataset to work with. You can find both in this repository.
You can also find a copy of the notebook on Google Colab [here](https://colab.research.google.com/drive/1ZUg5sAj-nW0k3E5fEcDuDBdQF-IhTQrd?usp=sharing).

The model is good enough; now it's time to build the supporting infrastructure.

### Challenge 1

**Develop an automated pipeline** that trains your model with fresh data, keeping it as sharp as the diamonds it processes. 
Pick the best linear model: do not worry about the xgboost model or hyperparameter tuning. 
Maintain a history of all the models you train and save the performance metrics of each one.

### Challenge 2

Level up! Now you need to support **both models** that Marta has developed: the linear regression and the XGBoost with hyperparameter optimization. 
Be careful. 
In the near future, you may want to include more models, so make sure your pipeline is flexible enough to handle that.

### Challenge 3

Build a **REST API** to integrate your model into a web app, making it a breeze for the team to use. Keep it developer-friendly – not everyone speaks 'data scientist'! 
Your API should support two use cases:
1. Predict the value of a diamond.
2. Given the features of a diamond, return n samples from the training dataset with the same cut, color, and clarity, and the most similar weight.

### Challenge 4

Observability is key. Save every request and response made to the APIs to a **proper database**.

---

## How to run
🦎
## Challenge 1
### Idea behind the code
The purpose of the code is to launch an automated pipeline for training several machine learning models. The user can decide which models to run, which metrics to use for evaluation, and what weights to assign to these metrics. Additionally, the user can also use their preferred data via the --url parameter.

The data cleaning process has also been automated through the `data_preparation_func` function within the `data_preparation.py` file. However, this function is highly specific to the current dataset. Because of this specificity, this step has not been integrated into the automated pipeline responsible for training the models.

The prepared data is stored in two CSV files, namely `x.csv` and `y.csv`, which contain the data that will be split by the automated pipeline (xtream-ai-assignment-developer > data_training_and_best_model_pick)

The pipeline involves the following steps:

1. Models are trained and saved within a dictionary for future use and also in a separate .pkl file under the name {modelname}_trained_model.pkl. Model predictions are made and stored in the same dictionary. In this step, the usable models are also saved, even if they haven't been fitted yet (in case they are elected as the best model). 

2. Next, the required metrics are computed and saved again in the dictionary. The results are plotted. Everything (models, fitted models, parameters, predictions and metrics scores) is now saved into a pkl file provided by the user (parameter scores_file).

3. If negative predictions are observed as for the initial notebook's request, the model responsible for these predictions is re-run. This time, adjustments are made to both y_train and the new predictions. The required metrics are recalculated and the results are plotted again. Of course, in this case as well, all results are saved in .pkl files and within a dictionary.

4. At this point, for improved computational complexity, a single dictionary is created containing all the models that have been run up to this point along with their respective information. However, the predictions are dropped from this dictionary (they still remain within the .pkl files seen in the previous steps).

5. The best model is finally chosen using the pick_best_model function. The scoring calculation proceeds as follows.
Firstly, I've decided to allow the user to assign weights to various metrics, as there might be a situation where one model excels on a certain metric but performs poorly on another, and vice versa for another model. How does one choose in such a case? The idea is to assign a higher weight to the most important metric.
To evaluate the best model, a function is used that normalizes the results of the various metrics, multiplies them by the assigned weight, and calculates the score for each model.
Since some metrics may need to be minimized (such as MAE), these are included as 1 minus the metric in the function calculation above, so that the best model will have the highest score.

For example, suppose we have the following scenario:
|            | r2_score | MAE_score |
|------------|----------|-----------|
| MODEL A    | 0.9      | 700       |
| MODEL B    | 0.6      | 50        |
| weight     | 1        | 2         | 

In a similar scenario, the score calculation is:

For model A:
- r2_score_normalized = (0.9 - 0.6)/(0.9 - 0.6) = 1
- MAE_normalized = 1 - (700 - 50)/(700 - 50) = 0
- score_model_A = weights * metrics_normalized = [1, 2] * [1, 0]

Finally, the model with the best score (highest one) is returned, already fitted and ready to be used.
Note: It is important to emphasize that *model selection should be performed on the validation set and not on the test set!*

### Run code
Here's how to run the code:
In order to launch the code to execute the pipeline, here are the steps to follow:
1. clone the current repository where you prefer on your local device through the following command: git clone https://github.com/GloriaSegurini/xtream-ai-assignment-developer

Now there are different possibilities depending on your needs:
#### If you want to run the code from command line:
2. move into the cloned repository's directory through the following command: cd repository-name
3. move into the following folder: data_training_and_best_model_pick
4. use the following command: python launchFromCommandLine.py to launch the pipeline with default parameters. Otherwise, here is the specification of every parameter:
      - xurl: type: str, content: data source, multiple args allowed: False, NOTE: data must be a cvs format
      - yurl: type: str, content: data source - target, multiple args allowed: False, NOTE: data must be a cvs format
      - imports: type: str, content: imports needed, multiple args allowed: True
      - models: type: str, content: models to launch, multiple args allowed: True
      - metrics: type: str, content: evaluation metrics, multiple args allowed: True
      - scores_file: type: str, content: pkl file where you save your scores, multiple args allowed: False, NOTE: needs .pkl extension
      - weights: type: int, content: metrics' weights, multiple args allowed: True
      - metric_tomin: type: str, content: metrics to be minimized, multiple args allowed: True
      - best_model_file: type: str, content: pkl file where you save the final results, multiple args allowed: False, NOTE: needs .pkl extension

      IMPORTANT: here are some important notes:
      1. please, use double quotes for args of type str.
      2. where multiple args are allowed do not use any comma, just type something like this: --arg_to_add "arg1" "arg2"
      3. The same as above goes for weights parameter: type something like this: --weights 1 2
      4. It is *fundamental* that you pay attentions to the weights' order: it must be the same as the metrics. For example, if I type --metrics "r2_score" "mean_absolute_error", if I type --weights 1 2 it means 1          is the weight for r2_score metric and 2 is the weight for mean_absolute_error metric.
      5. The metrics and models to be used must be typed exactly as they are written in the libraries. Below is an example snippet:
      ```python launchFromCommandLine.py --xurl "https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data_training_and_best_model_pick/x.csv" --yurl "https://raw.githubusercontent.com/GloriaSegurini/xtream-ai-assignment-developer/main/data_training_and_best_model_pick/y.csv" --imports "from sklearn.svm import SVR" "from sklearn.linear_model import LinearRegression" "from sklearn.metrics import r2_score, mean_absolute_error" "from sklearn.tree import DecisionTreeRegressor" --models "LinearRegression()" "SVR" "DecisionTreeRegressor(random_state=0)" --metrics "r2_score" "mean_absolute_error" --scores_file "my_scores.pkl" --weights 1 2 --metric_tomin "mean_absolute_error" --best_model_file "my_best_model.pkl"```

#### If you want to run the code from the .py file byt inside an IDE
2. Navigate to the folder where you cloned the original repository and go to `xtream-ai-assignment-developer > data_training_and_best_model_pick > launchpy`.

#### If you want to run the code from ipynb
2. Navigate to the folder where you cloned the original repository and go to `xtream-ai-assignment-developer > data_training_and_best_model_pick > launch_notebook`

##### Final Remarks
In the file `train_and_pick_bestModel.py`, where the functions used in `full_pip.py` are defined, there is also a function *to_df* that takes a .pkl file as input and allows it to be saved as a pandas DataFrame. This function can be useful for visualizing the final results.

For opening .pkl files it is possible to use the following sintax:
```python
import pickle

with open('file.pkl', 'rb') as file:
    loaded_file = pickle.load(file)
```

Of course, the trained models cannot be converted into a DataFrame using the *to_df* function, but the files containing the scores can be.



