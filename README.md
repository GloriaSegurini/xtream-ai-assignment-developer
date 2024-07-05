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
In order to launch the code to execute the pipeline, here are the steps to follow:
1. clone the current repository where you prefer on your local device through the following command: git clone https://github.com/GloriaSegurini/xtream-ai-assignment-developer
Now there are different possibilities depending on your needs:
- If you want to run the code from command line:
    2. move into the cloned repository's directory through the following command: cd repository-name
    3. move into the following folder: data_training_and_best_model_pick
    4. use the following command: python launchFromCommandLine.py to launch the pipeline with default parameters. Otherwise, here is the specification of every parameter:
      --> url: type: str, content: data source, multiple args allowed: False, NOTE: data must be a cvs format
      --> imports: type: str, content: imports needed, multiple args allowed: True
      --> models: type: str, content: models to launch, multiple args allowed: True
      --> metrics: type: str, content: evaluation metrics, multiple args allowed: True
      --> scores_file: type: str, content: pkl file where you save your scores, multiple args allowed: False, NOTE: needs .pkl extension
      --> weights: type: int, content: metrics' weights, multiple args allowed: True
      --> metric_tomin: type: str, content: metrics to be minimized, multiple args allowed: True
      --> best_model_file: type: str, content: pkl file where you save the final results, multiple args allowed: False, NOTE: needs .pkl extension

      IMPORTANT: here are some important notes:
      1. please, use double quotes for args of type str.
      2. where multiple args are allowed do not use any comma, just type something like this: --arg_to_add "arg1" "arg2"
      3. The same as above goes for weights parameter: type something like this: --weights 1 2
      4. It is *fundamental* that you pay attentions to the weights' order: it must be the same as the metrics. For example, if I type --metrics "r2_score" "mean_absolute_error", if I type --weights 1 2 it means 1          is the weight for r2_score metric and 2 is the weight for mean_absolute_error metric.
