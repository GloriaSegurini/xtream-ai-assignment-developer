# Imports
import pandas as pd
 #from sklearn.model_selection import train_test_split
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt




#create a function to automatize splitting
 #def splitting(df, test_size):
   #x = df.drop(columns='price') #drop tg variable
   #y = df.price

 # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

 # return x_train, x_test, y_train, y_test



#this function is used to plot the model's predictions against its the true labels, as shown in the colab notebook.
def plot_gof(y_true: pd.Series, dictionary):
  for model in dictionary['model']:
    y_pred = dictionary['model'][model]['preds'][0]
    plt.title("Plot for model "+ model)
    plt.plot(y_true, y_pred, '.')
    plt.plot(y_true, y_true, linewidth=3, c='black')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()





def models_training(models, x_train, y_train, x_test, preds_toexp): #the objective of the function is to both run the model's training n times keeping track of the trained models and save the performance metrics of each one
  #Let's first create the necessary structures to save models' names, models' parameters, iterations, scores
  results = {'model' : {}}

  for mod in models: 
    model_name = type(mod).__name__
    results['model'][model_name] = {'params': mod.get_params(), 'metrics': {}, 'fit_model_save': [],  'preds': [], 'score':[], 'model_ready_before_fit': mod}
  
    # train the model and make predicitons. Let's also save the fit model for later
    print("Training of model " + model_name + " **********************")
    mod.fit(x_train, y_train)

    model_file = f'{model_name}_trained_model.pkl'

    with open(model_file, 'wb') as file:
       pickle.dump(mod, file)
       results['model'][model_name]['fit_model_save'].append(model_file)
    
    
    print("Making predictions of model " + model_name + " **********************")
    pred = mod.predict(x_test)

    if preds_toexp == True: #if it is necessary to transform the preds
      pred = np.exp(pred)

    results['model'][model_name]['preds'].append(pred)
  

  return results



def models_evaluation(models, metrics, scores_file, results, y_test):
  for mod in models:
    model_name = type(mod).__name__

    print("Evaluating predictions of model "+ model_name +  " **********************")

    pred = results['model'][model_name]['preds'][0]

    #let's compute model's metrics and save them
    for metric in metrics:
      score = round(metric(y_test, pred), 4)
      print(f'{metric.__name__}: {score}')
      results['model'][model_name]['metrics'][metric.__name__] = score # create structure to save model's scores

    with open(scores_file, 'wb') as file:
      pickle.dump(results, file)



def to_df(scores_file): #this function tourns the results into a dataframe for a better visualization
    try:
        with open(scores_file, 'rb') as file:
            results = pickle.load(file)
    except FileNotFoundError:
        print(f"File '{scores_file}' not found.")
        return None
    
    if 'model' not in results or len(results['model']) == 0:
        print(f"No models found in '{scores_file}'.")
        return None
    
    data = []
    
    for model_name, model_data in results['model'].items():
        params = model_data['params']
        metrics = model_data['metrics']
        #saved_model = model_data['fit_model_save']
        
        for metric_name, metric_value in metrics.items():
            row = {'model': model_name, 'params': params, 'metric': metric_name, 'value': metric_value}
            data.append(row)
    
    df = pd.DataFrame(data)
    return df



def dictionary_without_preds(dictionary):
  # create new dictionary to have a better complexity
  dictionary2 = {k:v for k,v in dictionary.items()}

  for mod in dictionary2['model']:
    del dictionary2['model'][mod]['preds']


  return dictionary2



def dict_tolist(dictionary): #this function will be necessary if we have for example the same model applied on different data in different dictionaries
  dictionaries = [dictionary]
  return dictionaries



def pick_best_model(dictionaries, weights, to_min, scores_file):
    best_score = float('-inf')  # Initialize variable
    metrics_dict = {}

    # first iteration is needed to populate the metrics_dictionary which will be used to normalize the scores
    for dictionary in dictionaries:
        for model in dictionary['model']:
            for metric_name, metric_value in dictionary['model'][model]['metrics'].items():
                if metric_name not in metrics_dict:
                    metrics_dict[metric_name] = []
                metrics_dict[metric_name].append(metric_value)

    # Calculate max and min values for each metric for normalizing
    metrics_min_max = {metric: (min(values), max(values)) for metric, values in metrics_dict.items()}

    # second iteration is needed to calculate the scores 
    for dictionary in dictionaries:
        for model in dictionary['model']:
            values_list = []
            for metric_name, metric_value in dictionary['model'][model]['metrics'].items():
                min_val, max_val = metrics_min_max[metric_name]

                #normalize values for both metrics to be minimized and maximized
                if metric_name in to_min:
                    norm_value = 1 - (metric_value - min_val) / (max_val - min_val)
                else:
                    norm_value = (metric_value - min_val) / (max_val - min_val)
                values_list.append(norm_value)
            
            array1 = np.array(weights)
            array2 = np.array(values_list)
            temp = array1 * array2
            res_list = temp.tolist()
            score = np.sum(res_list)
            print('Score for model ' + model + ' is ' + str(score))
            dictionary['model'][model]['score'] = score

            if score > best_score: #iteratively set new best model
                best_score = score  # Switch to new score
                best_model_name = model
                best_model_file = dictionary['model'][model]['fit_model_save'][0]

    print("Best model is " + best_model_name + " with a score of " + str(best_score))

    #the following function is used to write the scores file so that, if needed, it will be possible to read it instead of looking into the dicitonaries
    with open(scores_file, 'wb') as file: # It is better to re-run this loop at the end, since otherwise we sould have to re-read the pkl file at every iteration of the first outer loop
        final_file_with_values = pickle.dump(dictionaries, file)

    # pick and return best model
    with open(best_model_file, 'rb') as file:
      loaded_model = joblib.load(file)

    return loaded_model



