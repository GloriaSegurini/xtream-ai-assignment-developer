from train_and_pick_bestModel import *

def full_pipeline(models, x_train, y_train, x_test, metrics, scores_file, y_test, weights, metric_tomin, best_model_file):
    dictionaries_list = [] #create a full list of dictionaries

    final_dictionary = models_training(models, x_train,y_train, x_test, False) # train models
    dictionaries_list.append(final_dictionary) # append dictionary for later
    models_evaluation(models, metrics, scores_file, final_dictionary, y_test) # evaluate models and keep history
    plot_gof(y_test, final_dictionary) #plot results

    #Check predictions
    for model in final_dictionary['model']:
        if any (final_dictionary['model'][model]['preds'][0] < 0):
            scores_file = f'scores_file_{model}'
            print("Model " + model + " predicts negative values!")
            y_train_log = np.log(y_train)
            mod = final_dictionary['model'][model]['model_ready_before_fit'] # take model ready to be used (not fit)
            new_dictionary = models_training([mod], x_train, y_train_log, x_test, True) # make new dictionary for current model
            dictionaries_list.append(new_dictionary) #save it for later use
            models_evaluation([mod], metrics, scores_file, new_dictionary, y_test) #evaluate new model
            plot_gof(y_test, new_dictionary) #plot new results

    new_dictionaries_list = [] #create new list for dictionaries without predictions
    for dict in dictionaries_list:
        new_dictionaries_list.append(dictionary_without_preds(dict)) 

    #now we have a list of dicitonaries without predictions (so as to speed time complexity up)
    best_model = pick_best_model(new_dictionaries_list, weights, metric_tomin, best_model_file)

    return best_model