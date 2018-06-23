import pandas as pd
import io
import requests
import re
import preprocessing
from oct import OCT
from datetime import datetime as dt
import os
import numpy as np
from copy import deepcopy

def get_results(train_df, test_df, alpha, tree_depth, max_time_per_run=300, threads=None, print_status=False, warm_start=False):
    
    #stats
    stats_data_urls = []
    stats_n_features = []
    stats_n_classes = []
    stats_baseline_accuracies = []
    stats_training_instances = []
    stats_testing_instances = []
    stats_tree_depths = []
    stats_alphas = []
    stats_training_times = []
    stats_gaps = []
    stats_objective_values = []
    stats_trees = []
    stats_training_accuracies = []
    stats_testing_accuracies = []
    
    
    #create oct instance
    o = OCT(data=train_df,
            target=target_col,
            tree_complexity=alpha,
            tree_depth=tree_depth,
            warm_start=warm_start)
    
    stats_n_classes.append(o.n_classes)
    stats_n_features.append(o.n_independent_var)
    stats_baseline_accuracies.append(o.L_hat)
    
    start_time = dt.now()
    o.fit(time_limit=max_time_per_run, threads=threads)
    stop_time = dt.now()
    total_time = stop_time-start_time
    #o.model.Runtime
    stats_training_times.append(total_time.total_seconds())
    
    stats_gaps.append(o.model.MIPGap)
    stats_objective_values.append(o.model.ObjVal)
    
    stats_trees.append(str(o.tree))
    stats_training_accuracies.append(o.training_accuracy())
    stats_testing_accuracies.append(o.accuracy_on_test(test_df, target_col))
    stats_data_urls.append(loc)
    stats_training_instances.append(len(train_df))
    stats_testing_instances.append(len(test_df))
    stats_tree_depths.append(tree_depth)
    stats_alphas.append(alpha)
    
    if print_status:
        print('Training parameters: alpha={0}, tree depth={1}\nBaseline accuracy: {2}'.format(alpha, tree_depth, stats_baseline_accuracies[-1]))
        print('Number of training instances: {0}'.format(stats_training_instances[-1]))
        print('Number of testing instances: {0}'.format(stats_testing_instances[-1]))
        print('Total training time: {0}.'.format(stats_training_times[-1]))
        print('Gap: {0}'.format(stats_gaps[-1]))
        print('Final objective value: {0}'.format(stats_objective_values[-1]))
        print('Accuracy on training set: {0}'.format(stats_training_accuracies[-1]))
        print('Accuracy on testing set: {0}'.format(stats_testing_accuracies[-1]))
        print('Resulting tree: {0}'.format(stats_trees[-1]))
    
    results_df = pd.DataFrame({'data_source':stats_data_urls,
                               'number_of_classes': stats_n_classes,
                               'number of features': stats_n_features,
                               'baseline_accuracy': stats_baseline_accuracies,
                              'training_instances': stats_training_instances,
                              'testing_instances': stats_testing_instances,
                              'tree_depth':stats_tree_depths,
                              'alpha': stats_alphas,
                              'training_time_sec': stats_training_times,
                              'gap': stats_gaps,
                              'objective_value':stats_objective_values,
                              'tree': stats_trees,
                              'training_accuracy':stats_training_accuracies,
                              'testing_accuracy':stats_testing_accuracies})
    
    return results_df

def calc_mean_accuracy_per_alpha(df, mean_cols = ['gap', 'objective_value', 'testing_accuracy', 'testing_instances', 'training_accuracy', 'training_instances', 'number_of_classes', 'number of features', 'baseline_accuracy', 'training_time_sec']):
    
    #df = pd.read_csv(path)
    #grouped = df.groupby('alpha')
    aggregated = df.groupby('alpha')[mean_cols].mean()
    
    return aggregated

def persist_results(dir_name, f_name, results_df, aggregated):
    
    #check if directory experiments exists
    #dir_name = 'experiments' #for now: just subdirectory of current directory    
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    date_string = dt.now().strftime('%Y-%b-%d_%H-%M-%S')
    
    if not isinstance(f_name, str):
        file_name = date_string+'.csv'
    else:
        file_name=f_name+date_string+'.csv'
    
    path = dir_name+'/'+file_name
    results_df.to_csv(path)
    
    ind = path.find('.csv')
    save_to = path[:ind]+'_aggregated'+path[ind:]
    aggregated.to_csv(save_to)    
    print('Results saved to "{0}" and "{1}"'.format(path, save_to))

def baseline_accuracy(df, target_col_name):    
    baseline_accuracy = np.max((np.unique(df.groupby(by=target_col_name).count().iloc[:,:].values)[-1]))/len(df)
    misclassified_points = (1-baseline_accuracy)*len(df)
    print('Baseline Accuracy: {0} and number of misclassified points: {1}'.format(baseline_accuracy, misclassified_points))
    return baseline_accuracy, misclassified_points

def bayesian_tuning(train_val_df, train_val_ratio, tree_depths, target_col_name, val_repeat=8, print_status=True, max_time_per_run=300, warm_start=False):
    
    from bayes_opt import BayesianOptimization
    print('Starting bayesian optimization...')
    norm_cols = [col for col in train_val_df.columns if not col==target_col_name]
    
    #target function for bayesian optimization needs to be defined (strange scoping) 
    all_results_df = []
    all_aggregated_df = []
    
    train_df, val_df = preprocessing.train_test_split(train_val_df, split=train_val_ratio)
    if val_repeat==1:
        random_state = 42
    else:
        random_state = None
    def oct_target(alpha):
        print('Solving ILP for hyperparameter tuning...')
        all_results = []
        tree_depth = tree_depths[0]
        for r in range(val_repeat):
            train_df, val_df = preprocessing.train_test_split(train_val_df, split=train_val_ratio, random_state=random_state)
            preprocessing.normalize(train_df, norm_cols=norm_cols)
            preprocessing.normalize(val_df, norm_cols=norm_cols)
            all_results.append(get_results(train_df=train_df,
                               test_df=val_df,
                               alpha=alpha,
                               tree_depth=tree_depth, 
                               max_time_per_run=max_time_per_run,
                               threads=threads,
                               print_status=print_status,
                               warm_start=warm_start))
        
        results_df = pd.concat(all_results)
        all_results_df.append(results_df)
        aggregated = calc_mean_accuracy_per_alpha(results_df)
        all_aggregated_df.append(aggregated)
        best_alpha_acc = aggregated.max()['testing_accuracy']
        return best_alpha_acc
    
    alpha_min = 0
    train_df, val_df = preprocessing.train_test_split(train_val_df, split=train_val_ratio) # need to split for an initial guess on max alpha
    l_hat, mis_points = baseline_accuracy(train_df, target_col_name)
    alpha_max = mis_points/l_hat
    
    bo = BayesianOptimization(oct_target, {'alpha': (alpha_min, alpha_max)})
    n_iter = int(alpha_max*(2/15))+1
    if n_iter<5:
        n_iter = 10
    bo.maximize(init_points=2, n_iter=n_iter, kappa=2)
    
    
    return pd.concat(all_results_df), pd.concat(all_aggregated_df), bo.res['max']['max_params']['alpha']
    

def gd_tuning(train_val_df, train_val_ratio, tree_depths, target_col_name, val_repeat=8, decrease_threshold = 0.05, p = 0.02, print_status=True, max_time_per_run=300, warm_start=False):
    """
    stop if accuracy is worse than best_accuracy-decrease_threshold
    p: after running algorithm, calculate alpha by taking take mean of all alphas that achieved accuracy within range of p of best acc
    """   
    print('Starting parameter tuning.')    
    train_df, val_df = preprocessing.train_test_split(train_val_df, split=train_val_ratio)
    l_hat, mis_points = baseline_accuracy(train_df, target_col_name)
    alpha_max = mis_points/l_hat
    alpha_min = 0
    #alpha_min = 9.92419825072886
    #alpha_max = 9.92419825072886
    test_n_alphas = 50
    #test_n_alphas = 1
    print('Testing maximum of {0} values for alpha between {1} and {2}.'.format(test_n_alphas, alpha_min, alpha_max))
    alphas = np.linspace(alpha_min, alpha_max, test_n_alphas)
    
    all_results = []
    norm_cols = [col for col in train_val_df.columns if not col==target_col_name]
    
    for no, alpha in enumerate(alphas):
        print('Testing alpha={0}'.format(alpha))   
        
        for tree_depth in tree_depths:
            for r in range(val_repeat):
                #create new train/val
                train_df, val_df = preprocessing.train_test_split(train_val_df, split=train_val_ratio)
                #preprocessing
                #normalize
                preprocessing.normalize(train_df, norm_cols=norm_cols)
                preprocessing.normalize(val_df, norm_cols=norm_cols)
                
                all_results.append(get_results(train_df=train_df,
                                test_df=val_df,
                                alpha=alpha,
                                tree_depth=tree_depth, 
                                max_time_per_run=max_time_per_run,
                                threads=threads,
                                print_status=print_status,
                                warm_start=warm_start)) #list of dataframes
        
        if not alpha==0:        
            results_df = pd.concat(all_results)
            aggregated = calc_mean_accuracy_per_alpha(results_df)
            best_alpha = aggregated.idxmax()['testing_accuracy'] #df is indexed by alpha
            best_alpha_acc = aggregated.max()['testing_accuracy']
            
            #check whether last tested alpha decreased significantly
            alpha_acc = aggregated['testing_accuracy'][alpha] #accuracy for current alpha
            
            if alpha_acc < best_alpha_acc-decrease_threshold:
                print('Accuracy for alpha={0}: {1} is worse than best accuracy for alpha={2}: {3}.\nStopping criterion is met...'.format(alpha, alpha_acc, best_alpha, best_alpha_acc ))
                break
    
    #take mean of top_n best alphas
    #best_alpha = np.mean(aggregated.sort_values(by='testing_accuracy', ascending=False).index[:top_n])
    
    #take mean of all alphas that achieved accuracy within p of best acc
    #p = 0.02
    best_alpha = np.mean(aggregated[aggregated['testing_accuracy']>best_alpha_acc-p].index)
    
    return results_df, aggregated, best_alpha



def hyperparameter_tuning(method, train_val_df, train_val_ratio, tree_depths, target_col_name, val_repeat, warm_start):
    
    if method=='auto' or method=='gradient_descent':
        results_df, aggregated, best_alpha = gd_tuning(train_val_df, train_val_ratio, tree_depths, target_col_name, val_repeat=val_repeat, warm_start=warm_start, max_time_per_run=max_time_per_run)
    if method=='bo':
        results_df, aggregated, best_alpha = bayesian_tuning(train_val_df=train_val_df, train_val_ratio=train_val_ratio, tree_depths=tree_depths, target_col_name=target_col_name, print_status=print_status, val_repeat=val_repeat, warm_start=warm_start, max_time_per_run=max_time_per_run)
        
    return results_df, aggregated, best_alpha

def uci_experiment(loc, target_col, hot_encode_cols, tree_depths, alphas_tuning, repeat, val_repeat=3, train_test_ratio=0.8, train_val_ratio=0.66, header=None, max_time_per_run=300, threads=None, save_to_file=True, print_status=False, f_name=None, character_encoding='utf-8', warm_start=False):
    """
    TODO: currently only numerical datasets are supported (preprocessing needs to be adjusted)
        input checks need to be added
        
    loc: location of dataset (string)
    target_col: number of target column  (to predict)
    tree_depths: list of tree depths to run experiments with
    alphas: list of tree complexity parameters to run experiments with
    repeat: integer indicating how often experiment should be repeated
    train_test_ratio: percentage (between zero and one) indicating how much of data is used for training and validation (rest for testing)
    train_val_ratio: percentage indicating how much of data of training and validation is used for training (rest for validation)
    header: whether or not data under url has a header to load. if no header: set to None, if header: integer indicating row number
    max_time_per_run: how much time is spend for one optimization run
    threads: how many threads in gurobi optimization (None falls back to gurobi default)
    save_to_file: boolean indicating whether results are saved to a file
    filename: filename to save results in (if none and save to file: time will be used as filename)
    characer_encoding: (string) how to decode characters
    """
   
    df = None

    if is_url(loc):
        #read dataframe from url
        html = requests.get(loc).content
        s = io.StringIO(html.decode(character_encoding))
        df = pd.read_csv(s, header=header)
    else:
        df = pd.read_csv(loc)

    #hot encode if needed
    if not hot_encode_cols is None:
        df, target_col = preprocessing.hot_encode(df, target_col, hot_encode_cols)                
    
    #split into training (+validation) and testing
    train_val_df, test_df = preprocessing.train_test_split(df, split=train_test_ratio) #test remains untouched until alpha is chosen
    
    target_col_name = df.columns[target_col]
    norm_cols = [col for col in df.columns if not col==target_col_name]
    
    
    
    #all_results = [] #all (repeat) experimental results for different values of alpha, tree depths
    
    results_df, aggregated, best_alpha = hyperparameter_tuning(method=alphas_tuning, train_val_df=train_val_df, train_val_ratio=train_val_ratio, tree_depths=tree_depths, target_col_name=target_col_name, val_repeat=val_repeat, warm_start=warm_start)
    
    print('Validation done. Best alpha: {0}'.format(best_alpha))
    
    #get final result/ accuracy
    final_results = []
    for tree_depth in tree_depths:
        for r in range(1):
            train_df, val_df = preprocessing.train_test_split(train_val_df, split=0.66)
            preprocessing.normalize(train_df, norm_cols=norm_cols)
            preprocessing.normalize(test_df, norm_cols=norm_cols)                
            
            final_results.append(get_results(train_df=train_df,
                           test_df=test_df,
                           alpha=best_alpha,
                           tree_depth=tree_depth, 
                           max_time_per_run=max_time_per_run,
                           threads=threads,
                           print_status=print_status,
                           warm_start=warm_start)) #list of dataframes

    final_results_df = pd.concat(final_results)
    aggregated_final = calc_mean_accuracy_per_alpha(final_results_df)
    
       
    if save_to_file:
        dir_name='experiments'
        persist_results(dir_name=dir_name,
                        f_name=f_name+'_validation_',
                        results_df=results_df,
                        aggregated=aggregated)
        persist_results(dir_name=dir_name,
                        f_name=f_name+'_final_',
                        results_df=final_results_df,
                        aggregated=aggregated_final)

        
    return results_df

def is_url(string):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    return re.match(regex, string) is not None

if __name__=='__main__':
    #target_col = 4#iris
    warm_start = True
    #target_col=9#fertility diagnosis
    target_col=0 #balance-scale
    train_test_ratio = 0.75
    train_val_ratio = 0.66
    tree_depths=[2] #TODO: CURRENTLY ONLY ONE TREE DEPTH AT A TIME WORKS CORECTLY!!!
    #alpha_tuning='auto'
    alpha_tuning = 'bo' #bayesian optimization
    repeat = 5
    val_repeat=1 #how many times should a validation experiment be repeated (average over all runs is final validation)
    threads = 10
    max_time_per_run = 1800 #seconds
    #loc='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' #iris
    #loc = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt'
    loc = 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'
    #loc = 'data/forecast/forecast.data'
    f_name = 'balance_scale'
    #f_name = 'iris'
    #f_name = 'forecast'
    hot_encode_cols = None #iris, fertility
    #hot_encode_cols = ['outlook','temperature','humidity','windy']
    #f_name = 'fertility_diagnosis'
    f_name+='_'+alpha_tuning
    print_status = True
    for r in range(repeat):
        results = uci_experiment(loc, target_col, hot_encode_cols, tree_depths, alpha_tuning, repeat, val_repeat, train_test_ratio=train_test_ratio, train_val_ratio=train_val_ratio, f_name=f_name, threads=threads, max_time_per_run=max_time_per_run, print_status=print_status, warm_start=warm_start)
    #print(results)
    #%%
