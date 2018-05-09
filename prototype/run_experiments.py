import pandas as pd
import io
import requests
from handy import Preprocessing
from oct_prototype import OCT
from datetime import datetime as dt
import os

def uci_experiment(url, target_col, tree_depths, alphas, repeat, train_test_ratio=0.8, header=None, max_time_per_run=300, threads=None, save_to_file=True, print_status=False, dataset_name=None, character_encoding='utf-8'):
    """
    TODO: currently only numerical datasets are supported (preprocessing needs to be adjusted)
        input checks need to be added
        
    url: url to data on uci repository (string)
    target_col: number of target column  (to predict)
    tree_depths: list of tree depths to run experiments with
    alphas: list of tree complexity parameters to run experiments with
    repeat: integer indicating how often experiment should be repeated
    train_test_ratio: percentage (between zero and one) indicating how much of data is used for training (rest for testing)
    header: whether or not data under url has a header to load. if no header: set to None, if header: integer indicating row number
    max_time_per_run: how much time is spend for one optimization run
    threads: how many threads in gurobi optimization (None falls back to gurobi default)
    save_to_file: boolean indicating whether results are saved to a file
    filename: filename to save results in (if none and save to file: time will be used as filename)
    characer_encoding: (string) how to decode characters
    """
    
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
    
    
    #read dataframe from url
    html = requests.get(url).content
    s = io.StringIO(html.decode(character_encoding))
    df = pd.read_csv(s, header=header)
    
    for alpha in alphas:
        for tree_depth in tree_depths:
            for r in range(repeat):
                stats_data_urls.append(url)
                #print(df.head())
                
                #preprocessing
                #split into training and testing
                train_df, test_df = Preprocessing.train_test_split(df, split=train_test_ratio)
                stats_training_instances.append(len(train_df))
                stats_testing_instances.append(len(test_df))
                
                stats_tree_depths.append(tree_depth)
                stats_alphas.append(alpha)
                
                #normalize
                target_col_name = df.columns[target_col]
                #print(target_col_name)
                norm_cols = [col for col in df.columns if not col==target_col_name]
                #print(norm_cols)
                Preprocessing.normalize(train_df, norm_cols=norm_cols)
                Preprocessing.normalize(test_df, norm_cols=norm_cols)
                
                #create oct instance
                o = OCT(data=train_df,
                        target=target_col,
                        tree_complexity=alpha,
                        tree_depth=tree_depth)
                
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
    if save_to_file:
        #check if directory experiments exists
        dir_name = 'experiments' #for now: just subdirectory of current directory
        
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        date_string = dt.now().strftime('%Y-%b-%d-%H:%M:%S')
        if not isinstance(dataset_name, str):
            file_name = date_string+'.csv'
        else:
            file_name=dataset_name+'_'+date_string+'.csv'
        
        path = dir_name+'/'+file_name
        print('Results saved to: {0}'.format(path))
        results_df.to_csv(path)
    
    return results_df

if __name__=='__main__':
    target_col = 4#iris
    #target_col=9#fertility diagnosis
    tree_depths=[1,2,3,4]
    alphas=[0.05]
    repeat=1
    threads = 2
    max_time_per_run = 900 #seconds
    url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data' #iris
    #url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00244/fertility_Diagnosis.txt'
    dataset_name = 'iris'
    #dataset_name = 'fertility_diagnosis'
    print_status = True
    results = uci_experiment(url, target_col, tree_depths, alphas, repeat, dataset_name=dataset_name, threads=threads, max_time_per_run=max_time_per_run, print_status=print_status)
    #print(results)
    #%%