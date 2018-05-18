import pandas as pd
from glob import glob

def calc_mean_accuracy_per_alpha(path):
    
    mean_cols = ['gap', 'objective_value', 'testing_accuracy', 'testing_instances', 'training_accuracy', 'training_instances', 'number_of_classes', 'number of features', 'baseline_accuracy']
    
    df = pd.read_csv(path)
    
    #grouped = df.groupby('alpha')
    aggregated = df.groupby('alpha')[mean_cols].mean()
    
    ind = path.find('.csv')
    save_to = path[:ind]+'_aggregated'+path[ind:]
    
    aggregated.to_csv(save_to)
    
    return aggregated
    

if __name__=='__main__':
    
    paths = glob('results/*')
    
    for path in paths:
        results = calc_mean_accuracy_per_alpha(path)
    #%%
    ind = path.find('.csv')
    
    path[:33]+'_aggregated'+path[33:]