from sklearn.preprocessing import MinMaxScaler
import pandas as pd

class Preprocessing:
    
    def normalize(df, norm_cols=None):
        """
        normalizes pandas dataframe inplace (no return)
        if norm_cols is specified (not None) only specified columns are normalized, if None: tries to normalize all columns
        #TODO: all kinds of input checks
        """         
        scaler = MinMaxScaler() 
        if norm_cols is None:
            norm_cols = df.columns
            
        scaled_values = scaler.fit_transform(df[norm_cols])
        df.loc[:,:][norm_cols] = scaled_values
    
    def hot_encode(df, target_col, columns):
        
        
        hot_encoded_df = pd.get_dummies(df, columns=columns)
        
        all_cols = list(hot_encoded_df.columns)
        hot_target = all_cols.index(target_col)       
        
        return hot_encoded_df, hot_target
    
    def categorical_to_numerical(df, categorical_type='object'):
        """
        converts categorical values to numerical values inplace (no return)
        categorical_type specifies column's type to convert
        """
        char_cols = df.dtypes.pipe(lambda x: x[x == categorical_type]).index
        label_mapping = {}
        
        for c in char_cols:
            df[c], label_mapping[c] = pd.factorize(df[c])
    
    def boolean_to_numerical(df, bool_type='bool'):
        """
        converts boolean values to numerical values inplace
        """
        bool_cols = df.dtypes.pipe(lambda x: x[x == bool_type]).index
        for c in bool_cols:
            df[c] = df[c].astype(int)
    
    def train_test_split(df, split=0.8):
        train=df.sample(frac=split)
        test=df.drop(train.index)
        
        return train, test
        
#%%
"""
import pandas as pd
import numpy as np
df = pd.read_csv('balance-scale.data', header=None)
df.head()   
#%%
target_col = 0
cols = [1,2,3,4]
hot_encoded = pd.get_dummies(df, columns=cols)
#%%
all_cols = list(hot_encoded.columns)
print(all_cols)
#%%
target_col = all_cols.index(target_col)
#%%
hot_encoded[target_col]
#%%
(np.sort(hot_encoded.groupby(by=target_col).count().iloc[0,:].values)[-1])/len(hot_encoded)
#%%
np.max(np.unique(hot_encoded.groupby(by=target_col).count().iloc[:,:].values))/len(hot_encoded)
#%%
49/625"""