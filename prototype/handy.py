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
        