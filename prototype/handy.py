from sklearn.preprocessing import MinMaxScaler

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
            