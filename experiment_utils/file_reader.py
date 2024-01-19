import pandas as pd

class ResultsReader:
    TIME = "time"
    REPS = "reps"
    ID = "id"
    
    DATASET = "data"
    ML = "ML method"
    ATTR = "sensitive attrs"
    BIAS_MIT = "bias mitigation"
    OTHER = "other"
    FILTERABLE = [DATASET, ML, ATTR, BIAS_MIT, OTHER]
    
    VAR_PREFIX = "VAR|"  
    
    FYP_VAE = "FYP VAE"
    BASE = "No Bias Mitigation"
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = self.read_csv()
        
        self.filters = {}

    def read_csv(self):
        try:
            df = pd.read_csv(self.file_path)
            print(f"File '{self.file_path}' successfully loaded as DataFrame.")
        except Exception as e:
            print(f"Error: Unable to open '{self.file_path}'", e)
        return self._proccess_df(df)
            
    def _proccess_df(self, df):
        # remove time and reps columns, keep only reps = 1
        df = df[df[self.REPS] == 1]
        var_cols = df.filter(like=self.VAR_PREFIX, axis=1).columns.tolist()
        cols_to_drop = var_cols + [self.TIME, self.REPS]
        
        return df.drop(columns=cols_to_drop, axis=1)
    
    def get_all_column_values(self, column_name) -> list:
        return self.df[column_name].unique().tolist()
        
    def set_filter(self, column_name, values: list):
        if column_name not in self.FILTERABLE:
            raise RuntimeError("invalid filter column name", column_name)
        self.filters[column_name] = values
        
    def clear_filters(self):
        self.filters = {}
        
    def get_filtered_df(self):
        df = self.df.copy()
        
        for col_name, allowed_vals in self.filters.items():
            df = df[df[col_name].isin(allowed_vals)]
            
        return df
