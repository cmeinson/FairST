import pandas as pd
from src.tester import Tester

class ResultsReader:
    # TODO: allor reading of multiple files
    DROP_COLUMNS = ["time", 'bias mit ML method', "reps"]
    REPS = "reps"
    
    ID = "id"
    
    DATASET = "data"
    ML = "ML method"
    ATTR = "sensitive attrs"
    BIAS_MIT = "bias mitigation"
    OTHER = "other"
    FILTERABLE = [DATASET, ML, ATTR, BIAS_MIT, OTHER]
    
    VAR_PREFIX = "VAR|"  
    FYP_VAE = Tester.FYP_VAE
    BASE = Tester.BASE_ML
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.metrics = []
        
        
        self.read_csv()
        self.filters = {}
        self.columns_shown = self.FILTERABLE + self.metrics

    def read_csv(self):
        try:
            df = pd.read_csv(self.file_path)
            print(f"File '{self.file_path}' successfully loaded as DataFrame.")
        except Exception as e:
            print(f"Error: Unable to open '{self.file_path}'", e)
            
        self.df = self._proccess_df(df)
            
    def _proccess_df(self, df):
        # remove time and reps columns, keep only reps = 1
        df = df[df[self.REPS] == 1]
        var_cols = df.filter(like=self.VAR_PREFIX, axis=1).columns.tolist()
        cols_to_drop = var_cols + self.DROP_COLUMNS
        df = df.drop(columns=cols_to_drop, axis=1)
    
        non_metric_cols = self.FILTERABLE + [self.ID]
        self.metrics = [col for col in df.columns if col not in non_metric_cols]
        
        return df

    def change_other_to_losses(self):
        # edit the "other" col values to have just the lossed used"
        self.df[self.OTHER] = self.df[self.OTHER].apply(self._get_losses_used)
        
    def _get_losses_used(self, other):
        other = other.split("losses_used=[")[-1]
        other = other.split("]")[0]
        return other
    
    def get_all_column_values(self, column_name) -> list:
        return self.df[column_name].unique().tolist()
        
    def set_filter(self, column_name, values: list):
        if column_name not in self.FILTERABLE:
            raise RuntimeError("invalid filter column name", column_name)
        self.filters[column_name] = values
        
    def clear_filters(self):
        self.filters = {}

    def set_columns_shown(self, metrics = None, other_columns_shown = None):
        if metrics is None:
            metrics = self.metrics
        if other_columns_shown is None:
            other_columns_shown = self.FILTERABLE
        self.columns_shown = other_columns_shown + metrics
        
    def get_filtered_df(self):
        """FILTERED,
        returns all columns based on the row filters"""
        df = self.df.copy()
        
        for col_name, allowed_vals in self.filters.items():
            df = df[df[col_name].isin(allowed_vals)]
            
        return df
    
    def get_filtered_metrics(self, mean=False, relative=False, base=None):
        """ FILTERED, SELECTED COLUMNS,
        returns selected columns based on the row filters.
        by default returns all filterable columns and all metric columns."""
        if mean and relative:
            return self.get_mean_relative_metrics(base)[self.columns_shown]
        if relative:
            return self.get_relative_metrics(base)[self.columns_shown]
        if mean:
            return self.get_mean_metrics()[self.columns_shown]
        
        return self.get_filtered_df()[self.columns_shown]
        
    
    def get_mean_metrics(self):
        """ FILTERED, SELECTED COLUMNS, MEAN
        returns metric values averaged over all runs of the experiment with the same config.
        returns selected columns based on the row filters.
        by default returns all filterable columns and all metric columns.
        """
        
        df = self.get_filtered_df()
        # take mean over same config aka "other col value"
        return self._get_mean_metrics(df)
    
    def get_mean_relative_metrics(self, base = None):
        """ FILTERED, SELECTED COLUMNS, MEAN, RELATIVE
        returns metric values with respect to the given "base" bias mitigation method of the same experiment.
        returns metric values averaged over all runs of the experiment with the same config.
        returns selected columns based on the row filters.
        by default returns all filterable columns and all metric columns.
        """
        df = self.get_relative_metrics(base)
        # take mean over same config aka "other col value"
        return self._get_mean_metrics(df)
    
    
    def get_relative_metrics(self, base = None):
        """ FILTERED, SELECTED COLUMNS, RELATIVE
        returns metric values with respect to the given "base" bias mitigation method of the same experiment.
        returns selected columns based on the row filters.
        by default returns all filterable columns and all metric columns.
        """
        if base is None:
            base = self.BASE
        df = self.get_filtered_df().reset_index()
        
        for index, row in df.iterrows():
            try:
                base_row = self.df[
                    (self.df[self.ID] == row[self.ID]) & (self.df[self.DATASET] == row[self.DATASET]) & 
                    (self.df[self.ML] == row[self.ML]) & (self.df[self.ATTR] == row[self.ATTR]) & 
                    (self.df[self.BIAS_MIT] == base)
                    ]
                for metric in self.metrics:
                    df.loc[index, metric] = row[metric] - base_row[metric].values[0]
            except:
                df.loc[index, metric] = 0
                print("unable to find base for", index)

        return df
    
    def _get_mean_metrics(self, df):
        mean_df = df.groupby(self.FILTERABLE).mean().reset_index()
        return mean_df  
            
        
