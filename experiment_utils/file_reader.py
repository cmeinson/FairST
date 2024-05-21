import pandas as pd
from src.tester import Tester
import numpy as np
from .other_col_reader import OtherColReader


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
    def __init__(self, file_paths):
        if isinstance(file_paths, str): # backwards compatability :)
            file_paths = [file_paths]
        self.file_paths = file_paths
        self.df = None
        self.metrics = []        
        
        self.read_csv()
        self.filters = {}
        self.filterable = self.FILTERABLE + []
        self.columns_shown = self.FILTERABLE + self.metrics
        ocr = OtherColReader(self.df)
        self.add_column = ocr.add_col
        self.relative_metrics_filter = lambda df: df
        

    def read_csv(self):
        dfs = []

        for file_path in self.file_paths:
            try:
                df = pd.read_csv(file_path)
                print(f"File '{file_path}' successfully loaded as DataFrame.")
                dfs.append(self._fix_metrics(df))
            except Exception as e:
                print(f"Error: Unable to open '{file_path}'", e)

        common_columns = set(dfs[0].columns)
        if len(df)>1:
            for df in dfs[1:]:
                common_columns &= set(df.columns)
                
        common_columns = [val for val in dfs[0].columns if val in common_columns] # keeping the og order

        # Select only the common columns from each DataFrame
        for i, df in enumerate(dfs):
            dfs[i] = df[list(common_columns)]
            
        self.df = pd.concat(dfs, ignore_index=True)
        self.df = self._proccess_df(self.df)
        
    def add_metrics(self, names: list):
        for name in names:
            if name not in self.df.columns:
                is_numeric = self.add_column(name)
                if is_numeric:
                    self.metrics.append(name)
                else:
                    self.filterable.append(name)
                self.columns_shown.append(name)
                
            
    def _proccess_df(self, df):
        # remove time and reps columns, keep only reps = 1
        if self.REPS in df.columns:
            df = df[df[self.REPS] == 1]
            
        var_cols = df.filter(like=self.VAR_PREFIX, axis=1).columns.tolist()
        cols_to_drop = var_cols + self.DROP_COLUMNS
        
        cols_to_drop = set(cols_to_drop).intersection(df.columns) # only if the columns are still there
        df = df.drop(columns=cols_to_drop, axis=1)

        df = self._add_new_metrics(df)
        
        non_metric_cols = self.FILTERABLE + [self.ID]
        self.metrics = [col for col in df.columns if col not in non_metric_cols]
        return df
    
    def _fix_metrics(self, df):
        # eq to the og SF
        SF_INV = "[SF] Statistical Parity Subgroup Fairness if 0 was the positive label" # same value

        # was incorrectly implemented first
        DF = "[DF] Differential Fairness"
        DF_INV = "[DF] Differential Fairness if 0 was the positive label"
        ONE_DF = "[DF] Differential Fairness for One Attribute"
        
        if DF_INV not in df.columns:
            return df
        
        one_df_cols = [col for col in df.columns if ONE_DF in col]
                
        df[DF] = df[[DF, DF_INV]].max(axis=1)
        
        return df.drop((one_df_cols + [SF_INV, DF_INV]), axis = 1)
    
    def _add_new_metrics(self, df):
        df['TP'] = df['precision'].mul(df['mean_pred'])
        df['FP'] = df['mean_pred'].mul(1 - df['precision'])
        df['FN'] = df['TP'].mul(1/df['recall'] - 1)
        df['TN'] = 1 - df['TP'] - df['FP'] - df['FN']
        
        tp, fp, fn, tn = df['TP'], df['FP'], df['FN'], df['TN']
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        df['MCC'] = ((tp * tn) - (fp * fn)) / denominator
        return df
        

    def change_other_to_losses(self):
        # edit the "other" col values to have just the lossed used"
        self.df[self.OTHER] = self.df[self.OTHER].apply(self._get_losses_used)
    
    def change_other_to_comment(self):
        # edit the "other" col values to have just the lossed used"
        self.df[self.OTHER] = self.df[self.OTHER].apply(lambda x: x.split(',')[0].strip())

        
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
    
    def get_mean_relative_metrics(self, base = None, use_percent = False):
        """ FILTERED, SELECTED COLUMNS, MEAN, RELATIVE
        returns metric values with respect to the given "base" bias mitigation method of the same experiment.
        returns metric values averaged over all runs of the experiment with the same config.
        returns selected columns based on the row filters.
        by default returns all filterable columns and all metric columns.
        """
        df = self.get_relative_metrics(base, use_percent=use_percent)
        # take mean over same config aka "other col value"
        return self._get_mean_metrics(df)
    
    
    def get_relative_metrics(self, base = None, use_percent = False):
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
                #print("roww", row[self.FILTERABLE + ["accuracy", "id"]])
                #print("base",base_row[self.FILTERABLE + ["accuracy", "id"]])
                for metric in self.metrics:
                    if not isinstance(df.loc[index, metric], str):
                        if use_percent:
                            df.loc[index, metric] = row[metric]/base_row[metric].values[0]
                        else:
                            df.loc[index, metric] = row[metric] - base_row[metric].values[0]
            except Exception as e:
                df.loc[index, metric] = 0
                print("unable to find base for", index, e)

        relative_df = self.relative_metrics_filter(df)
        return relative_df
    
    def _get_mean_metrics(self, df):
        mean_df = df.groupby(self.filterable).mean().reset_index()
        return mean_df  
            
        
