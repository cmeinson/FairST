from .file_reader import ResultsReader
from .grapher import ResultsGrapher
from .other_col_reader import OtherColReader


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import os


def metric_has_substring(metric, substrings):
    for sub in substrings:
        if sub in metric:
            return True
    return False

    
def plot_single(grapher, y_metric, x_metric):
    grapher.ax.tick_params(axis='both', which='major', labelsize=6)  # Adjust the font size as needed

    all_metrics = grapher.reader.metrics
    metrics_to_display = [m for m in all_metrics if metric_has_substring(m, [y_metric])]
    
    x_metrics_to_display = [m for m in all_metrics if metric_has_substring(m, [x_metric])]
    print("metrics to display:", metrics_to_display[-1], x_metrics_to_display[0], " | all found ", metrics_to_display, x_metrics_to_display)
    grapher.plot_metrics_vs_metric(mean=True, relative=True, metric=x_metrics_to_display[0], metrics = [metrics_to_display[-1]], style=grapher.STYLE_BASE)
    
    grapher.ax.set_ylabel(y_metric)
    grapher.ax.set_xlabel(x_metric)
    

def plot_multiple_metrics(f, files, dataset, metrics_to_use = ["SF", "DF"],  x_metrics_to_use = ["f1score", "MCC"], size = 2.5):
    reader = ResultsReader(files)
    
    def pre(df):
        df = f(df, reader)
        df = df[(df["data"].str.contains(dataset))]
        return df
    
    reader.df = pre(reader.df)
        
    #reader.relative_metrics_filter = f

    grapher = ResultsGrapher(reader)
    grapher.VERBOSE = False
    grapher.show_legend = False
    grapher.show_plots =  False
    grapher.separate_ml_models = True

    F, axes = plt.subplots(len(x_metrics_to_use), len(metrics_to_use), figsize=(size*(len(metrics_to_use)), size*len(x_metrics_to_use)))
    plt.title(dataset)
        
    for j, x_metric in enumerate(x_metrics_to_use):
        for i, metric in enumerate(metrics_to_use):
            grapher.ax = axes[j][i]
            plot_single(grapher, metric, x_metric)
            
            grapher.ax.set_title(dataset)
                    

    plt.rc('axes', axisbelow=True)
    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()
            
            
def plot_one_run(f, files, dataset, ax,  y_metric = 'SF', x_metric = 'f1score'):
   
    reader = ResultsReader(files)
    
    def pre(df):
        df = f(df, reader)
        df = df[(df["data"].str.contains(dataset))]
        return df
    
    reader.df = pre(reader.df)
    
    #reader.relative_metrics_filter = f # can add this if i want to remove the baseline
    
    grapher = ResultsGrapher(reader)
    grapher.VERBOSE = False
    grapher.show_legend = False
    grapher.show_plots =  False
    grapher.separate_ml_models = False
    
    grapher.ax = ax
    plot_single(grapher, y_metric, x_metric)

def get_all_single_attr_files(include_no_vae = False):
    filename = 'RESULTS_'
    if include_no_vae:
        filename = 'RESULTS_with_no_vae_'
    
    files = []
    datasets = []
    titles = []

    files.append(os.path.join("results",filename+"sex.csv"))
    datasets.append('Default')
    titles.append('Default sex')


    for dataset in ["Adult", "Compas"]:
        for attr in [["race"], ["sex"]]:
            files.append(os.path.join("results",filename+'_'.join(attr)+".csv"))
            datasets.append(dataset)
            titles.append(dataset + ' ' + ' '.join(attr))
            
    return files, datasets, titles


def get_all_multi_attr_files(include_no_vae = False):
    filename = 'RESULTS_'
    #if include_no_vae:
    #    filename = 'RESULTS_with_no_vae_'
        
    files = []
    datasets = []
    titles = []

    for dataset in ["Adult", "Compas"]:
        for attr in [["race", "sex"]]:
            files.append(os.path.join("results",filename+'_'.join(attr)+".csv"))
            datasets.append(dataset)
            titles.append(dataset + ' ' + ' '.join(attr))
            
    return files, datasets, titles


def get_all_result_files(include_no_vae = False):
    s_files, s_datasets, s_titles = get_all_single_attr_files(include_no_vae)
    m_files, m_datasets, m_titles = get_all_multi_attr_files(include_no_vae)
    return s_files+m_files, s_datasets+m_datasets, s_titles+m_titles