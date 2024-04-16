import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from .file_reader import ResultsReader
from src.ml_models.FYP_VAE.configs import VAEMaskConfig as vae_config  # has losses
from src.tester import Tester  # has bias mits
from .other_col_reader import get_config
import numpy as np
import seaborn as sns   


class ResultsGrapher:
    STYLE_BASE = "style: base"
    STYLE_2D_STD = "style: std line on each axis, same label"
    STYLE_Y_STD_SAME_METRIC = "style: std line on each values with same metric val"
    STYLE_IQR = "style: iqr over each label"
    def __init__(self, reader: ResultsReader) -> None:
        self.reader = reader
        self.cmap = sns.color_palette("flare",11) #get_cmap("tab10")
        self.cmap_bl = sns.color_palette("viridis",6) #get_cmap("tab10")
        self.show_legend = True
        self.use_comment_as_label = False
        self.separate_ml_models = True
        self.show_plots = True
        self.ax = None
        self.VERBOSE = True

    def plot_metrics_vs_metric(
        self,
        metric="accuracy",
        metrics=None,
        relative=True,
        mean=True,
        style = STYLE_BASE,
    ):
        self.reader.add_metrics([metric])
        if metrics is None:
            metrics = self.reader.metrics
        else:
            self.reader.add_metrics(metrics)

        plot_mvm_f = self._plot_metric_vs_metric
        if style == self.STYLE_2D_STD:
            mean = False
            plot_mvm_f = self._plot_metric_vs_metric_mean_each_label
        if style == self.STYLE_IQR:
            mean = False
            plot_mvm_f = self._plot_metric_vs_metric_iqr
        if style == self.STYLE_Y_STD_SAME_METRIC:
            mean = False
            plot_mvm_f = self._plot_metric_vs_metric_mean_each_loss

        plot_f = lambda p, df, y_m: plot_mvm_f(p, df, y_m, x_metric=metric)
        return self._plot_all_filter_values(metrics, plot_f, relative, mean)

    def plot_bias_metrics_vs_metric(self, metric="accuracy", relative=True, mean=True):
        metrics = [m for m in self.reader.metrics if "[" in m]

        plot_f = lambda p, df, y_m: self._plot_metric_vs_metric(
            p, df, y_m, x_metric=metric
        )
        return self._plot_all_filter_values(metrics, plot_f, relative, mean)

    def _row_label_generator(self, row):
        if row[ResultsReader.BIAS_MIT] != Tester.FYP_VAE:
            methods = {
                Tester.BASE_ML: "BASE",
                Tester.FAIRMASK: "B:FM",
                Tester.FAIRBALANCE: "B:FB",
                Tester.REWEIGHING: "B:RW",
                Tester.LFR: "B:LFR",
                Tester.EQODDS: "EqO",
                Tester.EQODDS_ALT: "EqO_ALT",
            }
            if row[ResultsReader.BIAS_MIT] not in methods:
                return row[ResultsReader.BIAS_MIT]
            return methods[row[ResultsReader.BIAS_MIT]]

        name = ""
        losses = [
            vae_config.LATENT_S_ADV_LOSS,
            vae_config.FLIPPED_ADV_LOSS,
            vae_config.KL_SENSITIVE_LOSS,
            vae_config.POS_VECTOR_LOSS,
        ]
        shorts = ["L", "F", "K", "P"]
        losses_used = get_config(row, "losses_used=", post="])", numeric=False)
        for short, loss in zip(shorts, losses):
            if loss in losses_used:
                name += short
        if name == "":
            return "VAE"
        return name

    def get_color(self, label):  # 5
        colors = {
            "K": 6,
            "KP": 10,
            
            "LF": 1,
            "FK": 2,
            "LK": 3,
            "LP": 4,
            "P": 5,
            "L": 7,
            "FP": 8,
            "F": 9,
            
            "VAE": 0
        }
        colors_bl = {            
            "B:FM":4,
            "B:FB":5,
            "B:RW":2,
            "B:LFR":3
        }
        if label in colors:
            return self.cmap[colors[label]]
        
        if label in colors_bl:
            return self.cmap_bl[colors_bl[label]]
        
        return self.cmap_bl[1]

    def _get_legend_text(self, other, point_label = None):
        if point_label is not None:
            labels = {                       
                "B:FM":'FairMask',
                "B:FB":'FairBalance',
                "B:RW":'Reweighing',
                "B:LFR":'LFR',
                'BASE':'Base Model'
            }
            if point_label in labels:
                return labels[point_label]
            
            return 'FVAEM loss '+point_label
        
        if not self.use_comment_as_label:
            return other[:50]

        label = other.split("'c': '")[-1]
        return label.split("'")[0]

    def _plot_metric_vs_metric(self, cur_plt, df, y_metric, x_metric):
        label_col = ResultsReader.OTHER
        # TODO: make it such that each loss combo is separate))?????/
        for i, label in enumerate(df[label_col].unique()):
            for index, row in df[df[label_col] == label].iterrows():
                point_label = self._row_label_generator(row)
                color = self.get_color(point_label)

                cur_plt.text(
                    row[x_metric],
                    row[y_metric],
                    point_label,  # +'|'+ self._get_legend_text(label),
                    fontsize=10,  # 12,
                    ha="right",
                    va="bottom",
                    color=color,
                    alpha=0.9
                )
                cur_plt.scatter(
                    row[x_metric],
                    row[y_metric],
                    label=self._get_legend_text(label, point_label),
                    color=color,
                    alpha=0.8,
                )

        
        cur_plt.tick_params(axis='x', labelrotation=45)
        cur_plt.tick_params(axis='y', labelrotation=45)
        cur_plt.set_xlabel(x_metric)
        cur_plt.set_ylabel(y_metric.split("]")[0] + "]")

    def _plot_metric_vs_metric_mean_each_loss(self, cur_plt, df, y_metric, x_metric):
        label_col = ResultsReader.OTHER
        # TODO: make it such that each loss combo is separate))?????/
        points = {}  # [x val][label] -> list of values
        for i, label in enumerate(df[label_col].unique()):
            for index, row in df[df[label_col] == label].iterrows():
                point_label = self._row_label_generator(row)

                if row[x_metric] not in points:
                    points[row[x_metric]] = {}

                if point_label not in points[row[x_metric]]:
                    points[row[x_metric]][point_label] = []

                points[row[x_metric]][point_label].append(row[y_metric])
        
        #boxplots = []
        for x_val in points:
            for point_label in points[x_val]:
                y_vals = points[x_val][point_label]
                color = self.get_color(point_label)

                cur_plt.text(
                    x_val,
                    np.mean(y_vals),
                    point_label,
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    color=color,
                )

                cur_plt.scatter(
                    x_val,
                    np.mean(y_vals),
                    label=self._get_legend_text(label),
                    color=color,
                )
                
                # TODO: change to Interquartile Range
                cur_plt.errorbar(
                    [x_val],
                    [np.mean(y_vals)],
                    yerr=[
                        np.sqrt(np.var(y_vals))
                    ],  # Use square root of variance as error
                    fmt="none",
                    ecolor=color,
                )
                #boxplots.append(y_vals)
        #plt.boxplot(boxplots)

        cur_plt.tick_params(axis='x', labelrotation=45)
        cur_plt.set_xlabel(x_metric)
        cur_plt.set_ylabel(y_metric.split("]")[0] + "]")

        
    def _plot_metric_vs_metric_iqr(self, cur_plt, df, y_metric, x_metric):
        label_col = ResultsReader.OTHER
        # TODO: make it such that each loss combo is separate))?????/
        points_y = {}  # [label] -> list of values
        for i, label in enumerate(df[label_col].unique()):
            for index, row in df[df[label_col] == label].iterrows():
                point_label = self._row_label_generator(row)

                if point_label not in points_y:
                    points_y[point_label] = []

                points_y[point_label].append(row[y_metric])
        
        boxplots = []
        for point_label in points_y:
            y_vals = points_y[point_label]
            color = self.get_color(point_label)

            cur_plt.text(
                point_label,
                np.mean(y_vals),
                point_label,
                fontsize=12,
                ha="right",
                va="bottom",
                color=color,
            )

            cur_plt.scatter(
                point_label,
                np.mean(y_vals),
                label=self._get_legend_text(label),
                color=color,
            )
            boxplots.append(y_vals)
        cur_plt.boxplot(boxplots,  labels=list(points_y.keys()), positions=range(len(boxplots)))

        cur_plt.tick_params(axis='x', labelrotation=45)
        cur_plt.set_xlabel(x_metric)
        cur_plt.set_ylabel(y_metric.split("]")[0] + "]")


    def _plot_metric_vs_metric_mean_each_label(self, cur_plt, df, y_metric, x_metric):
        label_col = ResultsReader.OTHER
        # TODO: make it such that each loss combo is separate))?????/
        points_y = {}  # [label] -> list of values
        points_x = {}  # [label] -> list of values
        for i, label in enumerate(df[label_col].unique()):
            for index, row in df[df[label_col] == label].iterrows():
                point_label = self._row_label_generator(row)

                if point_label not in points_y:
                    points_y[point_label] = []
                    points_x[point_label] = []

                points_y[point_label].append(row[y_metric])
                points_x[point_label].append(row[x_metric])

        for point_label in points_y:
            y_vals = points_y[point_label]
            x_vals = points_x[point_label]
            x_mu, x_std = np.mean(x_vals), np.std(x_vals)
            y_mu, y_std = np.mean(y_vals), np.std(y_vals)
            color = self.get_color(point_label)

            # cur_plt.axhline(np.mean(y_vals) - (np.sqrt(np.var(y_vals))), linestyle='--', color=color, alpha=0.5)
            # cur_plt.axhline(np.mean(y_vals) + (np.sqrt(np.var(y_vals))), linestyle='--', color=color, alpha=0.5)

            cur_plt.errorbar(
                x_mu,
                y_mu,
                yerr=[y_std],  # Use square root of variance as error
                xerr=[x_std],  # Use square root of variance as error
                fmt="none",
                ecolor=color,
                alpha=0.25,
            )
            
            x_points = [x_mu, x_mu, x_mu-x_std, x_mu+x_std]
            y_points = [y_mu-y_std, y_mu+y_std, y_mu, y_mu]
            cur_plt.scatter(
                (x_points),
                (y_points),
                color=color,
                marker='+',
                alpha=0.5,
            )

        for point_label in points_y:
            y_vals = points_y[point_label]
            x_vals = points_x[point_label]
            color = self.get_color(point_label)

            cur_plt.text(
                np.mean(x_vals),
                np.mean(y_vals),
                point_label,
                fontsize=12,
                ha="right",
                va="bottom",
                color=color,
            )

            cur_plt.scatter(
                np.mean(x_vals),
                np.mean(y_vals),
                label=self._get_legend_text(label),
                color=color,
            )

        cur_plt.tick_params(axis='x', labelrotation=45)
        cur_plt.set_xlabel(x_metric)
        cur_plt.set_ylabel(y_metric.split("]")[0] + "]")


    def _plot_each_metric(self, title, metrics, plot_f, relative, mean):
        df = self.reader.get_filtered_metrics(mean=mean, relative=relative)

        if len(df.index) == 0 and self.VERBOSE:
            print("no rows found")
            return

        for y_col in metrics:
            # HERE!!!
            if self.show_plots:
                F, ax = plt.subplots(1, 1, figsize=(10, 8))
            else:   
                ax = self.ax
                ax.clear()
                
            plot_f(ax, df, y_col)
            ax.set_title(title)
            if self.show_legend:
                ax.legend()
            ax.grid(True)
            if self.show_plots:
                plt.show()
            
            
    def _plot_all_filter_values(self, metrics, plot_f, relative, mean):
        ml_model_values = self.reader.get_all_column_values(self.reader.ML)
        if not self.separate_ml_models:
            ml_model_values = [ml_model_values]
        else: 
            ml_model_values = [[val] for val in ml_model_values]
        
        for dataset in self.reader.get_all_column_values(self.reader.DATASET):
            for ml in ml_model_values:
                for attrs in self.reader.get_all_column_values(self.reader.ATTR):
                    self.reader.clear_filters()
                    self.reader.set_filter(ResultsReader.DATASET, [dataset])
                    self.reader.set_filter(ResultsReader.ML, ml)
                    self.reader.set_filter(ResultsReader.ATTR, [attrs])

                    ml_title = ml[0]
                    if len(ml)>1:
                        ml_title = [n[0] for n in ml]
                    title = f"{ml_title} on {dataset} with protected {attrs}"
                    if self.VERBOSE:
                        print("_" * 100)
                        print(dataset, ",", ml, attrs)
                    self._plot_each_metric(title, metrics, plot_f, relative, mean)
                    