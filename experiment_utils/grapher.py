import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from .file_reader import ResultsReader
from src.ml_models.FYP_VAE.configs import VAEMaskConfig as vae_config  # has losses
from src.tester import Tester  # has bias mits
from .other_col_reader import get_config



class ResultsGrapher:
    def __init__(self, reader: ResultsReader) -> None:
        self.reader = reader
        self.cmap = get_cmap("tab10")
        self.show_legend = True
        self.use_comment_as_label = True

    def plot_metrics_vs_metric(
        self, metric="accuracy", metrics=None, relative=True, mean=True
    ):
        self.reader.add_metrics([metric])
        if metrics is None:
            metrics = self.reader.metrics
        else:
            self.reader.add_metrics(metrics)

        plot_f = lambda p, df, y_m: self._plot_metric_vs_metric(
            p, df, y_m, x_metric=metric
        )
        self._plot_all_filter_values(metrics, plot_f, relative, mean)

    def plot_bias_metrics_vs_metric(self, metric="accuracy", relative=True, mean=True):
        metrics = [m for m in self.reader.metrics if "[" in m]

        plot_f = lambda p, df, y_m: self._plot_metric_vs_metric(
            p, df, y_m, x_metric=metric
        )
        self._plot_all_filter_values(metrics, plot_f, relative, mean)

    def _row_label_generator(self, row):
        if row[ResultsReader.BIAS_MIT] != Tester.FYP_VAE:
            methods = {
                Tester.BASE_ML: "BASE",
                Tester.FAIRMASK: "B:FM",
                Tester.FAIRBALANCE: "B:FB",
                Tester.REWEIGHING: "B:RW",
                Tester.LFR: "LFR",
                Tester.EQODDS: "EqO",
                Tester.EQODDS_ALT: "EqO_ALT",
            }
            if row[ResultsReader.BIAS_MIT] not in methods: return row[ResultsReader.BIAS_MIT]
            return methods[row[ResultsReader.BIAS_MIT]]
        # TODO: DEL temp patch 
        if "mse2" in row[ResultsReader.OTHER]:
            return "[mse2]"
        if "mse" in row[ResultsReader.OTHER]:
            return "mse"
        
        
        name = ""
        losses = [
            vae_config.LATENT_S_ADV_LOSS,
            vae_config.FLIPPED_ADV_LOSS,
            vae_config.KL_SENSITIVE_LOSS,
            vae_config.POS_VECTOR_LOSS,
        ]
        shorts = ["L", "F", "K", "P"]
        losses_used = get_config(row, 'losses_used=', post='])', numeric=False)
        for short, loss in zip(shorts, losses):
            if loss in losses_used:
                name += short
        if name == "":
            return "VAE"
        return name
    
    def get_color(self, label):
        colors = {
            "LF":2,
            "LP":9,
            "LK":3,
            "FK":4,
            "FP":8,
            "KP":6,
            "VAE":7,
            "P": 1,
            "[mse2]": 5,
        }
        if label not in colors:
            return self.cmap(7)
        return self.cmap(colors[label])

    def _get_legend_text(self, other):
        if not self.use_comment_as_label:
            return other[:50]

        label = other.split("'c': '")[-1]
        return label.split("'")[0]

    def _plot_metric_vs_metric(self, cur_plt, df, y_metric, x_metric):
        label_col = ResultsReader.OTHER
        # TODO: make it such that each loss combo is separate))?????/
        for i, label in enumerate(df[label_col].unique()):
            for index, row in df[df[label_col] == label].iterrows():
                point_label=self._row_label_generator(row)
                color = self.get_color(point_label)

                cur_plt.text(
                    row[x_metric],
                    row[y_metric],
                    point_label,
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    color=color,
                )
            cur_plt.scatter(
                df[df[label_col] == label][x_metric],
                df[df[label_col] == label][y_metric],
                label=self._get_legend_text(label),
                color=color,
            )

        plt.xlabel(x_metric)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(y_metric.split("]")[0] + "]")
        

    def _plot_each_metric(self, title, metrics, plot_f, relative, mean):
        df = self.reader.get_filtered_metrics(mean=mean, relative=relative)

        if len(df.index) == 0:
            print("no rows found")
            return

        for y_col in metrics:
            plt.figure(figsize=(8, 6))
            plot_f(plt, df, y_col)
            plt.title(title)
            if self.show_legend:
                plt.legend()
            plt.grid(True)
            plt.show()

    def _plot_all_filter_values(self, metrics, plot_f, relative, mean):
        for dataset in self.reader.get_all_column_values(self.reader.DATASET):
            for ml in self.reader.get_all_column_values(self.reader.ML):
                for attrs in self.reader.get_all_column_values(self.reader.ATTR):
                    self.reader.clear_filters()
                    self.reader.set_filter(ResultsReader.DATASET, [dataset])
                    self.reader.set_filter(ResultsReader.ML, [ml])
                    self.reader.set_filter(ResultsReader.ATTR, [attrs])

                    title = f"{ml} on {dataset} with protected {attrs}"
                    print("_" * 100)
                    print(dataset, ",", ml, attrs)
                    self._plot_each_metric(title, metrics, plot_f, relative, mean)
