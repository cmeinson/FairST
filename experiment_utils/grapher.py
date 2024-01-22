import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from .file_reader import ResultsReader
from src.ml_models.FYP_VAE.configs import VAEMaskConfig as vae_config  # has losses
from src.tester import Tester  # has bias mits


class ResultsGrapher:
    def __init__(self, reader: ResultsReader) -> None:
        self.reader = reader
        self.cmap = get_cmap("tab10")
        self.show_legend = True
        self.use_comment_as_label = True

    def plot_metrics_vs_metric(
        self, metric="accuracy", metrics=None, relative=True, mean=True
    ):
        if metrics is None:
            metrics = self.reader.metrics

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

    def plot_metrics_vs_epochs(self, metrics=None, relative=True, mean=True):
        if metrics is None:
            metrics = self.reader.metrics

        plot_f = lambda p, df, y_m: self._plot_epochs_vs_metric(p, df, y_m)
        self._plot_all_filter_values(metrics, plot_f, relative, mean)

    def _row_label_generator(self, row):
        if row[ResultsReader.BIAS_MIT] != Tester.FYP_VAE:
            methods = {
                Tester.BASE_ML: "BASE",
                Tester.FAIRMASK: "B:FM",
                Tester.FAIRBALANCE: "B:FB",
                Tester.REWEIGHING: "B:RW",
                Tester.FFVAE: "FFVAE",
            }
            return methods[row[ResultsReader.BIAS_MIT]]

        name = ""
        losses = [
            vae_config.LATENT_S_ADV_LOSS,
            vae_config.FLIPPED_ADV_LOSS,
            vae_config.KL_SENSITIVE_LOSS,
            vae_config.POS_VECTOR_LOSS,
        ]
        shorts = ["L", "F", "K", "P"]
        for short, loss in zip(shorts, losses):
            if loss in row[ResultsReader.OTHER]:
                name += short
        if name == "":
            return "VAE"
        return name

    def _get_epochs(self, row):
        other = row[ResultsReader.OTHER]
        if "VAEMaskConfig" not in other:
            return 0

        other = other.split("epochs=")[-1]
        return int(other.split(",")[0])

    def _get_label_text(self, other):
        if not self.use_comment_as_label:
            return other[:50]

        label = other.split("'c': '")[-1]
        return label.split("'")[0]

    def _plot_metric_vs_metric(self, cur_plt, df, y_metric, x_metric):
        label_col = ResultsReader.OTHER
        for i, label in enumerate(df[label_col].unique()):
            color = self.cmap(i)
            cur_plt.scatter(
                df[df[label_col] == label][x_metric],
                df[df[label_col] == label][y_metric],
                label=self._get_label_text(label),
                color=color,
            )
            for index, row in df[df[label_col] == label].iterrows():
                cur_plt.text(
                    row[x_metric],
                    row[y_metric],
                    self._row_label_generator(row),
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    color=color,
                )

        plt.xlabel(x_metric)
        plt.ylabel(y_metric.split("]")[0] + "]")

    def _plot_epochs_vs_metric(self, cur_plt, df, y_metric):
        label_col = ResultsReader.OTHER
        for i, label in enumerate(df[label_col].unique()):
            color = self.cmap(i)
            for index, row in df[df[label_col] == label].iterrows():
                cur_plt.scatter(
                    [self._get_epochs(row)],
                    [row[y_metric]],
                    label=[self._row_label_generator(row)],
                    color=color,
                )
                cur_plt.text(
                    self._get_epochs(row),
                    row[y_metric],
                    self._row_label_generator(row),
                    fontsize=12,
                    ha="right",
                    va="bottom",
                    color=color,
                )

        plt.xlabel("epochs")
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
