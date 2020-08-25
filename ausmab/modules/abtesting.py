#!/usr/bin/venv python3.7
# -*- coding: utf8 -*-
from typing import List, Any, Union, Dict
import logging

# Data Science
import numpy as np
import pandas as pd
from scipy import stats
from bootstrapped import bootstrap as bs
from bootstrapped import compare_functions as bs_compare
from bootstrapped import stats_functions as bs_stats
from pandas import DataFrame, Series
from pandas.io.parsers import TextFileReader

# Visualisation
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
import plotly.graph_objs as go

filepath = 'AB_Test_Results.csv'
data_init: Union[Union[TextFileReader, Series, DataFrame, None], Any] = pd.read_csv(filepath, sep=',')


class NormalizDataMixin(object):
    def __init__(self, data: pd.DataFrame = pd.DataFrame([]), groups=None, control_groups=None):
        if groups is None:
            groups: Dict[str, list[int]] = dict()
        if control_groups is None:
            control_groups = list()
        self.groups = groups
        self.control_groups = control_groups
        self.data = data

    def apply_linearization(self):
        control = self.data[self.data.group.isin(self.control_groups)]
        experiment = self.data[~self.data.group.isin(self.control_groups)]
        numerator = 0
        denominator = 0
        for row in control:
            numerator += sum(row)
            denominator += len(row)
        control_mean = numerator / denominator
        new_control = [sum(row) - len(row) * control_mean for row in control]
        new_experiment = [sum(row) - len(row) * control_mean for row in experiment]
        return new_control, new_experiment

    @staticmethod
    def apply_bootstrap(self, data):
        '''
        ToDo: data_clear
        '''
        data_clear = data.strip()
        bs.bootstrap(data_clear[(data_clear['VARIANT_NAME'] == 'control')].groupby('USER_ID').action.sum().values,
                     stat_func=bs_stats.mean,
                     num_iterations=10000, iteration_batch_size=300, return_distribution=True)
        return list()

    @property
    def apply_cuped(self):
        return list()


class ConfigMetricsMixin(NormalizDataMixin, object):
    def __init__(self, data: pd.DataFrame = pd.DataFrame([]), control_groups=None):
        if control_groups is None:
            control_groups: List[str] = list()
        self.control_groups = control_groups
        self.data = data

    def get_aggregated_metrics(self, metrics_list=None):
        if metrics_list is None:
            metrics_list = []
        return True

    def get_normalized_metrics(self, metrics_list=None):
        if metrics_list is None:
            metrics_list = []
        control: object = self.data[self.data['group'].isin(self.control_groups)].metric.values
        experiment: object = self.data[~self.data['group'].isin(self.control_groups)].metric.values
        self.apply_linearization(control, experiment)
        return True

    def get_byuser_metrics(self, metrics_list=None):
        if metrics_list is None:
            metrics_list = []
        return True


class DistrTestsMixin(object):
    '''
    Example: DistrTestsMixin.kstest_norm_test(values=[0,0.02])
    If any error then try to remove `self` special word from `def decorate_testfunction(self, origin_func, pvalue: float = 0.05)`
    '''

    def decorate_testfunction(origin_func, pvalue: float = 0.05):
        def apply_wrapper(*args, **kwargs):
            stat = origin_func(*args, **kwargs)
            print(stat)
            print('pvalue_estimated = ', stat[1], ';\t alpha_level = ', pvalue,
                  ';\t pvalue_estimated < alpha_level: ', stat[1] < pvalue)
            if stat[1] > pvalue:
                print('is NOT normal\n')
                return False
            print('is normal\n')
            return True

        return apply_wrapper

    @staticmethod
    @decorate_testfunction
    def kstest_norm_test(values, pvalue: float = 0.05):
        # yield stats.kstest(values, 'norm') # `yield` Cuases in: `TypeError: 'generator' object is not subscriptable`
        return stats.kstest(values, 'norm')

    @staticmethod
    @decorate_testfunction
    def shapiro_norm_test(values, pvalue: float = 0.05):
        # yield stats.shapiro(values) # `yield` Cuases in: `TypeError: 'generator' object is not subscriptable`
        return stats.shapiro(values)


class StatTestsMixin(object):
    def __init__(self):
        super().__init__()


class DataVisualMixin(object):
    def __init__(self, data: pd.DataFrame = pd.DataFrame([]), colormap: List[str] = None, barmode: str = 'overlay',
                       metric_values: List[str] = None
                 ):
        if colormap is None:
            colormap: List[str] = ['rgb(114, 24, 204)', 'rgb(199, 20, 74)', 'rgb(23, 190, 207)', 'rgb(148, 103, 189)']
        if metric_values is None:
            metric_values: List[str] = ['phones', 'users', 'orders']
        self.data = data
        self.metric_values = metric_values
        self.colormap = colormap
        self.layout = dict(
            title=self.data.metric_name,
            xaxis=dict(
                title=self.data.metric_name,
                range=[0, len(self.data.metric_value)]
            ),
            yaxis=dict(
                title=self.data.observations
            ),
            barmode=barmode,
        )

    def plot_barchart(self):
        pass

    def plot_scatter(self):
        pass

    def plot_boxplot(self, metric_index: int = 0):
        trace = list()
        group: str
        assert isinstance(self.data.bucket.unique().to_list, object)
        for group in self.data.group.unique().to_list():
            trace.append(go.Box(
                x=self.data[self.data['bucket'] == group][self.data.metric_name == self.metric_values[metric_index]],
                opacity=0.75, name="Group " + group,
                line=dict(color=self.colormap[int("0x" + group, 0)]),
            ))
        fig = go.Figure(data=trace, layout=self.layout)
        iplot(fig, filename='')

    def plot_histogram(self, metric_index: int = 0):
        trace = list()
        group: str
        assert isinstance(self.data.bucket.unique().to_list, object)
        for group in self.data.group.unique().to_list():
            trace.append(go.Histogram(
                x=self.data[self.data['bucket'] == group][self.metric_values[metric_index]],
                opacity=0.75, name="Group " + group,
                marker_color=self.colormap[int("0x" + group, 0)],
            ))
        fig = go.Figure(data=trace, layout=self.layout)
        iplot(fig, filename='')


class Dataset(ConfigMetricsMixin, DistrTestsMixin, StatTestsMixin, object):
    assert isinstance(Any, object)
    categorical_columns: List[Any]
    numerical_columns: List[Any]

    def __init__(self, data: pd.DataFrame = pd.DataFrame([]), groups=None, control_groups=None):
        """

        :param data:
        :param groups: groups = {'a': buckets=list(), 'b': buckets=list(), 'c': buckets=list(),...}
        :param control_groups: control_groups = ['a']
        """
        super().__init__(data, groups, control_groups)
        if groups is None:
            groups: Dict[str, list[int]] = dict()
        if control_groups is None:
            control_groups = list()
        self.groups = groups
        self.control_groups = control_groups
        self.data = data.copy()
        self.categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
        self.numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']

    def __del__(self):
        pass
