"""
Contains reporting logic and objects.
"""
from __future__ import division

import math
import os
from datetime import datetime

import bt
import ffn
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

import matplotlib
import plotly
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.plotly as py

from bt.backtest import Result
from .cdns import cdns_dict
from ffn import GroupStats
from jinja2 import Environment, FileSystemLoader
from matplotlib import pyplot as plt
from plotly import tools
from plotly.offline import plot


class Report(Result):

    """
    NEED TO FILL IN

    """

    def __init__(self, result):
        self.result = result
        self.result = result
        self.backtest_list = result.backtest_list
        self.backtests = result.backtests

    def get_years(self):
        years = len(self.backtest_list[0].strategy.prices.resample("Y"))
        return years

    def get_nominal_volumes(self, backtest=0):
        temp_transactions = self.get_transactions().reset_index()
        nominal_volumes = temp_transactions.groupby("Security")["quantity"].apply(
            lambda x: x.abs().sum()
        )
        return nominal_volumes

    def get_value_volumes(self, backtest=0):
        strategy_name = self.backtest_list[0].name
        temp_transactions = self.get_transactions(
            strategy_name=strategy_name
        ).reset_index()
        temp_transactions["value_vol"] = (
            temp_transactions["quantity"] * temp_transactions["price"]
        )
        value_volumes = temp_transactions.groupby("Security")["value_vol"].apply(
            lambda x: x.abs().sum()
        )
        return value_volumes

    def get_individual_equity_curves(self, backtest=0):
        weights = self.get_weights()[self.get_weights().columns[1:]]
        weights.columns = [x.split(">")[1] for x in self.get_weights().columns[1:]]
        weights.replace(0, np.NaN, inplace=True)
        equity_curves = (
            (self.backtest_list[backtest].data.pct_change() * weights).dropna(how="all")
            + 1
        ).cumprod().ffill() * 100
        return equity_curves

    def get_monthly_return_table(self, backtest=0):
        table = (self.result[backtest].return_table * 100).round(2)
        return table.to_html(
            classes="table table-hover table-bordered table-striped dt dataTable"
        )

    def get_stats_table_strat(self, backtest=0):
        table = self.result[backtest].stats.to_frame()
        for col in table:
            table[col].iloc[2:] = table[col].iloc[2:].apply(round, args=(3,))
        table.columns = ["Strategy"]
        return table.to_html(
            classes="table strat-stats table-hover table-bordered table-striped  dataTable",
            header=True,
        )

    def get_stats_table_ind(self, equity_curves):
        table = GroupStats(equity_curves).stats
        for col in table:
            table[col].iloc[2:] = table[col].iloc[2:].apply(round, args=(3,))
        return table.to_html(
            classes="table ind-stats table-hover table-bordered table-striped dataTable"
        )

    def get_trade_numbers(self):
        return (
            self.get_transactions()
            .reset_index()
            .groupby("Security")
            .agg("count")["quantity"]
        )

    def get_acf(self, series):
        acf_strat = acf(
            series.loc[series[series != 0.0].first_valid_index() :], alpha=0.05
        )
        acf_strat_df = pd.DataFrame(
            {
                "acf_res": acf_strat[0],
                "acf_lower": [x[0] for x in acf_strat[1]],
                "acf__higher": [x[1] for x in acf_strat[1]],
            }
        )
        return acf_strat_df

    def plot_eq_chart(self, equity_curves, kind="Equity", size="auto"):

        title = kind

        if size == "half":
            width = 840
        elif size == "full":
            width = 1200
        elif size == "auto":
            width = None

        layout = go.Layout(
            title=title + " Chart",
            yaxis=dict(title=kind),
            height=600,
            width=width,
            autosize=True,
            showlegend=True,
            legend=dict(orientation="h"),
            template=self.theme,
        )

        # print(equity_curves)
        trace_list = []

        if kind == "Equity" or kind == "Weights":

            if isinstance(equity_curves, pd.Series):
                x = equity_curves.index
                y = equity_curves.values

                trace_eq = go.Scatter(
                    x=x,
                    y=y,
                    name=self.backtest_list[0].name,
                    marker=dict(line=dict(width=0.5)),
                )

                data = [trace_eq]

            elif isinstance(equity_curves, pd.DataFrame):
                for curve in equity_curves:
                    if ">" in curve:
                        name = curve.split(">")[1]
                    else:
                        name = curve

                    trace_eq = go.Scatter(
                        x=equity_curves[curve].index,
                        y=equity_curves[curve].values,
                        name=name,
                        marker=dict(line=dict(width=0.5)),
                    )

                    trace_list.append(trace_eq)
                data = trace_list

        elif kind == "Drawdown":

            if isinstance(equity_curves, pd.Series):
                x = equity_curves.index
                y = equity_curves.to_drawdown_series().values

                trace_dd = go.Scatter(
                    x=x,
                    y=y,
                    name=self.backtest_list[0].name,
                    marker=dict(line=dict(width=0.5)),
                )
                # line = dict(color = ('rgb(205, 12, 24)')))

                data = [trace_dd]

            elif isinstance(equity_curves, pd.DataFrame):
                eq_trace_list = []
                for curve in equity_curves:
                    trace_dd = go.Scatter(
                        x=equity_curves[curve].index,
                        y=equity_curves[curve].to_drawdown_series().values,
                        name=curve,
                        marker=dict(line=dict(width=0.5)),
                    )

                    trace_list.append(trace_dd)
                data = trace_list

        fig = go.Figure(data=data, layout=layout)

        # fig.layout.template = self.theme

        chart_div = plot(fig, output_type="div", include_plotlyjs=False)

        return chart_div

    def acf_plot(self, acf_df, size="auto"):

        if size == "half":
            width = 840
        elif size == "full":
            width = 1200
        elif size == "auto":
            width = None

        layout = go.Layout(
            title="ACF Chart",
            yaxis=dict(title=None),
            height=600,
            width=width,
            autosize=True,
            showlegend=True,
            legend=dict(orientation="h"),
            template=self.theme,
        )

        trace_list = []

        for series in acf_df:
            trace = go.Scatter(
                x=acf_df[series].index,
                y=acf_df[series].values,
                name=series,
                marker=dict(line=dict(width=0.5)),
            )

            trace_list.append(trace)
        data = trace_list

        fig = go.Figure(data=data, layout=layout)

        chart_div = plot(fig, output_type="div",include_plotlyjs=False)

        return chart_div

    def pie_plot(self, volumes, kind, title):
        labels = volumes.index.values
        values = volumes.round(0).values

        pie = go.Pie(
            labels=labels,
            values=values,
            sort=False,
            hoverinfo="label+percent",
            textinfo="value",
            textfont=dict(size=20),
            marker=dict(line=dict(color=("rgb(22, 96, 167)"), width=2)),
        )

        data = [pie]

        layout = go.Layout(
            title=title + " " + kind,
            legend=dict(orientation="h"),
            showlegend=True,
            margin=go.layout.Margin(l=10, r=10, b=10, t=50, pad=4),
        )

        fig = go.Figure(data=data, layout=layout)

        fig.layout.template = self.theme

        chart_div = plot(fig, output_type="div", include_plotlyjs =False)

        return chart_div

    def dist_plot(self, equity_curves):
        hist_data = []

        if isinstance(equity_curves, pd.Series):
            hist_data.append(equity_curves.pct_change().dropna().values)
            group_labels = ["strategy"]

        else:
            for symbol in equity_curves:
                data = equity_curves[symbol].pct_change().dropna().values
                hist_data.append(data)

            group_labels = equity_curves.columns

        fig = ff.create_distplot(
            hist_data, group_labels, bin_size=0.001, show_rug=False, show_hist=False
        )

        # Add title
        fig["layout"].update(title="Density Plot of Returns", height=565)

        fig.layout.template = self.theme

        chart_div = plot(fig, output_type="div", include_plotlyjs =False)

        return chart_div

    def scatter_matrix(self, dataframe):
        data = [
            dict(label=col, values=round(dataframe[col] * 100, 2)) for col in dataframe
        ]

        color_vals = list(range(dataframe.shape[0]))

        text = [x.strftime("%d %b, %Y") for x in dataframe.index]

        trace1 = go.Splom(
            dimensions=data,
            marker=dict(
                color=color_vals,
                #                               colorbar=dict(tickvals= color_vals),
                size=3,
                # colorscale='Viridis',
                line=dict(width=0.5, color="rgb(230,230,230)"),
            ),
            text=text,
            diagonal=dict(),
        )

        axis = dict(showline=True, zeroline=False, gridcolor="#fff", ticklen=4)

        layout = go.Layout(
            title="",
            dragmode="select",
            # width=100%,
            height=800,
            autosize=True,
            hovermode="closest",
            template=self.theme,
        )

        fig = dict(data=[trace1], layout=layout)

        # fig.layout.template = self.theme

        chart_div = plot(fig, output_type="div", include_plotlyjs =False)

        return chart_div

    def corr_heatmap(self, returns):

        z = returns.corr().iloc[::-1]
        x = z.columns
        y = z.columns[::-1]

        layout = go.Layout(
            title="",
            dragmode="select",
            width=500,
            height=500,
            # autosize=True,
            hovermode="closest",
            template=self.theme,
        )

        trace = go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[
                [0.0, "rgb(165,0,38)"],
                [0.111111111111, "rgb(215,48,39)"],
                [0.222222222222, "rgb(244,109,67)"],
                [0.333333333333, "rgb(253,174,97)"],
                [0.444444444444, "rgb(254,224,144)"],
                [0.555555555556, "rgb(224,243,248)"],
                [0.666666666667, "rgb(171,217,233)"],
                [0.888888888889, "rgb(69,117,180)"],
                [1.0, "rgb(49,54,149)"],
            ],
        )
        data = [trace]

        fig = dict(data=data, layout=layout)

        chart_div = plot(fig, output_type="div", include_plotlyjs =False)

        return chart_div

    def generate_html(self):
        """ Returns parsed HTML text string for report
        """
                
        goal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
        
        # with open(os.path.join(goal_dir,'template_v2.0.html'),'w') as f:

        #     f.write(HTML_STRING)

        env = Environment(loader=FileSystemLoader(goal_dir), autoescape=True)

        template = env.get_template("template.html")

        years = self.get_years()
        # header = self.get_header_data()
        # kpis = self.get_performance_stats()
        eq_chart = self.plot_eq_chart(
            self.backtest_list[0].strategy.prices, kind="Equity"
        )
        returns_table = self.get_monthly_return_table()

        stats_table = self.get_stats_table_strat()

        nominal_volumes = self.pie_plot(
            self.get_nominal_volumes(), "(Shares)", "Volume"
        )
        value_volumes = self.pie_plot(self.get_value_volumes(), "(Value)", "Volume")

        trade_numbers = self.pie_plot(self.get_trade_numbers(), "", "Trades")

        ind_equity_curves = self.get_individual_equity_curves()
        ind_returns = self.pie_plot(ind_equity_curves.iloc[-1], "Individual", "Returns")

        dd_chart = self.plot_eq_chart(
            self.backtest_list[0].strategy.prices, kind="Drawdown"
        )

        eq_ind_chart = self.plot_eq_chart(ind_equity_curves, kind="Equity", size="half")
        dd_ind_chart = self.plot_eq_chart(
            ind_equity_curves, kind="Drawdown", size="half"
        )

        returns_dist = self.dist_plot(self.backtest_list[0].strategy.prices)
        ind_returns_dist = self.dist_plot(ind_equity_curves)

        weights_chart = self.plot_eq_chart(
            self.get_weights().iloc[:, 1:], kind="Weights"
        )

        stats_table_ind = self.get_stats_table_ind(ind_equity_curves)

        ind_returns_df = self.get_individual_equity_curves().pct_change().fillna(0)
        ind_scatter_matrix = self.scatter_matrix(ind_returns_df)

        # scatter_matrix_test = self.scatter_matrix2()
        strat_returns = self.result.prices.pct_change()
        all_returns = pd.concat([strat_returns, ind_returns_df], axis=1)

        heatmap_corr = self.corr_heatmap(all_returns)

        acf_strat = self.get_acf(self.backtest_list[0].strategy.prices.pct_change())

        acf_chart = self.acf_plot(acf_strat, size="auto")

        # all_numbers = {**header, **kpis}
        # all_numbers = {eq_chart}
        html_out = template.render(
            cdn=self.cdn,
            years=years,
            eq_chart=eq_chart,
            returns_table=returns_table,
            stats_table=stats_table,
            heatmap_corr=heatmap_corr,
            nominal_volumes=nominal_volumes,
            value_volumes=value_volumes,
            trade_numbers=trade_numbers,
            ind_returns=ind_returns,
            dd_chart=dd_chart,
            eq_ind_chart=eq_ind_chart,
            dd_ind_chart=dd_ind_chart,
            returns_dist=returns_dist,
            ind_returns_dist=ind_returns_dist,
            weights_chart=weights_chart,
            stats_table_ind=stats_table_ind,
            ind_scatter_matrix=ind_scatter_matrix,  # scatter_matrix_test=scatter_matrix_test),
            acf_chart=acf_chart,
        )
        return html_out

    def generate_html_report(
        self, theme="plotly_dark", cdn="cyborg", output_file="report"
    ):
        """ Returns HTML report with backtest results
        """

        self.cdn = cdns_dict[cdn]
        self.theme = theme
        html = self.generate_html()
        #goal_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "reports"))
        outfile = os.path.join(os.getcwd(), output_file + ".html")

        file = open(outfile, "w")
        file.write(html)
        file.close()
        msg = "See {} for report with backtest results."
        print(msg.format(outfile))
