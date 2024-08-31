import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class AdvancedVisualizer:
    def __init__(self, trading_system):
        self.trading_system = trading_system
        self.app = dash.Dash(__name__)
        self.setup_layout()

    def setup_layout(self):
        self.app.layout = html.Div([
            html.H1("Advanced Gold Trading System Dashboard"),
            dcc.Tabs([
                dcc.Tab(label='Price and Signals', children=[
                    dcc.Graph(id='price-signals-chart')
                ]),
                dcc.Tab(label='Portfolio Performance', children=[
                    dcc.Graph(id='portfolio-performance-chart')
                ]),
                dcc.Tab(label='Risk Metrics', children=[
                    dcc.Graph(id='risk-metrics-chart')
                ]),
                dcc.Tab(label='Liquidity and Execution Quality', children=[
                    dcc.Graph(id='liquidity-execution-chart')
                ]),
                dcc.Tab(label='Feature Importance', children=[
                    dcc.Graph(id='feature-importance-chart')
                ])
            ]),
            dcc.Interval(
                id='interval-component',
                interval=5*60*1000,  # in milliseconds, update every 5 minutes
                n_intervals=0
            )
        ])

        self.setup_callbacks()

    def setup_callbacks(self):
        @self.app.callback(
            [Output('price-signals-chart', 'figure'),
             Output('portfolio-performance-chart', 'figure'),
             Output('risk-metrics-chart', 'figure'),
             Output('liquidity-execution-chart', 'figure'),
             Output('feature-importance-chart', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_charts(n):
            return (
                self.create_price_signals_chart(),
                self.create_portfolio_performance_chart(),
                self.create_risk_metrics_chart(),
                self.create_liquidity_execution_chart(),
                self.create_feature_importance_chart()
            )

    def create_price_signals_chart(self):
        df = self.trading_system.processed_data.copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

        # Price chart
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Price'),
                      row=1, col=1)

        # Signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]

        fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'],
                                 mode='markers', name='Buy Signal',
                                 marker=dict(symbol='triangle-up', size=10, color='green')),
                      row=1, col=1)

        fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'],
                                 mode='markers', name='Sell Signal',
                                 marker=dict(symbol='triangle-down', size=10, color='red')),
                      row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)

        fig.update_layout(title='Gold Price and Trading Signals',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          xaxis_rangeslider_visible=False)

        return fig

    def create_portfolio_performance_chart(self):
        df = self.trading_system.backtester.results.copy()
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df.index, y=df['Equity'], name='Portfolio Value'))
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gold Price'))

        fig.update_layout(title='Portfolio Performance vs Gold Price',
                          xaxis_title='Date',
                          yaxis_title='Value')

        return fig

    def create_risk_metrics_chart(self):
        df = self.trading_system.backtester.results.copy()
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03)

        # Rolling Sharpe Ratio
        returns = df['Returns']
        rolling_sharpe = (returns.rolling(window=252).mean() / returns.rolling(window=252).std()) * np.sqrt(252)
        fig.add_trace(go.Scatter(x=df.index, y=rolling_sharpe, name='Rolling Sharpe Ratio'), row=1, col=1)

        # Drawdown
        drawdown = (df['Equity'] / df['Equity'].cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=df.index, y=drawdown, name='Drawdown %'), row=2, col=1)

        fig.update_layout(title='Risk Metrics',
                          xaxis_title='Date',
                          yaxis_title='Value')

        return fig

    def create_liquidity_execution_chart(self):
        df = pd.DataFrame({
            'Date': self.trading_system.execution_history.index,
            'Liquidity': self.trading_system.execution_history['liquidity_estimate'],
            'Slippage': self.trading_system.execution_history['actual_slippage'],
            'Execution Time': self.trading_system.execution_history['execution_time']
        })

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03)

        fig.add_trace(go.Scatter(x=df['Date'], y=df['Liquidity'], name='Estimated Liquidity'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Slippage'], name='Actual Slippage'), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Execution Time'], name='Execution Time'), row=3, col=1)

        fig.update_layout(title='Liquidity and Execution Quality',
                          xaxis_title='Date',
                          yaxis_title='Value')

        return fig

    def create_feature_importance_chart(self):
        liquidity_importance = self.trading_system.liquidity_estimator.get_feature_importance()
        execution_importance = self.trading_system.execution_quality_predictor.get_feature_importance()

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Liquidity Estimation', 'Execution Quality Prediction'))

        fig.add_trace(go.Bar(x=liquidity_importance['importance'], y=liquidity_importance['feature'],
                             orientation='h', name='Liquidity Features'), row=1, col=1)

        fig.add_trace(go.Bar(x=execution_importance['slippage_importance'], y=execution_importance['feature'],
                             orientation='h', name='Slippage Features'), row=1, col=2)

        fig.update_layout(title='Feature Importance',
                          yaxis_title='Features',
                          xaxis_title='Importance')

        return fig

    def run_dashboard(self):
        self.app.run_server(debug=True)

# You can add more visualization methods here as needed