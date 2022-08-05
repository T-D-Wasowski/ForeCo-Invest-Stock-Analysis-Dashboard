from dash import dcc, html, Input, Output, callback

import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import datetime as dt

import yfinance as yf
from yahoofinancials import YahooFinancials

import forecast

def downloadStockData(stockName, startDate, endDate):
    #aapl_df = yf.download('^NDX', start='2020-01-01', end='2021-01-01', progress=False)
    df = yf.download(stockName, start=startDate, end=endDate)
    return df

stockData_df_1 = downloadStockData("^FTSE", (dt.date.today()-dt.timedelta(days=150)).strftime("%Y-%m-%d"), dt.date.today().strftime("%Y-%m-%d"))
stockData_df_2 = downloadStockData("^GSPC", (dt.date.today()-dt.timedelta(days=150)).strftime("%Y-%m-%d"), dt.date.today().strftime("%Y-%m-%d"))

#forecastResults_1, outputs_test_1, outputs_pred_1, rmse_1, mape_1, history_1 = forecast.forecast(stockData_df_1, 20, 10, [0,1,2,3,4,5], 50)
#forecastResults_2, outputs_test_2, outputs_pred_2, rmse_2, mape_2, history_2 = forecast.forecast(stockData_df_2, 20, 10, [0,1,2,3,4,5], 50)

### ----- FIGURES ----- ###

#Summary Price Sparkline
def createPriceSparklineFigure(stockData_df_1, stockData_df_2):
    priceSparkline = go.Figure(
        data=go.Scatter(
            x=stockData_df_1.index,
            y=stockData_df_1["Close"],
            mode="lines",
            line_color="#4E79A7"
    ))

    priceSparkline.add_trace(go.Scatter(
        x=stockData_df_2.index,
        y=stockData_df_2["Close"],
        mode="lines",
        line_color="#F28E2B"
    ))

    priceSparkline.update_layout(
        template="simple_white",
        annotations=[], 
        overwrite=True,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=10,l=10,b=10,r=10),
        height=82
    )

    priceSparkline.update_xaxes(visible=False, fixedrange=True)
    priceSparkline.update_yaxes(visible=False, fixedrange=True)

    priceSparkline.add_annotation(text="Price:",
                  xref="paper", yref="paper",
                  x=0.00, y=1.25, showarrow=False)

    return priceSparkline

#Summary Volume Bar Sparkline
def createVolumeSparklineFigure(stockData_df_1, stockData_df_2):
    volumeSparkline = go.Figure(
        data = go.Bar(
            x=stockData_df_1.index,
            y=stockData_df_1["Volume"],
            marker_color="#4E79A7" 
    ))

    volumeSparkline.add_trace(go.Bar(
        x=stockData_df_2.index,
        y=stockData_df_2["Volume"],
        marker_color="#F28E2B" 
    ))

    volumeSparkline.update_layout(
        template="simple_white",
        annotations=[], 
        overwrite=True,
        showlegend=False,
        plot_bgcolor="white",
        margin=dict(t=10,l=10,b=10,r=10),
        height=82,
    )

    volumeSparkline.update_xaxes(visible=False, fixedrange=True)
    volumeSparkline.update_yaxes(visible=False, fixedrange=True)

    volumeSparkline.add_annotation(text="Volume:",
                xref="paper", yref="paper",
                x=0.00, y=1.25, showarrow=False)

    return volumeSparkline

### ----- LAYOUT ----- ###

layout = html.Div([
    dbc.Card(
        dbc.CardBody([

            #Error alert
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("An error has occured!")),
                    dbc.ModalBody(["Please re-select valid data parameters and try again."]),
                ],
                id="error-modal",
                centered=True,
                is_open=False,
            ),

            #Row 1 - Title
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1([
                            html.Span("Fore", style={"color": "#F28E2B"}),
                            html.Span("Co ", style={"color": "#4E79A7"}),
                            html.Span("Invest | "),
                            html.Span("Compare", style={"color": "#4E79A7"})
                        ])
                    ], style={"textAlign": "center"})
                ], width={"size": 6, "offset": 3}),

                dbc.Col([
                    dcc.Loading(
                        id="loading",
                        type="circle", #default
                        color="#4E79A7",
                        children=[
                            html.Div([], style={"display": "none"}, id="loading-div")
                        ]    
                    )
                ], width={"size": 1, "offset": 2}, className="pt-4 mt-2")

            ], className="mb-4"),

            #Row 2 - Data select
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Dropdown(
                                        options = [
                                            {"label": "FTSE 100 (^FTSE)", "value": "^FTSE"},
                                            {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
                                            {"label": "Dow Jones Industrial Average (^DJI)", "value": "^DJI"},
                                            {"label": "NASDAQ Composite (^IXIC)", "value": "^IXIC"},
                                            {"label": "Russel 2000 (^RUT)", "value": "^RUT"},
                                            {"label": "NIFTY 50 (^NSEI)", "value": "^NSEI"},
                                            {"label": "Apple Inc. (AAPL)", "value": "AAPL"},
                                            {"label": "Alphabet Inc. (GOOGL)", "value": "GOOGL"},
                                            {"label": "Microsoft Corporation (MSFT)", "value": "MSFT"},
                                            {"label": "Tesla, Inc. (TSLA)", "value": "TSLA"}
                                        ], 
                                        value="^FTSE", 
                                        id="stock-select-dropdown-1"
                                    )
                                ], width=4),
                                dbc.Col([
                                    dcc.Dropdown(
                                        options = [
                                            {"label": "FTSE 100 (^FTSE)", "value": "^FTSE"},
                                            {"label": "S&P 500 (^GSPC)", "value": "^GSPC"},
                                            {"label": "Dow Jones Industrial Average (^DJI)", "value": "^DJI"},
                                            {"label": "NASDAQ Composite (^IXIC)", "value": "^IXIC"},
                                            {"label": "Russel 2000 (^RUT)", "value": "^RUT"},
                                            {"label": "NIFTY 50 (^NSEI)", "value": "^NSEI"},
                                            {"label": "Apple Inc. (AAPL)", "value": "AAPL"},
                                            {"label": "Alphabet Inc. (GOOGL)", "value": "GOOGL"},
                                            {"label": "Microsoft Corporation (MSFT)", "value": "MSFT"},
                                            {"label": "Tesla, Inc. (TSLA)", "value": "TSLA"}
                                        ], 
                                        value="^FTSE", 
                                        id="stock-select-dropdown-2"
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Div([
                                        html.P("Start Date:", style={'textAlign': 'right', "margin": "0px 0px 0px 0px"})
                                    ])
                                ], width=1),
                                dbc.Col([
                                    dcc.DatePickerSingle(
                                        date=(dt.date.today() - dt.timedelta(days=150)),
                                        max_date_allowed=dt.date.today(),
                                        min_date_allowed=(dt.date.today() - dt.timedelta(days=365*3)),
                                        display_format="DD/MM/YYYY",
                                        id="start-date-picker"
                                    )
                                ], width=1),
                                dbc.Col([
                                    html.Div([
                                        html.P("End Date:", style={'textAlign': 'right', "margin": "0px 0px 0px 0px"})
                                    ])   
                                ], width=1),
                                dbc.Col([
                                    dcc.DatePickerSingle(
                                        date=dt.date.today(),
                                        max_date_allowed=dt.date.today(),
                                        min_date_allowed=(dt.date.today() - dt.timedelta(days=365*3)),
                                        display_format="DD/MM/YYYY",
                                        id="end-date-picker"
                                    )
                                ], width=1),
                            ], align="center")
                        ])
                    )
                ])
            ], className="mb-3"),

            #Row 3 - Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.P("Which one is better?"),
                                html.H3("^FTSE", 
                                    id="suggestion-label", 
                                    style={"color": "#F28E2B"}
                                )
                            ], style={'textAlign': 'center'})
                        ])
                    ])
                ], width = 2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.P("Difference in ROI:"),
                                html.H3("{:.2f}%".format(2.56), 
                                    id="change-label",
                                    style={"color": "#F28E2B"}
                                )
                            ], style={'textAlign': 'center'})
                        ])
                    ])
                ], width = 2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.P("Difference in Forecast MAPE"),
                                html.H3("{:.2f}%".format(3.33), 
                                    id="mape-label", 
                                    style={"color": "#4E79A7"}
                                )
                            ], style={'textAlign': 'center'})
                        ])
                    ])
                ], width = 2),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dcc.Graph(
                                        figure=createPriceSparklineFigure(stockData_df_1, stockData_df_2),
                                        config={"displayModeBar" : False},
                                        id="price-sparkline"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(
                                        figure=createVolumeSparklineFigure(stockData_df_1, stockData_df_2),
                                        config={"displayModeBar" : False},
                                        id="volume-sparkline"
                                    )
                                ], width=6)
                            ])
                        ], className="pb-2 pt-4")
                    ])
                ], width=5),
                dbc.Col([
                    html.Div([
                        dcc.Link(dbc.Button("Analyse", color="analyse", className="me-1", size="lg", style={"width": 120}), href="/")
                    ])
                ], width=1, align="center")
            ], className="mb-3"),

        ]),
        color="light",
        className="px-3"
    )
])