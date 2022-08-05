from dash import dcc, html, Input, Output, callback, dash_table
import dash

import dash_bootstrap_components as dbc

import requests, lxml
import re
from bs4 import BeautifulSoup

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import datetime as dt

import yfinance as yf
from yahoofinancials import YahooFinancials

import forecast

def downloadStockData(stockName, startDate, endDate):
    #aapl_df = yf.download('^NDX', start='2020-01-01', end='2021-01-01', progress=False)
    df = yf.download(stockName, start=startDate, end=endDate)
    return df

stockCode = "AAPL"

stockData_df = downloadStockData(stockCode, (dt.date.today()-dt.timedelta(days=150)).strftime("%Y-%m-%d"), dt.date.today().strftime("%Y-%m-%d"))
forecastResults, outputs_test, outputs_pred, rmse, mape, history = forecast.forecast(stockData_df, 20, 10, [0,1,2,3,4,5], 50)
stockData_ticker = yf.Ticker(stockCode)

### ----- FIGURES ----- ###

#Summary Price Sparkline
def createPriceSparklineFigure(stockData_df):
    priceSparkline = go.Figure(
        data=go.Scatter(
            x=stockData_df.index,
            y=stockData_df["Close"],
            mode="lines",
            line_color="#4E79A7"
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
def createVolumeSparklineFigure(stockData_df):
    volumeSparkline = go.Figure(
        data = go.Bar(
            x=stockData_df.index,
            y=stockData_df["Volume"],
            marker_color="#4E79A7" #F28E2B
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

#Forecast Results Line Chart
def createForecastResultsLineFigure(stockData_df, forecastResults, stockCode):
    forecastResultsLine = go.Figure()

    forecastPlot = pd.concat([stockData_df.iloc[[-1], [3]], forecastResults])

    forecastResultsLine.add_trace(go.Scatter(
            x=stockData_df.index,
            y=stockData_df["Close"],
            mode="lines",
            line_color="#4E79A7",
            name="Actual"
    ))
    forecastResultsLine.add_trace(go.Scatter(
            x=forecastPlot.index,
            y=forecastPlot["Close"],
            mode="lines",
            line_color="#F28E2B",
            name="Forecast"
    ))

    forecastResultsLine.add_vline(x=stockData_df.iloc[[-1], [3]].index[0], line_width=2, line_dash="dash", line_color="#7a7a7a")

    forecastResultsLine.update_layout(
        title="{0} Close Price Forecast".format(stockCode) ,
        xaxis_title="Date",
        yaxis_title="Price",
        template="simple_white",
        margin=dict(t=10,l=10,b=10,r=10),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1, #-0.265
            xanchor="right",
            x=1
        )
    )

    return forecastResultsLine

#Forecast Test Line Chart
def createForecastTestLineFigure(outputs_test, outputs_pred, stockCode):
    forecastTestLine = go.Figure()

    for i in range(len(outputs_test[1, :])):
        forecastTestLine.add_trace(go.Scatter(
            y=outputs_test[:, i],
            mode="lines",
            line_color="#4E79A7",
            showlegend=False
        ))
    for i in range(len(outputs_pred[1, :])): 
        forecastTestLine.add_trace(go.Scatter(
            y=outputs_pred[:, i],
            mode="lines",
            line_color="#F28E2B",
            showlegend=False
        ))

    forecastTestLine.update_layout(
        title="{0} Forecast Model Testing".format(stockCode),
        xaxis_title="No. of Tests",
        yaxis_title="Price",
        template="simple_white",
        margin=dict(t=10,l=10,b=10,r=10),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1
        )
    )

    forecastTestLine['data'][0]['showlegend']=True
    forecastTestLine['data'][0]['name']='Actual'
    forecastTestLine['data'][(0+len(outputs_test[1, :]))]['showlegend']=True
    forecastTestLine['data'][(0+len(outputs_test[1, :]))]['name']='Predicted'

    return forecastTestLine

#Forecast Training Line Chart
def createForecastTrainingLineFigure(history, stockCode):
    forecastTrainingLine = go.Figure()

    forecastTrainingLine.add_trace(go.Scatter(
        y=history.history['mse'],
        mode="lines",
        line_color="#4E79A7",
        name="MSE"
    ))
    forecastTrainingLine.add_trace(go.Scatter(
        y=history.history['mae'],
        mode="lines",
        line_color="#F28E2B",
        name="MAE"
    ))

    forecastTrainingLine.update_layout(
        title="{0} Forecast Model Training".format(stockCode),
        xaxis_title="No. of Epochs",
        yaxis_title="Error Proportion",
        template="simple_white",
        margin=dict(t=10,l=10,b=10,r=10),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.1,
            xanchor="right",
            x=1
        )
    )

    return forecastTrainingLine

#Create candlestick chart figure
def createPriceCandleFigure(stockData_df, stockCode):
    priceCandle = go.Figure(data=[
        go.Candlestick(
            x=stockData_df.index,
            open=stockData_df['Open'],
            high=stockData_df['High'],
            low=stockData_df['Low'],
            close=stockData_df['Close'],
        )
    ])

    priceCandle.update_layout(
        title="{0} Price Movement".format(stockCode),
        yaxis_title="Price",
        template="simple_white",
        margin=dict(t=60,l=10,b=10,r=10),
        height=500 #Maybe remove this?
    )

    priceCandle.data[0].increasing.fillcolor = "#6d93bb"
    priceCandle.data[0].increasing.line.color = "#4E79A7"
    priceCandle.data[0].decreasing.fillcolor = "#f5a85b"
    priceCandle.data[0].decreasing.line.color = "#F28E2B"

    return priceCandle

#Create volume bar chart figure
def createVolumeBarFigure(stockData_df, stockCode):
    volumeBar = go.Figure(
        data = go.Bar(
            x=stockData_df.index,
            y=stockData_df["Volume"],
            marker_color="#4E79A7" #F28E2B
    ))

    volumeBar.update_layout(
        title="{0} Trading Volume".format(stockCode),
        yaxis_title="Volume",
        template="simple_white",
        margin=dict(t=60,l=10,b=10,r=10),
        height=250,
    )

    return volumeBar

### ----- TABLES ----- ###

def createInfoTable(stockData_ticker):

    info_df = pd.DataFrame(list(stockData_ticker.info.items()))
    info_df.rename(columns={0:"Information", 1: "Data"}, inplace=True)

    #drops long business summary
    info_df = info_df.drop(info_df.index[3])

    infoTable = dash_table.DataTable(
        data=info_df.to_dict("records"), 
        columns=[{"name": i, "id": i} for i in info_df.columns],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '40px',
        },
        page_action='none',
        style_table={
            'height': '335px', 
        },
        style_cell={
            'textAlign': 'left',
        },
        tooltip_duration=None
    )
    return infoTable

def createRecommendationsTable(stockData_ticker):
    recommendations_df = pd.DataFrame(stockData_ticker.recommendations)

    try:
        recommendations_df["Date"]
    except:
        recommendations_df.reset_index(inplace=True)

    recommendations_df = recommendations_df[::-1]

    recommendationsTable = dash_table.DataTable(
        data=recommendations_df.to_dict("records"), 
        columns=[{"name": i, "id": i} for i in recommendations_df.columns],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        page_action='none',
        style_cell={
            'textAlign': 'left',
        },
        style_table={'height': '335px'}
    )
    return recommendationsTable

def createActionsTable(stockData_ticker):
    actions_df = pd.DataFrame(stockData_ticker.actions)   
    actions_df.reset_index(inplace=True)
    actions_df = actions_df[::-1]

    actionsTable = dash_table.DataTable(
        data=actions_df.to_dict("records"), 
        columns=[{"name": i, "id": i} for i in actions_df.columns],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '40px',
        },
        page_action='none',
        style_cell={
            'textAlign': 'left',
        },
        style_table={'height': '335px'}
    )
    return actionsTable

def createHoldersTable(stockData_ticker):
    holders_df = pd.DataFrame(stockData_ticker.institutional_holders)
    #actions_df.reset_index(inplace=True)
    #actions_df = actions_df[::-1]

    holdersTable = dash_table.DataTable(
        data=holders_df.to_dict("records"), 
        columns=[{"name": i, "id": i} for i in holders_df.columns],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        page_action='none',
        style_cell={
            'textAlign': 'left',
        },
        style_table={'height': '335px'} 
    )
    return holdersTable

def createSustainabilityTable(stockData_ticker):
    sustainability_df = pd.DataFrame(stockData_ticker.sustainability)

    if len(sustainability_df.columns) == 1:
        sustainability_df.reset_index(inplace=True)

    #sustainability_df.rename(columns={0:"Information", 1: "Data"}, inplace=True)

    sustainabilityTable = dash_table.DataTable(
        data=sustainability_df.to_dict("records"), 
        columns=[{"name": i, "id": i} for i in sustainability_df.columns],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '40px',
        },
        page_action='none',
        style_cell={
            'textAlign': 'left',
        },
        style_table={'height': '335px'} 
    )
    return sustainabilityTable

### ----- NEWS ----- ###
def createNewsList(stockCode):
    url = "https://www.marketwatch.com/investing/stock/" + stockCode + "?mod=quote_search"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, 'html.parser')
    headlines = soup.find('mw-scrollable-news-v2').find_all("h3")
    links = soup.find('mw-scrollable-news-v2').find_all("a", attrs={'href': re.compile("^https://")})
    providers = soup.find('mw-scrollable-news-v2').find_all("span", {"class": "article__author"})
    timestamps = soup.find('mw-scrollable-news-v2').find_all("span", {"class": "article__timestamp"})

    news = []

    for i in range(10):
        news.append(
            dbc.ListGroupItem(
                [
                    html.H5(html.A(headlines[i].text.strip(), href=links[i*2].get("href"), target="_blank")),
                    html.Small(timestamps[i].text.strip() + " " + providers[i].text.strip()),
                ]
            )
        )

    newsList = dbc.ListGroup(
        children = news,
    )

    return newsList

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
                            html.Span("Analyse", style={"color": "#F28E2B"})
                        ])
                    ], style={"textAlign": "center"})
                ], width={"size": 6, "offset": 3}),

                dbc.Col([
                    dcc.Loading(
                        id="loading",
                        type="circle", #default
                        color="#4E79A7",
                        children=[
                            html.Div([], style={"display": "none"}, id="loading-div"),
                            html.Div([], style={"display": "none"}, id="loading-div-2")
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
                                            {"label": "Apple Inc. (AAPL)", "value": "AAPL"},
                                            {"label": "Alphabet Inc. (GOOGL)", "value": "GOOGL"},
                                            {"label": "Microsoft Corporation (MSFT)", "value": "MSFT"},
                                            {"label": "Tesla, Inc. (TSLA)", "value": "TSLA"},
                                            {"label": "Advanced Micro Devices, Inc. (AMD)", "value": "AMD"},
                                            {"label": "NVIDIA Corporation (NVDA)", "value": "NVDA"},
                                            {"label": "Intel Corporation (INTC)", "value": "INTC"},
                                            {"label": "The Coca-Cola Company (KO)", "value": "KO"},
                                            {"label": "McDonald's Corporation (MCD)", "value": "MCD"},
                                            {"label": "Starbucks Corporation (SBUX)", "value": "SBUX"},
                                        ], 
                                        value="AAPL", 
                                        id="stock-select-dropdown"
                                    )
                                ], width=8),
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
                                html.P("Is it worth a buy?"),
                                html.H3("No", 
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
                                html.P("Expected chage:"),
                                html.H3("{:.2f}%".format(((forecastResults.iloc[0, 0] - stockData_df.iloc[-1, 3])/stockData_df.iloc[-1, 3])*100), 
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
                                html.P("Forecast model test MAPE:"),
                                html.H3("{:.2f}%".format(mape), 
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
                                        figure=createPriceSparklineFigure(stockData_df),
                                        config={"displayModeBar" : False},
                                        id="price-sparkline"
                                    )
                                ], width=6),
                                dbc.Col([
                                    dcc.Graph(
                                        figure=createVolumeSparklineFigure(stockData_df),
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
                        dcc.Link(dbc.Button("Compare", color="compare", className="me-1", size="lg"), href="/compare")
                    ])
                ], width=1, align="center")
            ], className="mb-3"),

            #Row4 - Forecast
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            #Row1 - Charts
                            dbc.Row([
                                dbc.Col([
                                    #html.H5("(^FTSE) Price Forecast"),
                                    dcc.Graph(
                                        figure=createForecastResultsLineFigure(stockData_df, forecastResults, "AAPL"),
                                        config={"displayModeBar" : False},
                                        id="forecast-results-line"
                                    )
                                ], width=4, style={"text-align": "center"}),
                                dbc.Col([
                                    #html.H5("(^FTSE) Model Test Results"),
                                    dcc.Graph(
                                        figure=createForecastTestLineFigure(outputs_test, outputs_pred, "AAPL"),
                                        config={"displayModeBar" : False},
                                        id="forecast-test-line"
                                    )
                                ], width=4, style={"text-align": "center"}),
                                dbc.Col([
                                    #html.H5("(^FTSE) Model Training Progress"),
                                    dcc.Graph(
                                        figure=createForecastTrainingLineFigure(history, "AAPL"),
                                        config={"displayModeBar" : False},
                                        id="forecast-training-line"
                                    )   
                                ], width=4, style={"text-align": "center"})
                            ], className="mb-2"),
                            #Row2 - Filters
                            dbc.Row([
                                dbc.Col([
                                    dbc.Row([
                                        dbc.Col([
                                            html.P("Input: ", style={'textAlign': 'right', "margin": "0px 0px 0px 0px"})
                                        ], width=7),
                                        dbc.Col([
                                            dcc.Input(
                                                type="number",
                                                value=20,
                                                style={'width': 80, "text-align": "center"},
                                                id="input-days-input"
                                            )
                                        ], width=5)
                                    ], align="center")
                                ], width=1),
                                dbc.Col([
                                    dbc.Row([
                                        dbc.Col([
                                            html.P("Output: ", style={'textAlign': 'right', "margin": "0px 0px 0px 0px"})
                                        ], width=7),
                                        dbc.Col([
                                            dcc.Input(
                                                type="number",
                                                value=10,
                                                style={'width': 80, "text-align": "center"},
                                                id="output-days-input"
                                            )
                                        ], width=5)
                                    ], align="center")
                                ], width=2),
                                dbc.Col([
                                    dcc.Dropdown(
                                        options = [
                                            {"label": "Open", "value": 0},
                                            {"label": "High", "value": 1},
                                            {"label": "Low", "value": 2},
                                            {"label": "Close", "value": 3},
                                            {"label": "Adj Close", "value": 4},
                                            {"label": "Volume", "value": 5}
                                        ],
                                        multi=True,
                                        value=[0,1,2,3,4,5],
                                        id="attributes-select-dropdown"
                                    )
                                ], width=5, className="attributes-dropdown"),
                                dbc.Col([
                                    dcc.Slider(10, 100, 10, value=50, id="epochs-slider")
                                ], width=4)                             
                            ], align="center")
                        ])
                    ])
                ], align="center")
            ], className="mb-3"),

            #Row 5 - Company statistics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dcc.Graph(
                                    figure=createPriceCandleFigure(stockData_df, "AAPL"),
                                    config={"displayModeBar" : False},
                                    id="price-candle"
                                )
                            ]),
                            dbc.Row([
                                dcc.Graph(
                                    figure=createVolumeBarFigure(stockData_df, "AAPL"),
                                    config={"displayModeBar" : False},
                                    id="volume-bar"                                   
                                )
                            ])
                        ])
                    ])
                ], width=7, style={"text-align": "center"}),
                dbc.Col([
                    dbc.Card([  
                        dbc.CardHeader(
                            dbc.Tabs(
                                [
                                    dbc.Tab(label="Stock Info", tab_id="stock-info-tab"),
                                    dbc.Tab(label="Recommendations", tab_id="recommendations-tab"),
                                    dbc.Tab(label="Actions", tab_id="actions-tab"),
                                    dbc.Tab(label="Holders", tab_id="holders-tab"),
                                    dbc.Tab(label="Sustainability", tab_id="sustainability-tab"),
                                ],
                                id="stock-stats-tabs",
                                active_tab="stock-info-tab",
                            )
                        ),  

                        dbc.CardBody([
                            #createInfoTable(stockData_ticker)
                        ], id="stock-stats-content", style={"overflow-y": "auto"}), 

                    ], className="pb-3 mb-3"),
                    dbc.Card([
                        dbc.CardBody([
                            html.Div(
                                createNewsList(stockCode),
                                style={"maxHeight": "284px"},
                                id="news-list"
                            ),
                        ], style={"overflow-x": "auto"})
                    ], className="pb-3")
                ], width=5)
            ]),

        ]),
        color="light",
        className="px-3"
    )
])

### ----- CALLBACKS ----- ###

#Change on stock select!
@callback(
    Output("suggestion-label", "children"),
    Output("suggestion-label", "style"),

    Output("change-label", "children"),
    Output("change-label", "style"),

    Output("mape-label", "children"),
    Output("mape-label", "style"),

    Output("price-sparkline", "figure"),
    Output("volume-sparkline", "figure"),

    Output("forecast-results-line", "figure"),
    Output("forecast-test-line", "figure"),
    Output("forecast-training-line", "figure"),

    Output("price-candle", "figure"),
    Output("volume-bar", "figure"),

    Output("stock-stats-tabs", "active_tab"),  
    Output("news-list", "children"),

    Output("loading-div", "children"),
    Output("error-modal", "is_open"),

    Input("stock-select-dropdown", "value"),
    Input("start-date-picker", "date"),
    Input("end-date-picker", "date"),

    Input("input-days-input", "value"),
    Input("output-days-input", "value"),
    Input("attributes-select-dropdown", "value"),
    Input("epochs-slider", "value")
)

def update_figures(selected_stock, selected_startDate, selected_endDate, inputDim, outputDim, attributes, epochs):

    try:
        startDate = dt.date.fromisoformat(selected_startDate).strftime("%Y-%m-%d")
        endDate = dt.date.fromisoformat(selected_endDate).strftime("%Y-%m-%d")

        stockData_df = downloadStockData(selected_stock, startDate, endDate)
        forecastResults, outputs_test, outputs_pred, rmse, mape, history = forecast.forecast(stockData_df, inputDim, outputDim, attributes, epochs)

        global stockCode
        if selected_stock != stockCode:
            global stockData_ticker
            stockData_ticker = yf.Ticker(selected_stock)
            stockCode = selected_stock

        activeTab = "stock-info-tab"
        newsList = createNewsList(selected_stock)

        expectedChange = ((forecastResults.iloc[-1, 0] - stockData_df.iloc[-1, 3])/stockData_df.iloc[-1, 3])*100
        if expectedChange - mape > 0:
            suggestionLabelChildren = "Yes"
            suggestionLabelStyle = {"color": "#4E79A7"}
        else:
            suggestionLabelChildren = "No"
            suggestionLabelStyle = {"color": "#F28E2B"}

        changeLabelChildren = "{:.2f}%".format(expectedChange)
        if expectedChange > 0:
            changeLabelStyle = {"color": "#4E79A7"}
        else:
            changeLabelStyle = {"color": "#F28E2B"}

        mapeLabelChildren = "{:.2f}%".format(mape)
        if mape < 5:
            mapeLabelStyle = {"color": "#4E79A7"}
        else:
            mapeLabelStyle = {"color": "#F28E2B"}

        priceSparkline = createPriceSparklineFigure(stockData_df)
        volumeSparkline = createVolumeSparklineFigure(stockData_df)

        forecastResultsLine = createForecastResultsLineFigure(stockData_df, forecastResults, selected_stock)
        forecastTestLine = createForecastTestLineFigure(outputs_test, outputs_pred, selected_stock)
        forecastTrainingLine = createForecastTrainingLineFigure(history, selected_stock)

        priceCandle = createPriceCandleFigure(stockData_df, selected_stock)
        volumeBar = createVolumeBarFigure(stockData_df, selected_stock)

        #For loading element
        loaded = "loaded"
        modalShow = False

        return suggestionLabelChildren, suggestionLabelStyle, changeLabelChildren, changeLabelStyle, mapeLabelChildren, \
            mapeLabelStyle, priceSparkline, volumeSparkline, forecastResultsLine, forecastTestLine, forecastTrainingLine, \
            priceCandle, volumeBar, activeTab, newsList, loaded, modalShow
    except Exception as e:
        modalShow = True

        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, \
            dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, modalShow

@callback(
    Output("stock-stats-content", "children"),
    Output("loading-div-2", "children"),

    Input("stock-stats-tabs", "active_tab"),
    Input("stock-select-dropdown", "value"),
)
def change_tab_content(activeTab, stockCode):

    if activeTab == "stock-info-tab":
        tabContents = createInfoTable(stockData_ticker)
    elif activeTab == "recommendations-tab":
        tabContents = createRecommendationsTable(stockData_ticker)
    elif activeTab == "actions-tab":
        tabContents = createActionsTable(stockData_ticker)
    elif activeTab == "holders-tab":
        tabContents = createHoldersTable(stockData_ticker)
    elif activeTab == "sustainability-tab":
        tabContents = createSustainabilityTable(stockData_ticker)

    loaded = "loaded"

    return tabContents, loaded
    