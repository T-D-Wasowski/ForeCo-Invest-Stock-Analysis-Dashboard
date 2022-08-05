import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from dashboards import analysis_dash, comparison_dash

import webbrowser

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
url = "http://127.0.0.1:8050/"

### ----- LAYOUT ----- ###

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

### ----- CALLBACKS ----- ###

#Change Page
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/compare":
        return comparison_dash.layout
    else:
        return analysis_dash.layout

### ----- Open Browser and Run server ----- ###

webbrowser.open(url, new=0, autoraise=True)

if __name__ == "__main__":
    app.run_server()

