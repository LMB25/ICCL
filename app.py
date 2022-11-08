import dash
import dash_bootstrap_components as dbc

# Set Dash Options
app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True
)
