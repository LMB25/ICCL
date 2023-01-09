import dash
import dash_bootstrap_components as dbc
import diskcache
from dash.long_callback import DiskcacheLongCallbackManager

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Set Dash Options
app = dash.Dash(
    external_stylesheets=[dbc.themes.LUX, dbc.icons.FONT_AWESOME], suppress_callback_exceptions=True, long_callback_manager=long_callback_manager
)
