
# Import necessary libraries
from dash import html
import dash_bootstrap_components as dbc


# Define the sidebar structure
def Sidebar():

    # sidebar style
    SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 50,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
    }

    # submenus
    submenu_1 = [
    html.Li(
        dbc.Row(
            [
                dbc.Col("Data Management"),
                dbc.Col(
                    html.I(className="fas fa-chevron-right me-3"),
                    width="auto",
                ),
            ],
            className="my-1",
        ),
        style={"cursor": "pointer"},
        id="submenu-1",
    ),
    dbc.Collapse(
        [
            dbc.NavLink("Import Data", href="/page-1/1", className="page-link"),
            dbc.NavLink("View OCEL", href="/page-1/2", className="page-link"),
        ],
        id="submenu-1-collapse",
        is_open=True
    ),
    ]

    submenu_2 = [
        html.Li(
            dbc.Row(
                [
                    dbc.Col("Configuration"),
                    dbc.Col(
                        html.I(className="fas fa-chevron-right me-3"),
                        width="auto",
                    ),
                ],
                className="my-1",
            ),
            style={"cursor": "pointer"},
            id="submenu-2",
        ),
        dbc.Collapse(
            [
                dbc.NavLink("Clustering", href="/page-2/1", className="page-link"),
            ],
            id="submenu-2-collapse",
            is_open=True
        ),
        dbc.Collapse(
            [
                dbc.NavLink("Automatic Clustering", href="/page-2/2", className="page-link"),
            ],
            id="submenu-2-2-collapse",
            is_open=True
        ),
    ]

    submenu_3 = [
        html.Li(
            dbc.Row(
                [
                    dbc.Col("Process Model"),
                    dbc.Col(
                        html.I(className="fas fa-chevron-right me-3"),
                        width="auto",
                    ),
                ],
                className="my-1",
            ),
            style={"cursor": "pointer"},
            id="submenu-3",
        ),
        dbc.Collapse(
            [
                dbc.NavLink("Original OCEL", href="/page-3/1", className="page-link"),
                dbc.NavLink("Clustered OCEL", href="/page-3/2", className="page-link"),
            ],
            id="submenu-3-collapse",
            is_open=True,
        ),
    ]


    # layout
    layout = html.Div(
    [
        html.H2("ICCL", className="display-4"),
        html.Hr(),
        html.P(
            "Improve Comprehensibility With Clustering Tool", className="lead"
        ),
        dbc.Nav(submenu_1 + submenu_2 + submenu_3, vertical=True),
    ],
    style=SIDEBAR_STYLE,
    id="sidebar",
    )

    return layout

