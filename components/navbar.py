# Import necessary libraries
from dash import html
import dash_bootstrap_components as dbc


# Define the navbar structure
def Navbar():

    layout = html.Div([
        dbc.NavbarSimple([
                        dbc.NavItem(dbc.NavLink(html.I(className="far fa-question-circle"),href="/page-4/1")),
                         ],
                brand="RWTH Aachen",
                brand_href="#",
                fluid=True,
                style={"height":40}) 
    ])

    return layout