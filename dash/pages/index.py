from dash import html

import dash_bootstrap_components as dbc
from components.uploadForm import UploadForm

index_layout = html.Div([
    dbc.Container([
        dbc.Row([
            UploadForm,
        ])
    ],  className="responsive-container"),
])
