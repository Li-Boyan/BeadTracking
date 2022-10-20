from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import trackpy as tp
import numpy as np
from dash import Dash, dcc, html, Input, Output
from data import MicroscopyData


def test_feature_size(track_frames):
    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.Label("x0"),
            dcc.Slider(
                0,
                track_frames[0].shape[0],
                1,
                value=1000,
                id="x-slider",
                tooltip={"always_visible": True},
            ),
            html.Br(),
            html.Label("y0"),
            dcc.Slider(
                0,
                track_frames[0].shape[0],
                1,
                value=1000,
                id="y-slider",
                tooltip={"always_visible": True},
            ),
            html.Br(),
            html.Label("Window size"),
            dcc.Slider(
                0,
                400,
                50,
                value=100,
                id="window-slider",
                tooltip={"always_visible": True},
            ),
            html.Br(),
            html.Label("Feature size"),
            dcc.Slider(
                3, 15, value=3, step=2, id="slider", tooltip={"always_visible": True}
            ),
            html.Br(),
            html.Label("Separation"),
            dcc.Slider(-5, 5, 1, value=0, id="separation-slider", tooltip={"always_visible": True}),
            html.Br(),
            html.Label("Mass cutoff (log)"),
            dcc.Input(id="minmass-input", type="number", placeholder="", style={'marginRight':'10px'}),
            dcc.Graph(id="graph-with-slider"),
        ]
    )

    @app.callback(
        Output("graph-with-slider", "figure"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("window-slider", "value"),
        Input("slider", "value"),
        Input("separation-slider", "value"),
        Input("minmass-input", "value")
    )
    def update_figure_size_determination(x0, y0, window, d, sep, minmass=None):
        foi = track_frames[0][x0 : x0 + window, y0 : y0 + window]
        if minmass is not None:
            minmass = np.exp(minmass)
        feature_df = tp.locate(foi, d, invert=False, minmass=minmass, separation=d+sep)
        fig = make_subplots(1, 4, column_widths=[0.4, 0.3, 0.15, 0.15])
        fig.add_trace(trace=go.Heatmap(z=foi), row=1, col=1)
        fig.add_trace(
            row=1,
            col=1,
            trace=go.Scatter(
                x=feature_df.x,
                y=feature_df.y,
                mode="markers",
                marker={
                    "color": "rgba(255, 255, 255, 0)",
                    "line": {
                        "color": "white",
                        "width": 1,
                    },
                    "size": d * 1.5,
                },
            ),
        )
        fig.add_trace(trace=go.Histogram(x=np.log(feature_df.mass)), row=1, col=2)
        fig.add_trace(
            trace=go.Histogram(x=feature_df.x.apply(lambda x: x % 1)), row=1, col=3
        )
        fig.add_trace(
            trace=go.Histogram(x=feature_df.y.apply(lambda x: x % 1)), row=1, col=4
        )
        fig.update_coloraxes(showscale=True)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_layout(
            height=600, width=1500, transition_duration=500, showlegend=False
        )
        return fig

    return app

def main():
    raw_data = MicroscopyData("../data/220930", "220930_0.2um-beads003.nd2")
    track_ch = 1
    track_frames = raw_data._median(track_ch)
    app = test_feature_size(track_frames)
    app.run_server(debug=True)

if __name__ == "__main__":
    main()
