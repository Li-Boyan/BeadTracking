import os
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import trackpy as tp
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
from data import MicroscopyData
from tqdm import tqdm
import pickle


def test_params(dataset):
    track_frames = dataset.track_frames
    app = Dash(__name__)
    param_div = html.Div(
        [
            html.H1("Parameters"),
            ### x and y slider ###
            html.Label("Brightness range"),
            dcc.RangeSlider(
                0,
                track_frames[0].max(),
                1,
                value=[0, track_frames[0].max()],
                marks=None,
                id="brightness-rangeslider",
            ),
            html.Label("x0"),
            dcc.Slider(
                0,
                track_frames[0].shape[0],
                1,
                value=1000,
                marks=None,
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
                marks=None,
                id="y-slider",
                tooltip={"always_visible": True},
            ),
            html.Br(),
            ######
            ### window size slider ###
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
            ######
            ### Feature size slider ###
            html.Label("Feature size"),
            dcc.Slider(
                3,
                15,
                value=3,
                step=2,
                id="slider",
                tooltip={"always_visible": True},
            ),
            html.Br(),
            ######
            ### Separation distance slider ###
            html.Label("Separation"),
            dcc.Slider(
                -5,
                5,
                1,
                value=0,
                id="separation-slider",
                tooltip={"always_visible": True},
            ),
            html.Br(),
            html.Br(),
            ######
            ### Text to display ###
            html.Label("Feature to display"),
            dcc.Dropdown(
                ["none", "mass", "size", "ecc", "signal", "raw_mass", "ep"],
                "none",
                id="textlabel-dropdown",
            ),
        ],
        style={"display": "inline-block", "width": "30%", "height": 800, "margin": 0},
    )

    image_div = html.Div(
        [dcc.Graph(id="image")],
        style={
            "display": "incline-block",
            "width": "70%",
            "float": "right",
            "height": 800,
            "margin": 0,
        },
    )

    param_img_div = html.Div([param_div, image_div])

    statistics_div_1 = html.Div(
        [dcc.Graph(id="statistics")],
        style={"margin": 0, "width": "100%", "height": 600, "display": "inline-block"},
    )

    param_range_children = []
    for i, col in enumerate(["mass", "size", "ecc", "signal", "raw_mass", "ep"]):
        param_range_children.append(
            html.Div(
                [
                    html.Div(html.Label(col), style={"display": "inline-block"}),
                    html.Div(
                        dcc.Input(
                            id="param%d-min" % (i + 1),
                            type="number",
                            placeholder="",
                            debounce=True,
                        ),
                        style={"width": "20%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.Input(
                            id="param%d-max" % (i + 1),
                            type="number",
                            placeholder="",
                            debounce=True,
                        ),
                        style={"width": "20%", "display": "inline-block"},
                    ),
                    html.Div(
                        dcc.RadioItems(
                            ["linear", "lg"],
                            "linear",
                            id="xaxis-type-%d" % (i + 1),
                            inline=True,
                        ),
                        style={"width": "20%", "display": "inline-block"},
                    ),
                    html.Br(),
                ]
            )
        )
    param_range_children += [
        html.Button("Submit", n_clicks=0, id="param-set-button"),
        dcc.Graph(id="statistics-2"),
    ]
    param_range_div = html.Div(param_range_children)

    statistics_div_2 = html.Div(
        [
            html.Div(
                [
                    html.Label("x"),
                    dcc.Dropdown(
                        ["mass", "size", "ecc", "signal", "raw_mass", "ep"],
                        "signal",
                        id="col1-dropdown",
                    ),
                    dcc.RadioItems(
                        ["linear", "lg"],
                        "linear",
                        id="scatter-xaxis-type",
                        inline=True,
                    ),
                ],
                style={"width": "20%", "display": "inline-block"},
            ),
            html.Div(
                [
                    html.Label("y"),
                    dcc.Dropdown(
                        ["mass", "size", "ecc", "signal", "raw_mass", "ep"],
                        "signal",
                        id="col2-dropdown",
                    ),
                    dcc.RadioItems(
                        ["linear", "lg"],
                        "linear",
                        id="scatter-yaxis-type",
                        inline=True,
                    ),
                ],
                style={"width": "24%", "display": "inline-block"},
            ),
            dcc.Graph(id="statistics-3"),
        ],
        style={"width": "100%", "height": 600, "display": "inline-block", "margin": 0},
    )

    app.layout = html.Div(
        [
            param_img_div,
            param_range_div,
            statistics_div_1,
            statistics_div_2,
        ]
    )

    @app.callback(
        Output("image", "figure"),
        Output("statistics", "figure"),
        Output("statistics-2", "figure"),
        Output("statistics-3", "figure"),
        Input("brightness-rangeslider", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("window-slider", "value"),
        Input("slider", "value"),
        Input("separation-slider", "value"),
        Input("textlabel-dropdown", "value"),
        Input("xaxis-type-1", "value"),
        Input("xaxis-type-2", "value"),
        Input("xaxis-type-3", "value"),
        Input("xaxis-type-4", "value"),
        Input("xaxis-type-5", "value"),
        Input("xaxis-type-6", "value"),
        Input("col1-dropdown", "value"),
        Input("col2-dropdown", "value"),
        Input("scatter-xaxis-type", "value"),
        Input("scatter-yaxis-type", "value"),
        State("param1-min", "value"),
        State("param1-max", "value"),
        State("param2-min", "value"),
        State("param2-max", "value"),
        State("param3-min", "value"),
        State("param3-max", "value"),
        State("param4-min", "value"),
        State("param4-max", "value"),
        State("param5-min", "value"),
        State("param5-max", "value"),
        State("param6-min", "value"),
        State("param6-max", "value"),
        Input("param-set-button", "n_clicks"),
    )
    def update_figure_size_determination(
        brightness_range,
        x0,
        y0,
        window,
        d,
        sep,
        text_col,
        xt1,
        xt2,
        xt3,
        xt4,
        xt5,
        xt6,
        scatter_col1,
        scatter_col2,
        scatter_xt,
        scatter_yt,
        param1_min,
        param1_max,
        param2_min,
        param2_max,
        param3_min,
        param3_max,
        param4_min,
        param4_max,
        param5_min,
        param5_max,
        param6_min,
        param6_max,
        n_clicks,
    ):
        # Update features
        foi = track_frames[0][x0 : x0 + window, y0 : y0 + window]
        feature_df = tp.locate(
            foi,
            d,
            invert=False,
            minmass=param1_min,
            maxsize=param2_max,
            separation=d + sep,
            engine="numba",
        )
        print(param6_min, param6_max)
        param_settings = [
            d,
            sep,
            [param1_min, param1_max],
            [param2_min, param2_max],
            [param3_min, param3_max],
            [param4_min, param4_max],
            [param5_min, param5_max],
            [param6_min, param6_max],
        ]
        for i in range(2, 8):
            if param_settings[i][0] is None:
                param_settings[i][0] = 0
            if param_settings[i][1] is None:
                param_settings[i][1] = np.inf

        feature_df = feature_df[
            (
                (feature_df.mass <= param_settings[2][1])
                & (feature_df.size >= param_settings[3][0])
                & (feature_df.ecc >= param_settings[4][0])
                & (feature_df.ecc <= param_settings[4][1])
                & (feature_df.signal >= param_settings[5][0])
                & (feature_df.signal <= param_settings[5][1])
                & (feature_df.raw_mass >= param_settings[6][0])
                & (feature_df.raw_mass <= param_settings[6][1])
                & (feature_df.ep >= param_settings[7][0])
                & (feature_df.ep <= param_settings[7][1])
            )
        ]

        # Image
        fig1 = px.imshow(
            foi, aspect="equal", zmin=brightness_range[0], zmax=brightness_range[1]
        )
        if text_col == "none":
            fig1.add_trace(
                go.Scatter(
                    x=feature_df.x,
                    y=feature_df.y,
                    mode="markers",
                    marker={
                        "color": "rgba(255, 255, 255, 0)",
                        "line": {
                            "color": "red",
                            "width": 1,
                        },
                        "size": d * 2,
                    },
                ),
            )
        else:
            fig1.add_trace(
                go.Scatter(
                    x=feature_df.x,
                    y=feature_df.y,
                    text=["%.4g" % num for num in feature_df[text_col]],
                    mode="markers+text",
                    marker={
                        "color": "rgba(255, 255, 255, 0)",
                        "line": {
                            "color": "red",
                            "width": 1,
                        },
                        "size": d * 2,
                    },
                    textfont=dict(family="Arial", size=14, color="white"),
                ),
            )
        fig1.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        fig1.update_xaxes(row=1, col=1)
        fig1.update_layout(
            height=700, width=700, transition_duration=500, showlegend=False
        )
        fig1.update_coloraxes(showscale=False)

        # Statistics 1 - subpixel bias
        fig2 = make_subplots(1, 2)
        fig2.add_trace(
            trace=go.Histogram(x=feature_df.x.apply(lambda x: x % 1)), row=1, col=1
        )
        fig2.add_trace(
            trace=go.Histogram(x=feature_df.y.apply(lambda x: x % 1)), row=1, col=2
        )
        fig2.update_coloraxes(showscale=True)
        fig2.update_layout(
            height=400, width=1000, transition_duration=500, showlegend=False
        )
        fig2.update_xaxes(title_text="Mass", row=1, col=1)
        fig2.update_xaxes(title_text="x subpixel bias", row=1, col=2)
        fig2.update_xaxes(title_text="y subpixel bias", row=1, col=3)

        # Statistics 2
        fig3 = make_subplots(1, 6)
        for i, (axis_type, col) in enumerate(
            zip(
                [xt1, xt2, xt3, xt4, xt5, xt6],
                ["mass", "size", "ecc", "signal", "raw_mass", "ep"],
            )
        ):
            if axis_type == "linear":
                fig3.add_trace(trace=go.Histogram(x=feature_df[col]), row=1, col=i + 1)
            elif axis_type == "lg":
                fig3.add_trace(
                    trace=go.Histogram(
                        x=np.log10(feature_df[feature_df[col] > 0][col])
                    ),
                    row=1,
                    col=i + 1,
                )
            fig3.update_xaxes(title_text=col, row=1, col=i + 1)
        fig3.update_layout(
            height=400, width=1000, transition_duration=500, showlegend=False
        )

        # Statistics 3
        scatter_x_data = feature_df[
            (feature_df[scatter_col1] > 0) & (feature_df[scatter_col2] > 0)
        ][scatter_col1]
        scatter_y_data = feature_df[
            (feature_df[scatter_col1] > 0) & (feature_df[scatter_col2] > 0)
        ][scatter_col2]
        if scatter_xt == "lg":
            scatter_x_data = np.log10(scatter_x_data)
        if scatter_yt == "lg":
            scatter_y_data = np.log10(scatter_y_data)
        fig4 = px.scatter(x=scatter_x_data, y=scatter_y_data)
        fig4.update_layout(
            height=600, width=600, transition_duration=500, showlegend=False
        )

        if n_clicks > 0:
            with open(dataset.full_feature_param_settings, "wb") as f:
                pickle.dump(param_settings, f)

        return fig1, fig2, fig3, fig4

    return app


def run_batch(dataset):
    with open(dataset.full_feature_param_settings, "rb") as f:
        settings = pickle.load(f)
    print(settings)
    features = tp.batch(
        dataset.track_frames,
        settings[0],
        minmass=settings[2][0],
        maxsize=settings[3][1],
        separation=settings[0] + settings[1],
        engine="numba",
        processes=16,
    )
    features = features[
        (
            (features.mass <= settings[2][1])
            & (features.size >= settings[3][0])
            & (features.ecc >= settings[4][0])
            & (features.ecc <= settings[4][1])
            & (features.signal >= settings[5][0])
            & (features.signal <= settings[5][1])
            & (features.raw_mass >= settings[6][0])
            & (features.raw_mass <= settings[6][1])
            & (features.ep >= settings[7][0])
            & (features.ep <= settings[7][1])
        )
    ]
    features.to_csv(dataset.full_batch_features, index=False)


def main():
    raw_data = MicroscopyData("../data/220930", "220930_0.2um-beads003.nd2")
    raw_data.generate_track_frames(1)
    app = test_feature_size(raw_data)
    app.run_server(debug=True)


if __name__ == "__main__":
    main()
