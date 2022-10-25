#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Particle tracking pipeline
pipeline.py: process the raw data of NGS
Boyan Li

Usage:
    pipeline.py --date=<str> --file=<str> --track-channel=<int> [--test]

Options:
    -h --help                               show this screen.
    --date=<str>                            date of the experiment.
    --file=<str>                            name of the data file.
    --track-channel=<int>                   channel to perform tracking.
    --test                                  test parameters interactively.
"""

import os
from data import MicroscopyData
from feature_detection import test_params, run_batch
from docopt import docopt


def run_pipeline():
    args = docopt(__doc__)
    data_path = os.path.join("../data", args["--date"])
    data_name = args["--file"]
    track_ch = int(args["--track-channel"])
    raw_data = MicroscopyData(data_path, data_name)
    raw_data.generate_track_frames(track_ch)
    if args["--test"]:
        app = test_params(raw_data)
        app.run_server(debug=True)
    else:
        run_batch(raw_data)


if __name__ == "__main__":
    run_pipeline()
