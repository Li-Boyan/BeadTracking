#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Particle tracking pipeline
pipeline.py: process the raw data of NGS
Boyan Li

Usage:
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --display
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --initialize
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --test
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --feature-detect
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --track [--search-range=<int>] [--memory=<int>] [--appear=<float>]
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --mtoc [--mtoc-ch=<int>] [--mtoc-num=<int>] [--thresh=<float>] [--frame-range=<int>]

Options:
    -h --help                               show this screen.
    --date=<str>                            date of the experiment.
    --file=<str>                            name of the data file.
    --track-channel=<int>                   channel to perform tracking.
    --test                                  test parameters interactively.
    --feature-detect                        run only feature detection.
    --track                                 run only tracking.
    --initialize                            initialize data.
    --display                               display images.
    --search-range=<int>                    search range for tracking.
    --memory=<int>                          memory for tracking.
    --appear=<float>                        cutoff for appearance in frames.
    --mtoc                                  track mtoc.
    --mtoc-ch=<int>                         mtoc channel.
    --frame-range=<int>                     range of frames for mtoc tracking.
"""

import os
from data import MicroscopyData
from feature_detection import test_params, locate_batch
from track import particleTracker, mtoc_track
from docopt import docopt


def run_pipeline():
    args = docopt(__doc__)
    data_path = os.path.join("../data", args["--date"])
    data_name = args["--file"]
    track_ch = int(args["--track-channel"])
    raw_data = MicroscopyData(data_path, data_name)
    if args["--initialize"]:
        raw_data.generate_track_frames(track_ch=track_ch)
    elif args["--display"]:
        raw_data.display()
    elif args["--test"]:
        raw_data.fetch_track_data()
        app = test_params(raw_data)
        app.run_server(debug=True)
    elif args["--feature-detect"]:
        raw_data.fetch_track_data()
        locate_batch(raw_data)
    elif args["--track"]:
        if not args["--search-range"]:
            search_range = 10
        else:
            search_range = int(args["--search-range"])
        if not args["--memory"]:
            memory = 5
        else:
            memory = int(args["--memory"])
        if not args["--appear"]:
            appear_cutoff = 0.4
        else:
            appear_cutoff = float(args["--appear"])
        tracker = particleTracker(raw_data, load_traj=False)
        tracker.run_tracking(memory, 100, search_range)  # , frame_range=[-1, 224])
        tracker.filter_traj(appear_cutoff)
        tracker.visualize_traj(1, run=True)
    elif args["--mtoc"]:
        if not args["--thresh"]:
            thresh = 3.0
        else:
            thresh = float(args["--thresh"])
        if not args["--frame-range"]:
            frame_range = None
        else:
            frame_range = int(args["--frame-range"])
        mtoc_track(
            raw_data,
            mtoc_ch=int(args["--mtoc-ch"]),
            n_centrosome=int(args["--mtoc-num"]),
            thresh=thresh,
            frame_range=frame_range
        )


if __name__ == "__main__":
    run_pipeline()
