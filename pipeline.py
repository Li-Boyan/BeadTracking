#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Particle tracking pipeline
pipeline.py: process the raw data of NGS
Boyan Li

Usage:
    pipeline.py --date=<str> --file=<str> --display [--until=<int>]
    pipeline.py --date=<str> --file=<str> --track-channel=<int> --initialize [--parallel=<int>] [--kernel-sz]
    pipeline.py --date=<str> --file=<str> --test
    pipeline.py --date=<str> --file=<str> --feature-detect [--parallel=<int>]
    pipeline.py --date=<str> --file=<str> --track [--search-range=<int>] [--memory=<int>] [--appear=<float>] [--frame-start=<int>] [--frame-end=<int>]
    pipeline.py --date=<str> --file=<str> --mtoc [--mtoc-ch=<int>] [--mtoc-num=<int>] [--thresh=<float>] [--frame-range=<int>] [--parallel=<int>]
    pipeline.py --date=<str> --file=<str> --mtoc-neighbor [--mtoc-ch=<int>] [--mtoc-num=<int>] [--search-range=<int>] [--init-thresh=<float>] [--thresh=<float>] [--frame-range=<int>]

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
    --until=<int>                           until the frames.
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
from track import particleTracker, mtoc_track, mtoc_track_neighbor
from docopt import docopt
import numpy as np


def run_pipeline():
    args = docopt(__doc__)
    data_path = os.path.join("../data", args["--date"])
    data_name = args["--file"]
    raw_data = MicroscopyData(data_path, data_name)
    if args["--initialize"]:
        track_ch = int(args["--track-channel"])
        parallel = int(args["--parallel"]) if args["--parallel"] else 12
        kernel_size = int(args["--kernel-sz"]) if args["--kernel-sz"] else 3
        raw_data.generate_track_frames(
            track_ch=track_ch, parallel=parallel, kernel_size=kernel_size
        )
    elif args["--display"]:
        if args["--until"] is not None:
            raw_data.display(frame_end=int(args["--until"]))
        else:
            raw_data.display()
    elif args["--test"]:
        raw_data.fetch_track_data()
        app = test_params(raw_data)
        app.run_server(debug=True)
    elif args["--feature-detect"]:
        raw_data.fetch_track_data()
        parallel = int(args["--parallel"]) if args["--parallel"] else 4
        locate_batch(raw_data, parallel)
    elif args["--track"]:
        search_range = int(args["--search-range"]) if args["--search-range"] else 10
        memory = int(args["--memory"]) if args["--memory"] else 5
        appear_cutoff = float(args["--appear"]) if args["--appear"] else 0.4
        frame_start = int(args["--frame-start"]) if args["--frame-start"] else -1
        frame_end = int(args["--frame-end"]) if args["--frame-end"] else np.inf
        tracker = particleTracker(raw_data, load_traj=False)
        tracker.run_tracking(
            memory, 100, search_range, frame_range=[frame_start, frame_end]
        )
        tracker.filter_traj(appear_cutoff)
        tracker.visualize_traj(1, run=True)
    elif args["--mtoc"]:
        thresh = float(args["--thresh"]) if args["--thresh"] else 3.0
        frame_range = int(args["--frame-range"]) if args["--frame-range"] else None
        parallel = int(args["--parallel"]) if args["--parallel"] else 4
        mtoc_track(
            raw_data,
            mtoc_ch=int(args["--mtoc-ch"]),
            n_mtoc=int(args["--mtoc-num"]),
            thresh=thresh,
            frame_range=frame_range,
            parallel=parallel,
        )
    elif args["--mtoc-neighbor"]:
        init_thresh = float(args["--init-thresh"]) if args["--init-thresh"] else 3.0
        thresh = float(args["--thresh"]) if args["--thresh"] else 1.0
        frame_range = int(args["--frame-range"]) if args["--frame-range"] else None
        search_range = int(args["--search-range"]) if args["--search-range"] else 10
        mtoc_track_neighbor(
            raw_data,
            mtoc_ch=int(args["--mtoc-ch"]),
            n_mtoc=int(args["--mtoc-num"]),
            search_range=search_range,
            initial_thresh=init_thresh,
            thresh=thresh,
            frame_range=frame_range,
        )


if __name__ == "__main__":
    run_pipeline()
