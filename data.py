import os
import napari
import numpy as np
from pims import ND2Reader_SDK, pipeline
from pims_nd2 import ND2_Reader
from scipy.signal import medfilt2d
import multiprocessing
from functools import partial


class MicroscopyData(object):
    def __init__(self, data_path, data_name):
        self.data_path = data_path
        self.data_name = data_name
        self.full_path = os.path.join(self.data_path, self.data_name)
        self.feature_param_settings = (
            self.data_name[: self.data_name.rfind(".")] + "_feature_setting.pkl"
        )
        self.full_feature_param_settings = os.path.join(
            self.data_path, self.feature_param_settings
        )
        self.batch_features = (
            self.data_name[: self.data_name.rfind(".")] + "_features.h5"
        )
        self.full_batch_features = os.path.join(self.data_path, self.batch_features)
        self.ftype = self.data_name[self.data_name.rfind(".") + 1 :]
        self.traj_file_path = self.data_name[: self.data_name.rfind(".")] + "_traj.h5"
        self.full_traj_file_path = os.path.join(self.data_path, self.traj_file_path)
        self.track_frames_file = (
            self.data_name[: self.data_name.rfind(".")] + "_track_frames.npy"
        )
        self.full_track_frames_file = os.path.join(
            self.data_path, self.track_frames_file
        )
        self.mtoc_path = self.data_name[: self.data_name.rfind(".")] + "_mtoc.h5"
        self.full_mtoc_path = os.path.join(self.data_path, self.mtoc_path)

        """
        if os.path.exists(self.full_track_frames_file):
        """
        if self.ftype == "nd2":
            self.frames = ND2Reader_SDK(self.full_path)
            self.timpoints = {
                i: self.frames[i].metadata["t_ms"] / 1e3
                for i in range(len(self.frames))
            }
            self.mpp = self.frames[0].metadata["mpp"]
            self.frames.iter_axes = "t"
            self.frames.bundle_axes = "cyx"
        else:
            raise ValueError("Following file types are supported: nd2")

    def generate_track_frames(self, track_ch, parallel=12, kernel_size=3):
        process_frame = partial(preprocess, ch=track_ch, kernel_size=kernel_size)
        with multiprocessing.Pool(processes=parallel) as pool:
            results = pool.map(process_frame, self.frames)
        self.track_frames = np.stack(results, axis=0)
        np.save(self.full_track_frames_file, self.track_frames)

    def fetch_track_data(self):
        self.track_frames = np.load(self.full_track_frames_file)

    def display(self, c: int = None, t: int = None, run: bool = True):
        self.frames.iter_axes = "c"
        self.frames.bundle_axes = "tyx"
        if c is None and t is None:
            for c in range(self.frames.sizes["c"]):
                if c == 0:
                    viewer = napari.view_image(self.frames[c], name="Channel %d" % c)
                else:
                    viewer.add_image(self.frames[c], name="Channel %d" % c)

        elif c is not None and t is None:
            viewer = napari.view_image(self.frames[c], name="Channel %d" % c)

        elif c is None and t is not None:
            for c in range(self.frames.sizes["c"], name="Channel %d" % c):
                if c == 0:
                    viewer = napari.view_image(self.frames[c])[t, :, :]
                else:
                    viewer.add_image(self.frames[c][t, :, :])

        elif c is not None and t is not None:
            viewer = napari.view_image(self.frames[c][t, :, :], name="Channel %d" % c)
        if run:
            napari.run()
        return viewer

    def reset_axes(self):
        self.frames.iter_axes = "t"
        self.frames.bundle_axes = "cyx"


def preprocess(im, ch, kernel_size):
    return medfilt2d(im[ch, :, :], kernel_size=kernel_size)


def main():
    data = MicroscopyData("../data/220930", "220930_1um-beads.nd2")
    # data.display(1)
    data.generate_track_frames(track_ch=1)


if __name__ == "__main__":
    main()
