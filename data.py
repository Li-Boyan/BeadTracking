import os
import napari
import numpy as np
from pims import ND2Reader_SDK, pipeline
from pims_nd2 import ND2_Reader
from scipy.signal import medfilt2d


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
            self.data_name[: self.data_name.rfind(".")] + "_features.csv"
        )
        self.full_batch_features = os.path.join(self.data_path, self.batch_features)
        self.ftype = self.data_name[self.data_name.rfind(".") + 1 :]
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

    def generate_track_frames(self, track_ch):
        self.track_frames = self._median(track_ch)

    def _median(self, c, filter_size=3):
        median_pip = pipeline(lambda im: medfilt2d(im[c, :, :], filter_size))
        return median_pip(self.frames)

    def display(self, c: int = None, t: int = None):
        self.frames.iter_axes = "c"
        self.frames.bundle_axes = "tyx"
        if c is None and t is None:
            for c in range(self.frames.sizes["c"]):
                if c == 0:
                    viewer = napari.view_image(self.frames[c], name="Channel %d" % c)
                else:
                    viewer.add_image(self.frames[c], name="Channel %d" % c)

        elif c is not None and t is None:
            napari.view_image(self.frames[c], name="Channel %d" % c)

        elif c is None and t is not None:
            for c in range(self.frames.sizes["c"], name="Channel %d" % c):
                if c == 0:
                    viewer = napari.view_image(self.frames[c])[t, :, :]
                else:
                    viewer.add_image(self.frames[c][t, :, :])

        elif c is not None and t is not None:
            napari.view_image(self.frames[c][t, :, :], name="Channel %d" % c)

        napari.run()


def main():
    data = MicroscopyData("../data/220930", "220930_0.2um-beads003.nd2")
    # data.display(c=0)
    frames_median = data._median(0)


if __name__ == "__main__":
    main()
