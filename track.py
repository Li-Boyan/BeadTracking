import os
import trackpy as tp
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import numpy as np
from data import MicroscopyData
import scipy.ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from tqdm import tqdm
import pandas as pd
from itertools import product
import napari
import multiprocessing
from functools import partial


class particleTracker(object):
    def __init__(self, dataset: MicroscopyData, load_traj=True):
        self.dataset = dataset
        if load_traj:
            if not os.path.exists(self.dataset.full_traj_file_path):
                print(self.dataset.full_traj_file_path)
                raise FileNotFoundError("No trajectory file found!")
            else:
                self.traj_df = pd.read_hdf(self.dataset.full_traj_file_path, key="traj")

    def run_tracking(self, memory, subnet_size, search_range, frame_range=[-1, np.inf]):
        feature_df_batch = pd.read_hdf(self.dataset.full_batch_features, key="features")
        tp.linking.Linker.MAX_SUB_NET_SIZE = subnet_size
        print("Perform tracking ...", end="")
        self.traj_df = tp.link(
            feature_df_batch[
                (feature_df_batch.frame > frame_range[0])
                & (feature_df_batch.frame < frame_range[1])
            ],
            search_range,
            memory=memory,
        )
        print("Done.")
        self.traj_df.to_hdf(self.dataset.full_traj_file_path, key="traj")

    def filter_traj(
        self,
        appear_cutoff=0.5,
        param_range={},
        show_dist=False,
        sample=1000,
        cols=["mass", "size", "ecc", "signal", "raw_mass", "ep"],
    ):
        print("Before filtering: %d trajectories" % self.traj_df.particle.nunique())
        self.traj_df = tp.filter_stubs(
            self.traj_df, int(len(self.dataset.frames) * appear_cutoff)
        )
        if show_dist:
            scatter_matrix(
                self.traj_df.groupby("particle").mean()[cols], figsize=(15, 15)
            )
            plt.show()
        for k, v in param_range.items():
            self.traj_df = self.traj_df[
                (self.traj_df[k] > v[0]) & (self.traj_df[k] < v[1])
            ]
        print("After filtering: %d trajectories" % self.traj_df.particle.nunique())
        self.traj_df.to_hdf(self.dataset.full_traj_file_path, key="traj")

    def visualize_traj(self, ch, run=True):
        traj = list(
            zip(
                self.traj_df.particle,
                self.traj_df.frame,
                self.traj_df.y,
                self.traj_df.x,
            )
        )
        viewer = self.dataset.display(run=False)
        viewer.add_tracks(traj, name="tracks")
        if run:
            napari.run()
        return viewer


def mtoc_track(
    dataset,
    mtoc_ch,
    n_mtoc,
    thresh=3,
    display=True,
    parallel=14,
    frame_range=None,
):
    dataset.frames.iter_axes = "t"
    dataset.frames.bundle_axes = "cyx"
    centrosome_df = pd.DataFrame(
        columns=["centrosome_%d_%s" % (i, ax) for i, ax in product(range(n_mtoc), "yx")]
    )
    locate = partial(
        locate_by_threshold,
        kernel_size=3,
        thresh=thresh,
        n_mtoc=n_mtoc,
        ch=mtoc_ch,
    )
    with multiprocessing.Pool(processes=parallel) as pool:
        if frame_range is None:
            results = pool.map(locate, dataset.frames)
        else:
            results = pool.map(locate, dataset.frames[:frame_range])
    centrosome_df = pd.concat([centrosome_df, pd.DataFrame.from_records(results)])
    centrosome_df.reset_index(drop=True, inplace=True)
    centrosome_locs = [
        (
            centrosome_df.loc[0, "centrosome_%d_x" % cen],
            centrosome_df.loc[0, "centrosome_%d_y" % cen],
        )
        for cen in range(n_mtoc)
    ]
    centrosome_df_corr = centrosome_df.copy()
    for i in range(len(centrosome_df)):
        for cen in range(n_mtoc):
            x, y = (
                centrosome_df.loc[i, "centrosome_%d_x" % cen],
                centrosome_df.loc[i, "centrosome_%d_y" % cen],
            )
            correct_cen = np.argmin(
                [(x - x0) ** 2 + (y - y0) ** 2 for x0, y0 in centrosome_locs]
            )
            centrosome_df_corr.loc[i, "centrosome_%d_x" % correct_cen] = x
            centrosome_df_corr.loc[i, "centrosome_%d_y" % correct_cen] = y
    print(centrosome_df.head())
    centrosome_df.to_hdf(dataset.full_mtoc_path, key="mtoc")
    if display:
        dataset.frames.iter_axes = "c"
        dataset.frames.bundle_axes = "tyx"
        if frame_range is None:
            viewer = napari.view_image(dataset.frames[mtoc_ch])
        else:
            viewer = napari.view_image(dataset.frames[mtoc_ch][:frame_range])

        for i in range(len(centrosome_df.columns) // 2):
            points = np.concatenate(
                [
                    np.arange(len(centrosome_df), dtype=int).reshape(-1, 1),
                    centrosome_df.values[:, 2 * i : 2 * i + 2],
                ],
                axis=1,
            )
            viewer.add_points(points, name="track%d" % i, face_color="red")
        dataset.reset_axes()
        napari.run()


def mtoc_track_neighbor(
    dataset,
    mtoc_ch,
    n_mtoc,
    search_range=10,
    initial_thresh=3,
    thresh=1.5,
    display=True,
    frame_range=None,
):
    dataset.reset_axes()
    centrosome_df = pd.DataFrame(
        columns=["centrosome_%d_%s" % (i, ax) for i, ax in product(range(n_mtoc), "yx")]
    )
    im = dataset.frames[0]
    mtoc_dict = locate_by_threshold(im, mtoc_ch, 3, initial_thresh, n_mtoc)
    print(mtoc_dict)
    results = [mtoc_dict]
    if frame_range is None:
        frame_rg = range(1, len(dataset.frames))
    else:
        frame_rg = range(1, frame_range)
    for i in tqdm(frame_rg):
        new_mtoc_dict = {}
        for j in range(n_mtoc):
            x, y = int(mtoc_dict["centrosome_%d_x" % j]), int(
                mtoc_dict["centrosome_%d_y" % j]
            )
            im = dataset.frames[i][
                :,
                y - search_range : y + search_range,
                x - search_range : x + search_range,
            ]
            single_mtoc_dict = locate_by_threshold(im, mtoc_ch, 3, thresh, 1)
            single_mtoc_dict["centrosome_%d_x" % j] = (
                single_mtoc_dict["centrosome_0_x"] + x - search_range
            )
            single_mtoc_dict["centrosome_%d_y" % j] = (
                single_mtoc_dict["centrosome_0_y"] + y - search_range
            )
            if j > 0:
                del single_mtoc_dict["centrosome_0_x"]
                del single_mtoc_dict["centrosome_0_y"]
            new_mtoc_dict.update(single_mtoc_dict)
        results.append(new_mtoc_dict)
        mtoc_dict = new_mtoc_dict

    centrosome_df = pd.concat([centrosome_df, pd.DataFrame.from_records(results)])
    centrosome_df.reset_index(drop=True, inplace=True)
    print(centrosome_df.head())
    centrosome_df.to_hdf(dataset.full_mtoc_path, key="mtoc")
    if display:
        if "c" in dataset.frames.axes:
            dataset.frames.iter_axes = "c"
        dataset.frames.bundle_axes = "tyx"
        if frame_range is None:
            viewer = napari.view_image(dataset.frames[mtoc_ch])
        else:
            viewer = napari.view_image(dataset.frames[mtoc_ch][:frame_range])
        for i in range(len(centrosome_df.columns) // 2):
            points = np.concatenate(
                [
                    np.arange(len(centrosome_df), dtype=int).reshape(-1, 1),
                    centrosome_df.values[:, 2 * i : 2 * i + 2],
                ],
                axis=1,
            )
            viewer.add_points(points, name="track%d" % i, face_color="red")
        dataset.reset_axes()
        napari.run()


def locate_by_threshold(im, ch, kernel_size, thresh, n_mtoc):
    if len(im.shape) == 3:
        im = im[ch, :, :]
    im = medfilt2d(im, kernel_size=kernel_size)
    im = ndi.binary_fill_holes(im > threshold_otsu(im) * thresh)
    im = label(im)
    regions = regionprops(im)
    areas = np.array([region.area for region in regions])
    locs = np.array([list(regions[i].centroid) for i in areas.argsort()[-n_mtoc:]])
    new_dict = {
        "centrosome_%d_%s" % (i, ax): locs[i, "yx".find(ax)]
        for i, ax in product(range(n_mtoc), "xy")
    }
    # print("%d spots found." % len(regions))
    return new_dict
