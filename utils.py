from matplotlib import pyplot as plt
import fire
from pathlib import Path


def defaultStyle(fs=12):
    plt.rc("font", family="Arial")
    plt.rc("text", usetex=False)
    plt.rc("xtick", labelsize=fs)
    plt.rc("ytick", labelsize=fs)
    plt.rc("axes", labelsize=fs)
    plt.rc("mathtext", fontset="custom", rm="Arial")


def save_fig(fig_id, path="../figures", tight_layout=True, fmt="pdf"):
    Path(path).mkdir(parents=True, exist_ok=True)
    fig_path = Path(path) / f"{fig_id}.{fmt}" 
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(fig_path, format=fmt, transparent=False)


if __name__ == "__main__":
    fire.Fire()
