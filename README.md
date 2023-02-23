# Bead tracking and trajectory analysis


This repository includes the code for generating all figures for `Rectified Signal Propagation in Synthetic Optogenetic STING system`.

## OS requirements
The code was tested on the following systems:
- Linux: Ubuntu 22.04

## Environment setup
```
conda env create -f environment.yml
```

## Intrusctions
The full pipeline to reproduce the analysis is as follows:

- Initialize the data, which basically did the preprocessing in advance to save memory
```
python3 pipeline.py --date=220930 --file=220930_1um-beads.nd2 --track-channel=1 --initialize --parallel=8
```

- Optimize the parameters for detecting the beads
```
python3 pipeline.py --date=220930 --file=220930_1um-beads.nd2 --test
```
Running above program is going to popup a GUI in the browser where you should test all the parameters and click `submit` when you are satisfied.

- Detect the beads
```
python3 pipeline.py --date=220930 --file=220930_1um-beads.nd2 --feature-detect --parallel=4
```

- Track the beads
```
python3 pipeline.py --date=220930 --file=220930_1um-beads.nd2 --track --search-range=12 --memory=3 --appear=0.4
```

- Track the MToC
```
python3 pipeline.py --date=220930 --file=220930_1um-beads.nd2 --mtoc-neighbor --mtoc-ch=0 --mtoc-num=1 --init-thresh=1 --thresh=2 --search-range=50
```

Many of the parameters above requires trial and error. For the meaning of the parameters, run
```
python3 pipeline.py --help
```

If you simply want to visualize the data, run
```
python3 pipeline.py --date=220930 --file=220930_1um-beads.nd2 --display
```

The above pipeline generate trajectories of beads and MToC. For further subsequent analysis, refer to the jupyter notebook. After you finish the analysis, you should find all the result figures in `figures` folder

