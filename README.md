
<p align="center">
  <img src="https://github.com/user-attachments/assets/1e156ecb-75d5-4004-b768-cbd8edab7940" width="300">
</p>

<!-- ![sb_logo](https://github.com/user-attachments/assets/6c5d9280-70b2-4f66-8bfb-c513317aea06) -->

# Spaceborne

---
[![Documentation Status](https://readthedocs.org/projects/spaceborne/badge/?version=latest)](https://spaceborne.readthedocs.io/en/latest/?badge=latest)

For detailed instructions on how to install and use Spaceborne, please refer to the [documentation](https://spaceborne.readthedocs.io/en/latest/).

# TL;DR

To install the code, we recommend using a dedicated Conda environment. Clone the 
repository and run

```bash
$ conda env create -f environment.yaml
$ conda activate spaceborne
$ pip install .
```

🐍 note: using `mamba` instead of `conda` in the first line will significantly speed up the environment creation. To install `mamba`, run `conda install mamba` in your `base` environment

Spaceborne leverages `julia` for computationally intensive tasks. We recommend installing `julia` via [`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
$ curl -fsSL https://install.julialang.org | sh  # Install juliaup
$ juliaup default 1.10                           # Install Julia version 1.10
```

Then, install the required Julia packages:

```bash
$ julia -e 'using Pkg; Pkg.add("LoopVectorization"); Pkg.add("YAML"); Pkg.add("NPZ")'
```

---

# Running the Code

All the available options and configurations can be found, along with their explanation, in the `example_config.yaml` file. To run `Spaceborne` *with the configuration specified in the* `Spaceborne/example_config.yaml` *file*, simply execute the following command:

```bash
$ python main.py
```

If you want to use a configuration file with a different name and/or location, you can instead run with

```bash
$ python main.py --config=<path_to_config_file>
```

for example:

```bash
$ python main.py --config="path/to/my/config/config.yaml"
```

To display the plots generated by the code, add the `--show_plots` flag:

```bash
$ python main.py --config="path/to/my/config/config.yaml" --show_plots
```
