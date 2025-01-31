
# Installation

We recommend using Spaceborne in a dedicated Conda environment. This ensures all dependencies are properly managed.

## 1. Create and Activate the Environment

In the root folder of the Spaceborne repository, run

```bash
conda env create -f environment_<platform>.yaml
conda activate spaceborne
```

Replace `<platform>` with either `linux` or `macOS`, depending on your operating system.
Some of the depenencies may be remove in case you do not plan to use some interfaces, such as 
`OneCovariance` 

---

## 2. Install Spaceborne

### Option A: Using `pip`

To install Spaceborne directly:

```bash
pip install .
```

### Option B: Using Poetry

[Poetry](https://python-poetry.org/) is an alternative package manager. To use it:

1. Install Poetry:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. Install Spaceborne:
   ```bash
   poetry install
   ```

---

## 3. Install Julia for Computational Efficiency

Spaceborne leverages `julia` for computationally intensive tasks. We recommend installing `julia` via [`juliaup`](https://github.com/JuliaLang/juliaup):

```bash
curl -fsSL https://install.julialang.org | sh  # Install juliaup
juliaup default 1.10                           # Install Julia version 1.10
```

Then, install the required Julia packages:

```bash
julia -e 'using Pkg; Pkg.add("LoopVectorization"); Pkg.add("YAML"); Pkg.add("NPZ")'
```

---

# Running the Code

To run Spaceborne, execute the following command:

```bash
python main.py --config=<path_to_config_file>
```

for example:

```bash
python main.py --config="config.yaml"
```

To display the plots generated by the code, add the `--show_plots` flag:

```bash
python main.py --config="config.yaml" --show_plots
```
