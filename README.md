## Istallation
Spaceborne's dependencies are handled following modern standards; for this, we use a combination of [conda]([url](https://www.anaconda.com/)) and [Poetry]([url](https://python-poetry.org/)).


If you have nit yet done so, install Poetry with
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Create a `conda` environment with
```bash
conda create python=3.12 -n spaceborne
```
Then activate it with 
```bash
conda activate spaceborne
```
Once having activated the environment, install the dependencies with 
```bash
poetry install
```
which may take some time, especially for installing `pyccl`.

Run code with 

```bash
 conda activate spaceborne
 python main.py
