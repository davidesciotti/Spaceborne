## Istallation
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
which may take some time, especially for installing `pyccl`



Run code with 

```bash
 conda activate spaceborne
 python main_release.py
