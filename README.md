# label-legends


## Development
We are storing the project dependencies and do the project setup using the [pyproject.toml](./pyproject.toml) file.
To install the package with the dependencies in a new Python virtual environment:

```bash
uv sync
```
If you do not have uv available, you can also use pip directly in an already set up Python environment:
```bash
pip install -e .
```
> NOTE: If you install the project using pip, make sure to manually install the development dependencies listed in `[dependency-groups]`.


## Milestone
The download of the data and basic loading is done in [load.ipynb](./notebooks/load.ipynb).

Exploration and cleaning of the data is done in [db_investigate.ipynb](./notebooks/db_investigate.ipynb).
Saving to CoNNL-U format of the dataset is done in [create_conllu.ipynb](./notebooks/create_conllu.ipynb), make sure to execute the other 2 notebooks first.


