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


## Milestone 1
The download of the data and basic loading is done in [load.ipynb](./notebooks/load.ipynb).

Exploration and cleaning of the data is done in [db_investigate.ipynb](./notebooks/db_investigate.ipynb).
Saving to CoNNL-U format of the dataset is done in [create_conllu.ipynb](./notebooks/create_conllu.ipynb), make sure to execute the other 2 notebooks first.

## Milestone 2
First we finished up any preprocessing steps that were left and got our clean dataset where the text is available in raw format, lemmatized and converted to ids.
For every following baseline we are using the format that best fits the model.

We chose 5 baseline models that somewhat align with the baseline models from the paper we were supposed to read at the beginning of the project. Here is a list of our balesines:
-  Most Frequent
-  Regex using Badwords
-  XGBoost
-  DistillBERT
-  DeBERTa-v3-base

Every baselone model has its own notebook in order to maintain a structure and for the group members to be able to work on different models at once. The implementation, result and any useful information for every baseline is provided below.

### Most Frequent baseline
text

### Regex baseline
text

### XGBoost baseline


### DistillBERT baseline
text

### DeBERTa-v3-base
text

### Saving the results
From every notebook we saved the reults in two places.
We saved the metrics for the quantitative comparison and analysis in a .yml(?) format. This is a good approach because then we can call or see the results from all baselines in one place.
For the error analysis and the qualitative analysis we saved the ID of every misclassified sample in a .json file.

### Error analysis
text

