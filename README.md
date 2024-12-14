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
Most Frequent is our basic baseline. This calculates the most frequent class from the target column and labels everything that class as prediction. 
This baseline will of course be the one that performs the worst. If we only look at the accuracy, we think it is a quite good model (0.74), but every other metric is 0 which tells us that that is not the case. 
The baseline is implemented in the *notebooks/mostFrequent.ipynb* file.

### Regex baseline
For the seconf baseline we chose a regex based one. The purpose of this was to see how well we can predict the labels using a common bad_word dataset. If a text had a word that could be found in the bad_word dataset, it would be labelled "sexist", if not, the label would be "not sexist".
With this we achieved better scores on average but worse accuracy (0.59). But most misclassifications were "sexist" instead on "not sexist" which is better than the other way around.
The baseline is implemented in the *notebooks/regex.ipynb* file.

### XGBoost baseline
text
The baseline is implemented in the *notebooks/xgBoost.ipynb* file.

### DistillBERT baseline
text
The baseline is implemented in the *notebooks/distilbert.ipynb* file.

### DeBERTa-v3-base
text
The baseline is implemented in the *notebooks/debert.ipynb* file.

### Saving the results
From every notebook we saved the reults in two places. 
We saved the metrics for the quantitative comparison and analysis in an mlflow experiment. This is a good approach because then we can call or see the results from all baselines in one place.
| Metrics  | Most Frequent Baseline | Regex Baseline | XGBoost Baseline | DistilBERT Baseline | DeBERTa-v3-base baseline |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Precision  | 0.0000  | 0.3414  | 0.7639  | ??  | 0.7235  |
| Recall  | 0.0000  | 0.6087  | 0.3884  | ??  | 0.6738  |
| F-Score  | 0.0000  | 0.4375  | 0.5150  | ??  | 0.6978  |
| Accuracy  | 0.7404  | 0.5937  | 0.8101  | ??  | 0.8485  |
| Train time  | 0.00 s  | 3.72 s  | ??  | 15516.91 s  | ??  |
| Test time  | 0.00 s  | 6.84 s  | ??  | 411.54 s  | ??  |

For the error analysis and the qualitative analysis we saved the ID of every misclassified sample in a .json file. Finding from this can be seen below under **error analysis**.

### Error analysis
text

