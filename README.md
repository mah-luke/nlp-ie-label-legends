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

## Model documentation

## Milestone 1

The download of the data and basic loading is done in [load.ipynb](./notebooks/load.ipynb).

Exploration and cleaning of the data is done in [db_investigate.ipynb](./notebooks/db_investigate.ipynb).
Saving to CoNNL-U format of the dataset is done in [create_conllu.ipynb](./notebooks/create_conllu.ipynb), make sure to execute the other 2 notebooks first.

## Milestone 2

First we finished up any preprocessing steps that were left and got our clean dataset where the text is available in raw format, lemmatized and converted to ids.
For every following baseline we are using the format that best fits the model.

We chose 5 baseline models that somewhat align with the baseline models from the paper we were supposed to read at the beginning of the project. Here is a list of our balesines:

- Most Frequent
- Regex using Badwords
- XGBoost
- DistillBERT
- DeBERTa-v3-base

Every baselone model has its own notebook in order to maintain a structure and for the group members to be able to work on different models at once. The implementation, result and any useful information for every baseline is provided below.

### Most Frequent baseline

Most Frequent is our basic baseline. This calculates the most frequent class from the target column and labels everything that class as prediction.
This baseline will of course be the one that performs the worst. If we only look at the accuracy, we think it is a quite good model (0.74), but every other metric is 0 which tells us that that is not the case.
The baseline is implemented in the _notebooks/mostFrequent.ipynb_ file.

### Regex baseline

For the second baseline we chose a regex based one. The purpose of this was to see how well we can predict the labels using a common bad*word dataset. If a text had a word that could be found in the bad_word dataset, it would be labelled "sexist", if not, the label would be "not sexist".
With this we achieved better scores on average but worse accuracy (0.59). But most misclassifications were "sexist" instead on "not sexist" which is better than the other way around.
The baseline is implemented in the \_notebooks/regex.ipynb* file.

### XGBoost baseline

As traditional ML baseline we chose XGBoost, a tree-based model which is often used as a baseline. It was also used in the paper we are reproducing, which allows us to compare the results we achieved to those from the paper.

After doing the error analysis we also added hyperparameter optimization of the model by using SMAC-3, which yielded slightly better results of an fscore of 55.7% compared to 51.5% for the default configuration. It also shows the limitations of this traditional approach of machine learning for our NLP task: Using only tf-idf as features seem to contain too little information to allow the model to create a well enough distinction of whether a text is sexist.
XGBoost was very performant, requiring substantially less time for both training and prediction on the test set. It could be used e.g. in an environment where it is substantial to get predictions within milliseconds or when only limited computational power is available.


The baseline is implemented in the _notebooks/xgBoost.ipynb_ file.
### DistillBERT baseline

text
The baseline is implemented in the _notebooks/distilbert.ipynb_ file.

### DeBERTa-v3-base

We also wanted to implement some transformer-based deep learning models since they were also used in the paper we were basing our project on. As a baseline we are using also the DeBERTa-v3-base. In the current state of the project we are predicting with no additional parameters trying to accurately indicate if a text is sexist or not. Here we are not using the preprocessed tokens rather applying the standard tokenizer for this model the DebertaV2Tokenizer. The training is performed over three epochs with a learning rate of 2*10on the -5 with batch sizes of 8 and weight decay regularization to prevent overfitting. Evaluation is done at the end of each epoch and the best model is loaded based on the evaluation loss.
The baseline is implemented in the *notebooks/debert.ipynb\* file.

### Saving the results

From every notebook we saved the results in two places.
We saved the metrics for the quantitative comparison and analysis in an mlflow experiment. This is a good approach because then we can call or see the results from all baselines in one place.
| Metrics | Most Frequent Baseline | Regex Baseline | XGBoost Baseline | DistilBERT Baseline | DeBERTa-v3-base baseline |
|:-------------:| -------------:| -------------:| -------------:| -------------:|------------:|
| Precision | 0.0000 | 0.3414 | 0.7639 | 0.7237 | 0.7235 |
| Recall | 0.0000 | 0.6087 | 0.3884 | 0.5583 | 0.6738 |
| F-Score | 0.0000 | 0.4375 | 0.5150 | 0.6303 | 0.6978 |
| Accuracy | 0.7404 | 0.5937 | 0.8101 | 0.8300 | 0.8485 |
| Train time | 0.00 s | 3.72 s | 2.88 s | 1538.93 s | 4893.40 s |
| Test time | 0.00 s | 6.84 s | 0.13 s | 52.21 s | 132.74 s |

For the error analysis and the qualitative analysis we saved the ID of every misclassified sample in a .json file. Finding from this can be seen below under **error analysis**.
From every notebook we saved the reults in two places.
We saved the metrics for the quantitative comparison and analysis in a .yml(?) format. This is a good approach because then we can call or see the results from all baselines in one place.

### Error analysis

text
