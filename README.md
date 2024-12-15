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

### Error analysis
The error analysis has the main goal of allowing us to do a thoughrough qualitative analysis. We are trying to find the main challanges that make the predictive models make mistakes. Our analysis can be found in the *notebooks/error_analysis.ipynb* file.

We first loaded the test and the training dataset separately. Then loaded all the model predictions from the .json files and merged them to the test dataset on the id. This way we got a big dataset where the texts, their tokens, true labels and all the predictions are together.

We then assigned an error level to every row. This error level represents the number of models that got wrong the prediction for each row. We have 5 baselines, so error level 5 is the worst (meaning all the models got the prediction wrong) and error level 0 is the best.

We also calculated how many texts are in the following error levels:
-  there are **474** with the error level 5,
-  **4659** times predicted every model the right label (error level 0) and
-  in error levels 4 and 5 there are **1269** instances.

For the following qualitatve analysis we saved the instances from the last category (error levels 4 and 5) into a dataset and interpreted the texts in that. We chose this category because we think this way we can see what sentences/words made it hard for the models to predict right.
We found that words pointing to the female sex appear 3 times among the top 20 tokens, making us think that the models have a hard time detecting sarcasm or stereotypes.

We saved the texts from this category and looked at the sentences trying to get more insights from them:
-  There are some words in the text which are left out currently from our tokenization for instance her/he/she which if solved could lead us to better classification. Instance "I cope with it like I'm imagining punching her in the face." Tokenization: ['cope' 'like' 'imagine' 'punch' 'face' '.']
-  Our models are poor when facing text which are stereotypical or sarcastic, for example "I just bought a smart car. It doesn't turn on when the wife is in the driving seat.",  "Well, Trump sure isn't your mother's golf playing President. Zowie.", "STEM ain't hard, wamans can do it too!" or another "But I thought women could instantly detect sexism, creepiness and bad personalities. Oh wait, I meant attractiveness"
-  Our models are also struggling whenever the text contains some kind of slang for example changing women with "wamans", "whamens"
-  There are also some texts which are referring to the LGBTQ community which we are completely missing: "I’m like seeing 2 girls right now, thinking of maybe adding another 1 or 2. No I’m not trolling, it’s doable, I need to make sure “they” are not needy haha"
-  One other insight would be that as labeling these texts aren't as objective as they should be, we are misclassifying sentences like "The next person to play James Bond will be female. Watch it happen." which for some people are sexist but not for everyone. Meaning we need a big enough sample of labels for all text that we can eliminate errors like this which are most likely not misclassification errors rather personal preference one.

### Potential improvemets
As we are only approaching Milestone 2 we are discovering a lot of ways to improve our project:
-  Fine tuning our models with hyperparameters
-  Figuring out ways to try to recognize sarcasm, stereotypical sentences
-  Tokenize some necessary words which were left out like "she", "her"
-  Extend our error analysis to account for weights of words (where possible)
-  Try to collect new labeled sentences with which we can have a different perspective on how our models are performing
-  Introduction of new models if advised
-  Try to tokenize emojis or somehow gain insights from them
-  Restrict our data so a text only appears once with one label based on the majority votes (now we have 1 text appear 3 times as 3 different people labeled each)
-  Trying out other tokenization methods
