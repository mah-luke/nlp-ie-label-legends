from functools import lru_cache
import math
import numpy as np
from polars import DataFrame, Int64, LazyFrame, List, Series, String, col, scan_csv
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from stanza import Pipeline, Document
from stanza.utils.conll import CoNLL

from label_legends.util import COLUMNS, RESOURCE, SEED, CONLL_DIR

MAX_FEATURES = 3000

LOG = logging.getLogger(__name__)


class ConlluTokenizer:
    """Class to use preloaded conll files"""

    default_stopwords = set([])

    def __init__(self, stopwords=None):
        self.stopwords = stopwords if stopwords else self.default_stopwords

    def __call__(self, input: str | Series):
        tokens = [input] if type(input) is str else input
        for token in tokens:
            if token not in self.stopwords:
                yield token


def add_tokens(df: LazyFrame):
    return df.join(load_conllu(), on="id", how="inner")


@lru_cache(1)
def load_data(tokens: bool = True):
    df = scan_csv(RESOURCE / "edos_labelled_clean.csv").rename({"": "id"})
    if tokens:
        df = add_tokens(df)
    return df


@lru_cache(1)
def load_train():
    return load_data().filter(col("split") == "train")


@lru_cache(1)
def load_test():
    return load_data().filter(col("split") == "test")


dataframes = {"train": load_train}


@lru_cache(1)
def holdout():
    shuffled = load_train().collect().sample(fraction=1, shuffle=True, seed=SEED)
    val_size = math.ceil(len(shuffled) * 0.3)
    val, tra = shuffled.head(val_size), shuffled.tail(-val_size)
    return val, tra


def transform(df: DataFrame):
    return vectorize_tokens(
        strip_stopwords(
            df.with_columns(
                col("label_sexist")
                .replace({"not sexist": 0, "sexist": 1})
                .alias("label")
            )
        )
    ).select(COLUMNS)


@lru_cache(1)
def load_vectorizer():
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        lowercase=False,
        tokenizer=ConlluTokenizer(),
        stop_words="english",
    )
    vectorizer.fit(load_train().select("tokens").collect().to_series().to_list())
    vectorizer.vocabulary_["[UNK]"] = MAX_FEATURES
    vectorizer.vocabulary_["[PAD]"] = MAX_FEATURES + 1
    return vectorizer


@lru_cache(1)
def analyzer():
    return load_vectorizer().build_analyzer()


@lru_cache(1)
def vocabulary() -> dict[str, np.int64]:
    return load_vectorizer().vocabulary_


@lru_cache(1)
def reverse_vocabulary():
    return {index: token for token, index in vocabulary().items()}


def strip_stopwords(df: DataFrame):
    return df.with_columns(col("tokens").map_elements(analyzer(), return_dtype=List(String)))


def tokens_to_ids(tokens, remove_miss=False):
    if remove_miss:
        return [vocabulary()[token] for token in tokens if token in vocabulary()]
    return [
        vocabulary()[token] if token in vocabulary() else MAX_FEATURES
        for token in tokens
    ]


def ids_to_tokens(ids, remove_miss=True):
    if remove_miss:
        return [
            reverse_vocabulary()[id] for id in ids
        ]  # if id in reverse_vocabulary()]
    return [
        reverse_vocabulary()[id]
        if id in reverse_vocabulary()
        else reverse_vocabulary()[np.int64(MAX_FEATURES)]
        for id in ids
    ]


def vectorize_tokens(df: DataFrame):
    return df.with_columns(col("tokens").map_elements(tokens_to_ids, return_dtype=List(Int64)).alias("token_ids"))


@lru_cache(1)
def load_conllu():
    files = (RESOURCE / "conll").glob("*.conll")

    docs = []
    for file in sorted(files):
        docs.append(
            [
                int(file.name.removesuffix(".conll")),
                [
                    word.lemma.lower()
                    for token in CoNLL.conll2doc(file).iter_tokens()
                    for word in token.words
                ],
            ]
        )
    return LazyFrame(docs, schema=["id", "tokens"], orient="row")


def create_conllu():
    LOG.info("starting CONLLU creation")
    df = load_data(tokens=False)
    series_text = list(df.select(col("text")).collect().to_series())
    series_id = list(df.select(col("id")).collect().to_series())
    nlp = Pipeline("en", processors="tokenize,mwt,lemma,pos", logging_level="warning")
    documents_in = [Document([], text=d) for d in series_text]
    LOG.info("Stanza Documents created")
    docs = nlp(documents_in)
    LOG.info("Stanza Pipeline finished")

    CONLL_DIR.mkdir(parents=True, exist_ok=True)
    for i, doc in zip(series_id, docs):
        CoNLL.write_doc2conll(doc, str(CONLL_DIR / f"{i}.conll"))
    LOG.info("CONLL documents written to disk")
