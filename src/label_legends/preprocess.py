from functools import lru_cache
import math
import numpy as np
from polars import (
    Boolean,
    DataFrame,
    Int64,
    LazyFrame,
    List,
    Series,
    String,
    col,
    scan_csv,
)
import polars as pl
import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from stanza import Pipeline, Document
import sklearn.pipeline
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


def token_lists(df):
    if type(df) is LazyFrame:
        df = df.collect()
    return df.select("tokens").to_series().to_list()


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


def load_own():
    df = (
        scan_csv(RESOURCE / "own_test_set.csv")
        .rename({"label": "label_sexist"})
        .collect()
    )
    texts: list[str] = list(df.select(col("text")).to_series())
    nlp = Pipeline("en", processors="tokenize,mwt,lemma,pos", logging_level="warning")
    documents_in = [Document([], text=d) for d in texts]
    docs: list[Document] = nlp(documents_in)

    lemmas = []
    for doc in docs:
        lemmas_sample = []
        for word in doc.iter_words():
            from stanza.models.common.doc import Word

            assert isinstance(word, Word)
            lemmas_sample.append(word.lemma)
        lemmas.append(lemmas_sample)
    tokens = Series("tokens", lemmas)
    return df.with_columns(tokens=tokens)


dataframes = {"train": load_train}


@lru_cache(1)
def holdout():
    shuffled = load_train().collect().sample(fraction=1, shuffle=True, seed=SEED)
    val_size = math.ceil(len(shuffled) * 0.3)
    val, tra = shuffled.head(val_size), shuffled.tail(-val_size)
    return val, tra


@lru_cache(1)
def holdout_majority():
    train = load_train().collect()
    shuffled = majority(train).sample(fraction=1, shuffle=True, seed=SEED)
    val_size = math.ceil(len(shuffled) * 0.3)
    val, tra = shuffled.head(val_size), shuffled.tail(-val_size)
    return val, tra


# Please delete if not good
def majority(df_help2: pl.LazyFrame):
    df_help1 = load_data().collect()
    df_help2 = transform(df_help2)

    df_help2 = df_help2.join(df_help1.select(["id", "rewire_id"]), on="id", how="left")
    print(df_help2.head())
    df_help2 = df_help2.with_columns(
        pl.col("rewire_id")
        .str.extract(r"(\d+)$", 1)
        .cast(pl.Int64)
        .alias("rewire_id_number")
    )

    df = df_help2.group_by("rewire_id_number").agg(
        [
            pl.col("label").sum().alias("label_sum"),
            pl.col("text").first().alias("text"),
            pl.col("tokens").first().alias("tokens"),
            pl.col("token_ids").first().alias("token_ids"),
        ]
    )

    df = df.with_columns(
        pl.when(pl.col("label_sum") >= 2).then(1).otherwise(0).alias("label")
    )
    return df


def transform(df: DataFrame):
    return vectorize_tokens(
        strip_stopwords(
            df.with_columns(
                col("label_sexist")
                .replace({"not sexist": 0, "sexist": 1}, return_dtype=Int64)
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
    vectorizer.fit(token_lists(load_train().collect()))
    vectorizer.vocabulary_["[UNK]"] = len(vectorizer.vocabulary_)
    vectorizer.vocabulary_["[PAD]"] = len(vectorizer.vocabulary_)
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
    return df.with_columns(
        col("tokens").map_elements(analyzer(), return_dtype=List(String))
    )


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
    return df.with_columns(
        col("tokens")
        .map_elements(tokens_to_ids, return_dtype=List(Int64))
        .alias("token_ids")
    )


@lru_cache(1)
def tfidf_pipeline():
    pipeline = sklearn.pipeline.Pipeline(
        [
            (
                "count",
                CountVectorizer(
                    vocabulary=vocabulary(),  # reuse already created vocab
                    stop_words="english",
                    max_features=MAX_FEATURES,
                    lowercase=False,
                    tokenizer=ConlluTokenizer(),
                ),
            ),
            ("tfidf", TfidfTransformer()),
        ]
    )
    return pipeline


@lru_cache(1)
def load_conllu():
    files = (RESOURCE / "conll").glob("*.conll")

    docs = []
    for file in list(sorted(files)):
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
