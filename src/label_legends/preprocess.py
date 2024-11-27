from functools import lru_cache
from polars import LazyFrame, col, scan_csv
from stanza import Pipeline

from label_legends.util import RESOURCE


def preprocess(df: LazyFrame):
    nlp = Pipeline('en', processors='tokenize,lemma,pos')

@lru_cache(1)
def load_data():
    return scan_csv(RESOURCE / "edos_labelled_clean.csv")


@lru_cache(1)
def load_train():
    return load_data().filter(col("split") == "train")


@lru_cache(1)
def load_test():
    return load_data().filter(col("split") == "test")
