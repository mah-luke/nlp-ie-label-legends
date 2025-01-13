from typing import Set
from polars import Boolean, DataFrame, Int32, col, element
from functools import lru_cache

PREDICT_COLUMNS = ["id", "female"]


@lru_cache
def female_keywords():
    return set(
        [
            "woman",
            "wife",
            "she",
            "her",
            "female",
            "girl",
            "witch",
            "lady",
            "feminist",
            "mother",
        ]
    )


def predict_female(data: DataFrame):
    """Predict if content of text is targeting a female. This is achieved by loading a set of keywords and
    checking whether such a keyword appears in the token list of the text.

    Expect `data` to contain following columns:
        - `tokens`: list[str] List of tokens of the text,
    """
    keywords = female_keywords()
    return data.with_columns(
        col("tokens")
        .list.eval(element().is_in(keywords).any())
        .alias("female")
        .list.first()
        .cast(Int32)
    ).select(PREDICT_COLUMNS)
