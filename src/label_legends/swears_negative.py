from polars import DataFrame, Int32, col
from functools import lru_cache
from label_legends.util import RESOURCE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

PREDICT_COLUMNS_SWEAR = ["id", "swear"]
PREDICT_COLUMNS_NEGATIVE = ["id", "negative"]


@lru_cache
def load_swear_words(file_path: str = str(RESOURCE / "en")):
    """
    Load a list of swear words from a file.
    Each line in the file represents a swear word or phrase.
    """
    with open(file_path, "r") as file:
        return [line.strip().lower() for line in file if line.strip()]


def predict_swear(data: DataFrame):
    """
    Predict if the content of text contains swear words.
    Swear words are loaded from a file and checked against the text.
    """

    swear_words = load_swear_words()
    # Check if any text contains a swear word
    return data.with_columns(
        col("text")
        .str.contains_any(
            swear_words,
            ascii_case_insensitive=True,
        )
        .alias("swear")
        .cast(Int32)
    ).select(PREDICT_COLUMNS_SWEAR)


def predict_negative_sentiment(data: DataFrame):
    """
    Predict if the sentiment of the text is negative using the VADER model.

    Adds a `negative` column based on the compound sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()

    return data.with_columns(
        col("text")
        .map_elements(
            lambda text: 1 if analyzer.polarity_scores(text)["compound"] < 0 else 0,
            return_dtype=Int32,
        )
        .alias("negative")
    ).select(PREDICT_COLUMNS_NEGATIVE)
