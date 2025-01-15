from typing import Set
from polars import Boolean, DataFrame, Int32, col, element
from functools import lru_cache
from label_legends.util import RESOURCE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@lru_cache
def load_swear_words(file_path: str) -> set:
    """
    Load a list of swear words from a file.
    Each line in the file represents a swear word or phrase.
    """
    with open(file_path, "r") as file:
        return set(line.strip().lower() for line in file if line.strip())

swear_words = load_swear_words(RESOURCE / 'en')

def predict_swear(data: DataFrame, swear_words_file: str):
    """
    Predict if the content of text contains swear words.
    Swear words are loaded from a file and checked against the text.
    """
    # Check if any text contains a swear word
    return data.with_columns(
        col("text")
        .apply(lambda text: 1 if any(swear_word in text.lower() for swear_word in swear_words) else 0)
        .alias("swear")
        .cast(Int32)
    ).select(PREDICT_COLUMNS)


def predict_negative_sentiment(data: DataFrame):
    """
    Predict if the sentiment of the text is negative using the VADER model.
    
    Adds a `negative` column based on the compound sentiment score.
    """
    analyzer = SentimentIntensityAnalyzer()
    
    return data.with_columns(
        col("text")
        .apply(lambda text: 1 if analyzer.polarity_scores(text)['compound'] < 0 else 0)
        .alias("negative")
    ).select(PREDICT_COLUMNS)



