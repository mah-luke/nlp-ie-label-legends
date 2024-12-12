from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import mlflow
from mlflow.client import MlflowClient
from polars import DataFrame

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from label_legends.util import RESOURCE


@dataclass
class Scores:
    precision: float
    recall: float
    fscore: float
    accuracy: float
    tp: int
    tn: int
    fp: int
    fn: int

    def __repr__(self) -> str:
        return f"""\
precision:\t{self.precision:.4f}
recall:\t\t{self.recall:.4f}
fscore:\t\t{self.fscore:.4f}
accuracy:\t{self.accuracy:.4f}
tn: {self.tn}\t fp: {self.fp}
fn: {self.fn}\t tp: {self.tp}"""

    def asdict(self):
        return asdict(self)


def calculate_scores(y_true, y_pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    accuracy = accuracy_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)

    return Scores(
        precision=precision,
        recall=recall,
        fscore=fscore,
        accuracy=accuracy,
        tp=confusion_mat[1, 1],
        tn=confusion_mat[0, 0],
        fp=confusion_mat[0, 1],
        fn=confusion_mat[1, 0],
    )


@lru_cache(1)
def client():
    return MlflowClient()


@lru_cache(1)
def get_experiment(name: str = "label-legends"):
    experiment = mlflow.get_experiment_by_name(name)
    if not experiment:
        experiment = mlflow.get_experiment(mlflow.create_experiment(name))
    return experiment


def get_current(model: str):
    return client().get_model_version_by_alias(model, "current")


def download_predictions(model: str, alias: str = "current"):
    mlflow.artifacts.download_artifacts(
        f"models:/{model}@{alias}/predictions.json",
        dst_path=str(RESOURCE / "mlflow" / model),
    )


def load_predictions(model: str):
    with open(RESOURCE / "mlflow" / model / "predictions.json", "r") as file:
        file_content = json.load(file)

    return DataFrame(file_content["data"], orient="row", schema=file_content["columns"])


# mlflow.xgboost.save_model(clf, RESOURCE / "mlflow" / "xgboost", model_format='json')
# mlflow.artifacts.list_artifacts("models:/xgboost/latest")
