from dataclasses import asdict, dataclass

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


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
