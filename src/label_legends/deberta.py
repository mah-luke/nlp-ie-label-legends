import torch
from functools import lru_cache
from polars import DataFrame, Series, String
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    DebertaV2Tokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from label_legends.preprocess import holdout, transform
from label_legends.result import calculate_scores

TRAINING_ARGS = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=20,
    per_device_eval_batch_size=20,
    warmup_steps=250,
    num_train_epochs=1, # FIXME: change back to 5
    adam_epsilon=1e-6,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=2,
    evaluation_strategy="epoch",
    logging_steps=100,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

MODEL_NAME = "microsoft/deberta-v3-base"


def compute_metrics(pred: EvalPrediction):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)  # type: ignore
    scores = calculate_scores(labels, preds)
    return scores.asdict()


@lru_cache
def tokenizer():
    return DebertaV2Tokenizer.from_pretrained(MODEL_NAME)


class SexistDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset(texts: list[str], labels: list[str]):
    encodings = tokenizer()(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    dataset = SexistDataset(encodings, labels)
    return dataset


def load_deberta():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    val, tra = map(transform, holdout())
    train_dataset = load_dataset(tra["text"].to_list(), tra["label"].to_list())
    val_dataset = load_dataset(val["text"].to_list(), val["label"].to_list())

    return Trainer(
        model=model,
        args=TRAINING_ARGS,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
