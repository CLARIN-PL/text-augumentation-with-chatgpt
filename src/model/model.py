import torch.nn as nn
import torch
from transformers import AutoModel, AutoTokenizer
from torchmetrics import F1Score, Accuracy, Precision, Recall
import pytorch_lightning as pl
import time


class ClassificationModel(pl.LightningModule):
    def __init__(
        self,
        out_size: int,
        transformer_name: str,
        max_length: int = 512,
        lr: float = 1e-5,
        cuda: bool = True,
        class_proportion: torch.Tensor = None,
    ) -> None:
        super().__init__()
        self.is_cuda = cuda and torch.cuda.is_available()
        self.transformer = AutoModel.from_pretrained(transformer_name)
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_name)

        embedding_size = self.transformer.config.hidden_size
        self.out_size = out_size
        self.linear = nn.Linear(embedding_size, self.out_size)
        # if self.is_cuda:
        #     self.transformer = self.transformer.to("cuda")
        #     self.linear = self.linear.to("cuda")
        self.activation = nn.Softmax(dim=1)
        self.max_length = max_length

        weights = torch.ones(self.out_size)
        weights = weights/class_proportion if class_proportion is not None else None         
        self.loss_function = nn.CrossEntropyLoss(weight=weights)
        self.metrics = {
            "train": self._prepare_metrics(),
            "val": self._prepare_metrics(),
            "test": self._prepare_metrics(),
        }
        self.predictions = {"true": [], "predicted": []}

        self.last_metrics = {}
        self.lr = lr

        self.times = []

    def _prepare_metrics(self):
        return {
            "acc_micro": Accuracy(task="multiclass", average="micro", num_classes=self.out_size),
            "f1_micro": F1Score(task="multiclass", average="micro", num_classes=self.out_size),
            "prec_micro": Precision(task="multiclass", average="micro", num_classes=self.out_size),
            "rec_micro": Recall(task="multiclass", average="micro", num_classes=self.out_size),
            "acc_macro": Accuracy(task="multiclass", average="macro", num_classes=self.out_size),
            "f1_macro": F1Score(task="multiclass", average="macro", num_classes=self.out_size),
            "prec_macro": Precision(task="multiclass", average="macro", num_classes=self.out_size),
            "rec_macro": Recall(task="multiclass", average="macro", num_classes=self.out_size),
            "acc_class": Accuracy(task="multiclass", average="none", num_classes=self.out_size),
            "f1_class": F1Score(task="multiclass", average="none", num_classes=self.out_size),
            "prec_class": Precision(task="multiclass", average="none", num_classes=self.out_size),
            "rec_class": Recall(task="multiclass", average="none", num_classes=self.out_size),
            "loss": []
        }

    def _prepare_input(self, texts):
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def forward(self, X):
        return self._shared_step(X)

    def _shared_step(self, X):
        # inputs = self._prepare_input(X)
        # try:
        inputs = X
                
        tokens = self.tokenizer.batch_encode_plus(
            inputs,
            padding="longest",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        if self.is_cuda:
            tokens = tokens.to("cuda")

        embeds = self.transformer(**tokens).pooler_output
        # return embeds
        out = self.linear(embeds)
        # out = self.activation(out)
        return out

    def training_step(self, batch, batch_idx):
        texts, labels = batch
        # print(labels)
        out = self._shared_step(texts)
        loss = self.loss_function(out, labels)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        # preds = self.activation(out)
        # self.update_metrics("train", labels, preds, loss)
        return loss

    # def on_training_epoch_end(self, *args):
        # metrics = self.calculate_metrics("train")
        # self.log_dict(metrics)

    def update_metrics(self, stage, labels, preds, preds_prob, loss=None):
        labels_short = labels.type(torch.int16)
        for k, metric in self.metrics[stage].items():
            if k == "loss":
                curr_loss = self.loss_function(preds_prob.to("cuda"), labels.to("cuda")).item() if loss is None else loss.item()
                metric.append(curr_loss)
            else:
                metric.update(preds, labels_short)

    def calculate_metrics(self, stage):
        metric_values = {}
        for k, metric in self.metrics[stage].items():
            if k == "loss":
                value = torch.mean(torch.tensor(metric))
                # print(value)
            else:
                value = metric.compute()
                metric.reset()

            if isinstance(value, torch.Tensor):
                if len(value.shape) > 0:
                    values =  value.tolist()
                    for i, v in enumerate(values):
                        metric_values[f"{stage}_{k}_{i}"] = v
                else:
                    value = value.item()
                    metric_values[f"{stage}_{k}"] = value
            else:
                metric_values[f"{stage}_{k}"] = value

        self.last_metrics.update(**metric_values)

        return metric_values

    def validation_step(self, batch, batch_idx):
        texts, labels = batch
        preds = self._shared_step(texts)
        # preds = self.get_predictions_from_output(preds)
        preds = self.activation(preds)
        preds_oh = self.one_hot_to_single(preds)
        labels = self.one_hot_to_single(labels)
        
        if not self.trainer.sanity_checking:
            self.update_metrics("val", labels.to("cpu"), preds_oh.to("cpu"), preds.to("cpu"))

    def on_validation_epoch_end(self, *args, **kwargs) -> None:
        if self.trainer.sanity_checking:
            return None

        metrics = self.calculate_metrics("val")
        self.log_dict(metrics)

    def binarization_prediction(self, out):
        idcs = torch.argmax(out, dim=-1)
        if len(out.shape) > 1:
            idcs = idcs.unsqueeze(1)
        preds = torch.zeros_like(out).scatter_(1, idcs, 1)
        return preds
    
    def one_hot_to_single(self, oh):
        return torch.argmax(oh, dim=1)

    def update_predictions(self, labels, preds):
        self.predictions["true"].append(labels)
        predictions = self.one_hot_to_single(self.binarization_prediction(preds))
        self.predictions["predicted"].append(predictions)

    def test_step(self, batch, batch_idx):
        texts, labels = batch

        # curr = labels[labels == 0.0].sum()
        # print("number of zeros", curr)
        start = time.time()
        preds = self._shared_step(texts)
        preds = self.activation(preds)
        end = time.time()
        self.times.append(end-start)
        # preds = self.binarization_prediction(preds)
        
        preds_oh = self.one_hot_to_single(preds)
        labels = self.one_hot_to_single(labels)
        
        self.update_metrics("test", labels.to("cpu"), preds_oh.to("cpu"), preds.to("cpu"))
        self.update_predictions(labels.to("cpu"), preds.to("cpu"))

    def on_test_epoch_end(self, *args, **kwargs):
        metrics = self.calculate_metrics("test")

        self.last_metrics["true"] = torch.cat(self.predictions["true"], dim=0).tolist()
        self.last_metrics["predicted"] = torch.cat(
            self.predictions["predicted"], dim=0
        ).tolist()
        self.predictions["true"] = []
        self.predictions["predicted"] = []
        self.log_dict(metrics)

    def configure_optimizers(self):
        params = self.parameters()
        wd = 0.001
        optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=wd)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, steps_per_epoch=30, epochs=20)
        # return [optimizer], [scheduler]
