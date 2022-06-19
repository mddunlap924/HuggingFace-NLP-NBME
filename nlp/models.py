import torch
import torch.nn as nn
from helper_fns.utils import scoring
from transformers import AdamW, AutoConfig, AutoModel
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup,
                          get_cosine_with_hard_restarts_schedule_with_warmup,
                          )


class BaseLineModel(pl.LightningModule):
    def __init__(self, autoconfig_path, automodel_path, *,
                 model_repo: str = None,
                 model_base: str = None,
                 initial_lr: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                 encoder_lr: float = 2.0E-5,
                 decoder_lr: float = 2.0E-5,
                 fc_dropout: float = 0.1,
                 th: float = 0.5,
                 schedule=None,
                 ):
        super().__init__()

        self.save_hyperparameters()
        self.config = AutoConfig.from_pretrained(autoconfig_path, output_hidden_states=True)
        self.base = AutoModel.from_pretrained(automodel_path, config=self.config)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, 1)

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.trainer.datamodule.cfg.batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) * ab_size

    def feature(self, inputs):
        outputs = self.base(**inputs)
        last_hidden_states = outputs[0]
        return last_hidden_states

    def forward(self, inputs):
        feature = self.feature(inputs)
        outs = self.fc(self.fc_dropout(feature))
        return outs

    def training_step(self, batch, batch_idx):
        inputs, labels, idx = batch
        y_preds = self(inputs)
        loss = nn.BCEWithLogitsLoss(reduction="none")(y_preds.view(-1, 1), labels.view(-1, 1))
        loss = torch.masked_select(loss, labels.view(-1, 1) != -1).mean()
        self.log("loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        inputs, labels, idx = batch
        preds = self(inputs)
        val_loss = nn.BCEWithLogitsLoss(reduction="none")(preds.view(-1, 1), labels.view(-1, 1))
        val_loss = torch.masked_select(val_loss, labels.view(-1, 1) != -1).mean()
        self.log("val_loss", val_loss, prog_bar=True)
        return {"loss": val_loss, "preds": preds, "labels": labels, "idxs": idx}

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        # https: // pytorch - lightning.readthedocs.io / en / stable / notebooks / lightning_examples / text -
        # transformers.html
        model = self
        if self.hparams_initial['schedule'].layer_select == 'option0':
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
        elif self.hparams_initial['schedule'].layer_select == 'option1':
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.base.named_parameters() if not any(nd in n for nd in no_decay)],
                 'lr': self.hparams.encoder_lr, 'weight_decay': self.hparams.weight_decay},
                {'params': [p for n, p in model.base.named_parameters() if any(nd in n for nd in no_decay)],
                 'lr': self.hparams.encoder_lr, 'weight_decay': 0.0},
                # {'params': [p for n, p in model.named_parameters() if "model" not in n],
                #  'lr': self.hparams.decoder_lr, 'weight_decay': 0.0}
            ]

        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.initial_lr,
                          eps=self.hparams.adam_epsilon)

        # Get Learning Rate Scheduler
        scheduler = self.get_scheduler(optimizer=optimizer,
                                       num_train_steps=self.total_steps,
                                       )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        # lrs = []
        # for epoch in range(int(self.total_steps)):
        #     if scheduler is not None:
        #         scheduler['scheduler'].step()
        #     lrs.append(optimizer.param_groups[0]["lr"])
        # plt.plot(lrs)

        return [optimizer], [scheduler]

    def get_scheduler(self, optimizer, num_train_steps):
        lr_scheduler_inputs = self.hparams_initial.schedule
        if lr_scheduler_inputs.name == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=lr_scheduler_inputs.num_warmup_steps,
                                                        num_training_steps=num_train_steps,
                                                        )
        elif lr_scheduler_inputs.name == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=lr_scheduler_inputs.num_warmup_steps,
                                                        num_training_steps=num_train_steps,
                                                        num_cycles=lr_scheduler_inputs.num_cycles,
                                                        )
        elif lr_scheduler_inputs.name == 'cosine_w_restarts':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=lr_scheduler_inputs.num_warmup_steps,
                num_training_steps=num_train_steps,
                num_cycles=2,
            )
        return scheduler

    def training_epoch_end(self, train_step_outputs):
        if not self.trainer.sanity_checking:
            # Average Training Loss
            train_avg_loss = float(torch.tensor([x["loss"] for x in train_step_outputs]).mean().cpu().numpy())
            self.log('train_avg_loss', train_avg_loss, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, val_step_outputs):
        if not self.trainer.sanity_checking:
            # Scoring
            idxs = torch.cat([x["idxs"] for x in val_step_outputs]).detach().cpu().numpy()
            val_texts = [self.trainer.datamodule.val_texts[idx] for idx in idxs]
            val_labels = [self.trainer.datamodule.val_labels[idx] for idx in idxs]
            predictions = np.squeeze(torch.cat([x["preds"].sigmoid() for x in val_step_outputs]).detach().cpu().numpy())
            char_probs = scoring.get_char_probs(val_texts, predictions, self.trainer.datamodule.cfg.tokenizer)
            results = scoring.get_results(char_probs, th=self.hparams.th)
            preds = scoring.get_predictions(results)
            score = scoring.get_score(val_labels, preds)
            self.log('val_avg_f1', score, on_epoch=True, prog_bar=True)

            # Average Validation Loss
            val_avg_loss = float(torch.tensor([x["loss"] for x in val_step_outputs]).mean().cpu().numpy())
            self.log('val_avg_loss', val_avg_loss, on_epoch=True, prog_bar=True)
