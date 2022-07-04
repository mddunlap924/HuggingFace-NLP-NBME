# NLP using HuggingFace: Named Entity Recognition (NER)
## Introduction
A state-of-the-art natural language processing (NLP) model is trained on a custom labeled dataset with the objective of extracting sub-text for given features. The model was developed using [Transformers from the Hugging Face API ](https://huggingface.co/docs/transformers/index). The model performs [named-entity recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition) by locating character spans for predefined categories in unstructured text.  This repository contains a subset of scripts and data I created when competing in the National Board of Medical Examiners (NBME) - Score Clinical Patients Notes competition on [Kaggle](https://www.kaggle.com/competitions/nbme-score-clinical-patient-notes).

In the below images annotations and NER visualizations are shown using [spaCy](https://spacy.io/). Annotations were performed by multiple practitioners/subject matter experts (SMEs); therefore, some annotations are repeated. Please refer to the [visualization_ner.ipynb](https://github.com/mddunlap924/HuggingFace-NLP-NBME/blob/main/visualization_ner.ipynb) notebook to see how to create these images.

![](https://github.com/mddunlap924/HuggingFace-NLP-NBME/blob/main/imgs/annotated_text.png)

In the below example  the spaCy [en_core_web_sm](https://spacy.io/models/en) english pipeline is used to perform a NER task. This pretrained pipeline consists of tok2vec, tagger, parser, ner, etc. In the below image named entities are assigned such as date, organization, cardinal, etc. Please note this pretrained model is not optimized to find the character spans for the labeled features and is simply a NER visualization example.

![](https://github.com/mddunlap924/HuggingFace-NLP-NBME/blob/main/imgs/visualize_ner.png)

## Custom Training and Fine-Tuning a Transformer Model

The [training.py](https://github.com/mddunlap924/HuggingFace-NLP-NBME/blob/main/training.py) script can be used as a generic workflow for building and tuning customized NER models using the HuggingFace API. A baseline [Microsoft deberta-large-v3](https://huggingface.co/microsoft/deberta-v3-large) model was fine-tuned on the labeled dataset by using a [PyTorch Lightning](https://www.pytorchlightning.ai/) (PL). The PL model for training can be [models.py](https://github.com/mddunlap924/HuggingFace-NLP-NBME/blob/main/nlp/models.py) and the snippet of code showing how to setup the model, forward function, and features are shown below. Other scripts are in the "nlp" folder to help with creating the datasets and dataloaders for efficient data intake and processing.

```python
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
```

Typically, I setup a bash execution for training so the computer can run continuously. This is very helpful with efficient use of your time because it allows you to work on other tasks and/or leave the computer running for days while still training flexible/customizable routines. Training is customized by use of YAML configuration files found in the [cfg](https://github.com/mddunlap924/HuggingFace-NLP-NBME/tree/main/cfgs) folder. Execution of the configuration files is performed using a simple bash script ([execute_training_cfgs.sh](https://github.com/mddunlap924/HuggingFace-NLP-NBME/blob/main/execute_training_cfgs.sh)) as shown in the below snippet. This workflow allows for lots of flexibility and customization during training. The training logs and metrics of interest are logged using [Weights and Biases](https://wandb.ai/site). 

```shell
# Bash script to execute customized model training.
# YAML configurations files are used to direct training factors such as datasets, models, hyperparameters, etc.

python3 training.py cfg0.yaml &&
wait
sleep 10

python3 training.py cfg1.yaml &&
wait
sleep 10

python3 training.py cfg2.yaml &&
wait
sleep 10
```

Users are encouraged to modify the files as they see fit to best work with their applications.
