""" Script used to train custom NER models """
import helper_fns.utils as hfu
import helper_fns.load as hfl
from helper_fns import split_data
from nlp.datasets import NBMEDataModule
import pytorch_lightning as pl
from nlp.models import BaseLineModel
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from transformers import AutoTokenizer
import os
import gc
import copy
import yaml
import sys
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from nlp.datasets import TestDataset
from torch.utils.data import DataLoader
import itertools
from helper_fns import scoring
import ast


def inference_fn(test_loader, model, device):
    preds = []
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions_ = np.concatenate(preds)
    return predictions_


def dict2obj(d):
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    # declaring a class

    class C:
        pass

    # constructor of the class passed to obj
    obj = C()
    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
    return obj


def get_char_probs(texts, predictions, tokenizer):
    """ Probability for each Character """
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text,
                            add_special_tokens=True,
                            return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(zip(encoded['offset_mapping'], prediction)):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [list(g) for _, g in itertools.groupby(result, key=lambda n, c=itertools.count(): n - next(c))]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers hey
import shutil
from pathlib import Path

transformers_path = Path("./venv/lib/python3.9/site-packages/transformers")
input_dir = Path("./input/deberta-v2-3-fast-tokenizer")
convert_file = input_dir / "convert_slow_tokenizer.py"
conversion_path = transformers_path / convert_file.name
if conversion_path.exists():
    conversion_path.unlink()
shutil.copy(convert_file, transformers_path)
deberta_v2_path = transformers_path / "models" / "deberta_v2"
for filename in ['tokenization_deberta_v2.py', 'tokenization_deberta_v2_fast.py']:
    filepath = deberta_v2_path / filename
    if filepath.exists():
        filepath.unlink()
    shutil.copy(input_dir / filename, filepath)
from transformers.models.deberta_v2.tokenization_deberta_v2_fast import DebertaV2TokenizerFast

INPUT_PATH = Path("./input")
DATA_PATH = INPUT_PATH / 'nbme-score-clinical-patient-notes'
os.environ["TOKENIZERS_PARALLELISM"] = "true"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ABBREVIATIONS = ['mi', 'cad', 'chd', 'tsh', 'hx', 'ht', 'cp', 'adhd', 'sob', 'd+', 'rlq', 'lmp', 'lap', 'pap',
                 'iud', 'n/v', 'n/v/d', 'pms', 'meds', 'fhx', 'pud', 'nsaid', 'etoh', 'brbpr', 'tums', 'ekg',
                 'cmp', 'cbc', 'a/w', 'uri', 'urti', 'hpi', 'ocp', 'ha', 'c/o', 'dx', 'uti']


class PeriodicCheckpoint(ModelCheckpoint):
    # https://github.com/PyTorchLightning/pytorch-lightning/issues/5473
    def __init__(self, every: int):
        super().__init__(period=-1)
        self.every = every

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ):
        if pl_module.global_step % self.every == 0:
            assert self.dirpath is not None
            current = Path(self.dirpath) / f"latest-{pl_module.global_step}.ckpt"
            prev = (
                Path(self.dirpath) / f"latest-{pl_module.global_step - self.every}.ckpt"
            )
            trainer.save_checkpoint(current)
            prev.unlink(missing_ok=True)


if __name__ == '__main__':
    # https://huggingface.co/docs/transformers/installation#offline-mode
    """ Load CFG """
    # cfg_name = 'cfg0.yaml'
    cfg_name = sys.argv[1]
    CFG = hfu.load_cfg(filename=cfg_name)
    debug = CFG.debug
    del CFG.debug

    """ Seed Everything """
    hfu.seed_all(CFG.seed)

    """ Tokenizer """
    if ("v3" in CFG.model) and ('v2' not in CFG.model):
        tokenizer_path = INPUT_PATH / CFG.model
        CFG.tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
    else:
        CFG.tokenizer = AutoTokenizer.from_pretrained(Path('./input') / CFG.model)

    if CFG.abbreviations:
        CFG.tokenizer.add_tokens(ABBREVIATIONS, special_tokens=False)

    """ Weight and Biases (WandB) """
    wandb_group_id = wandb.util.generate_id()

    """ Train Each Fold """
    for fold in CFG.trn_fold:
        """ Load Data from Disk with Some PreProcessing """
        if CFG.pseudo_labels.apply:
            if CFG.pseudo_labels.regex:
                pseudo_filename = CFG.pseudo_labels.filename.replace('XYZ', str(fold))
                train, features, patient_notes = hfl.additional_training_data(data_path=DATA_PATH,
                                                                              filename=pseudo_filename)
            else:
                pseudo_filename = CFG.pseudo_labels.filename.replace('XYZ', str(fold))
                train, features, patient_notes = hfl.load_csv_preprocess_pseudo(data_path=DATA_PATH,
                                                                                filename=pseudo_filename)
        else:
            train, features, patient_notes = hfl.load_csv_preprocess(data_path=DATA_PATH)

        """ Split Data into Training and Validation Folds """
        if CFG.cv_split == 'groupkfold':
            train = split_data.groupkfold(train=train, n_splits=CFG.num_folds)

        """ Weight and Biases Logging For Each Fold """
        CFG.fold = fold
        if not debug:
            wb_logger = hfu.WandB(cfg=copy.copy(CFG), group_id=wandb_group_id).get_logger()
            logger = [wb_logger]
        else:
            logger = []

        """ Extract Data into Folds """
        train_folds = train[train['fold'] != fold].copy().reset_index(drop=True)
        val_folds = train[train['fold'] == fold].copy().reset_index(drop=True)
        if CFG.pseudo_labels.apply:
            if CFG.pseudo_labels.regex:
                pseudo_data = patient_notes.copy()
                if CFG.pseudo_labels.split is not False:
                    pseudo_data = split_data.groupkfold(train=pseudo_data, n_splits=CFG.pseudo_labels.split)
                    pseudo_data = pseudo_data[pseudo_data['fold'] == fold].copy().reset_index(drop=True)
                    train_folds = pd.concat([train_folds, pseudo_data]).reset_index(drop=True)
                else:
                    train_folds = pd.concat([train_folds, pseudo_data]).reset_index(drop=True)
            else:
                pseudo_data = patient_notes.copy()
                pseudo_data['annotation'] = pseudo_data['annotation'].apply(ast.literal_eval)
                pseudo_data['location'] = pseudo_data['location'].apply(ast.literal_eval)
                if CFG.pseudo_labels.split is not False:
                    pseudo_data = split_data.groupkfold(train=pseudo_data, n_splits=CFG.pseudo_labels.split)
                    pseudo_data = pseudo_data[pseudo_data['fold'] != fold].copy().reset_index(drop=True)
                    if CFG.pseudo_labels.only_baseline:
                        train_folds = pseudo_data[pseudo_data['fold'] == fold].copy().reset_index(drop=True)
                    else:
                        train_folds = pd.concat([train_folds, pseudo_data]).reset_index(drop=True)
                else:
                    train_folds = pd.concat([train_folds, pseudo_data]).reset_index(drop=True)

            # Check if same id are in train_folds and val_folds
            if np.size(np.intersect1d(train_folds.id.to_numpy(), val_folds.id.to_numpy())) == 0:
                print('No "id" overlap exists between train_folds and val_folds')

        """ PL Data Module (contains DataLoaders) """
        dm = NBMEDataModule(cfg=CFG, train=train_folds, val=val_folds)
        # # Check data from datamodule
        # dm.setup()
        # train_dataloader = dm.train_dataloader()
        # batch_of_data = next(iter(train_dataloader))

        """ Define Checkpoint Callback """
        if not debug:
            ckpt_save_path = os.path.join(os.getcwd(), 'input', 'nbme-' + wandb_group_id, wb_logger.experiment.name)
            checkpoint_callback = ModelCheckpoint(
                dirpath=ckpt_save_path,
                monitor=CFG.checkpoint.monitor.metric,
                mode=CFG.checkpoint.monitor.mode,
                save_last=True,
            )

            """ Learning Rate Monitor """
            lr_monitor = LearningRateMonitor(logging_interval='step')
            callbacks = [checkpoint_callback, lr_monitor]
        else:
            ckpt_save_path = None
            callbacks = []

        """ Model """
        autoconfig_path = INPUT_PATH / CFG.model / 'config.json'
        automodel_path = INPUT_PATH / CFG.model
        if not CFG.checkpoint.continue_training:
            model = BaseLineModel(model_repo=CFG.model,
                                  model_base=CFG.model_base,
                                  autoconfig_path=autoconfig_path,
                                  automodel_path=automodel_path,
                                  initial_lr=CFG.optimizer.initial_lr,
                                  adam_epsilon=CFG.optimizer.epsilon,
                                  warmup_steps=CFG.optimizer.weight_decay,
                                  weight_decay=CFG.optimizer.weight_decay,
                                  encoder_lr=CFG.optimizer.encoder_lr,
                                  decoder_lr=CFG.optimizer.decoder_lr,
                                  fc_dropout=CFG.fc_dropout,
                                  schedule=CFG.schedule,
                                  )
            CFG.resume_from_checkpoint = False
        else:
            ckpt_folders = os.listdir(f'./input/nbme-{CFG.checkpoint.group_id}')
            for ckpt_folder in ckpt_folders:
                ckpt_config_path = f'./input/nbme-{CFG.checkpoint.group_id}/{ckpt_folder}/config.yaml'
                with open(ckpt_config_path, 'r') as file:
                    cfg_ckpt = yaml.safe_load(file)
                file.close()
                if cfg_ckpt['fold'] == fold:
                    ckpt_name = [i for i in os.listdir(f'./input/nbme-{CFG.checkpoint.group_id}/{ckpt_folder}')
                                 if ('ckpt' in i) and ('last' in i)][0]
                    ckpt_path = f'./input/nbme-{CFG.checkpoint.group_id}/{ckpt_folder}/{ckpt_name}'
            print(f"Loading Model: {ckpt_path}")
            model = BaseLineModel.load_from_checkpoint(ckpt_path,
                                                       autoconfig_path=autoconfig_path,
                                                       automodel_path=ckpt_path,
                                                       schedule=CFG.schedule,
                                                       )
            CFG.resume_from_checkpoint = True

        """ Update Model - Embedding is added new tokens """
        if CFG.abbreviations:
            model.base.resize_token_embeddings(len(CFG.tokenizer))

        """ Train the Model """
        trainer = pl.Trainer(gpus=1,
                             val_check_interval=CFG.check_n_steps,
                             log_every_n_steps=CFG.check_n_steps,
                             max_epochs=CFG.epochs,
                             logger=logger,
                             callbacks=callbacks,
                             limit_train_batches=CFG.limit_train_batches,
                             # limit_val_batches=CFG.limit_train_batches,
                             precision=CFG.precision,
                             )
        trainer.fit(model, datamodule=dm)
        # Learning Rate Finder
        # lr_finder = trainer.tuner.lr_find(model, dm, num_training=750)
        # lr_finder.results
        # fig = lr_finder.plot(suggest=True)
        # fig.show()

        """ Print Results to Terminal """
        best_model_path = trainer.callbacks[4].best_model_path
        best_model_score = trainer.callbacks[4].best_model_score.detach().cpu().numpy()
        best_model_score_monitor = trainer.callbacks[4].monitor
        print(f'Completed Fold {fold} of {len(CFG.trn_fold)}: {wb_logger.experiment.name}')
        print(f'Best Model Score {best_model_score_monitor}: {best_model_score}; Path {best_model_path}')

        """ Save Model Configuraiton File to Disk """
        model.base.config.save_pretrained(ckpt_save_path)
        print(f'Model Config File to Disk: {ckpt_save_path}/config.json')

        """ Weights and Biases: save summary metrics and Clean-up Memory """
        if not debug:
            print('Entered into Logging Summary')
            run_config = wb_logger.experiment.config._as_dict()
            with open(os.path.join(ckpt_save_path, 'config.yaml'), 'w') as file:
                yaml.dump(run_config, file)
            file.close()
            wandb.finish()
            api = wandb.Api()
            run = api.run("/".join([WANDB_USER_NAME,
                                    wb_logger.name,
                                    wb_logger.version]))
            run.summary['best_f1'] = best_model_score
            run.summary['best_model_save_name'] = best_model_path.split('/')[-1]
            run.summary['pseudo_labels_apply'] = CFG.pseudo_labels.apply
            run.summary['pseudo_labels_filename'] = CFG.pseudo_labels.filename
            run.summary['ckpt_path'] = ckpt_save_path
            run.summary['ckpt_metric'] = CFG.checkpoint.monitor.metric
            run.summary['kaggle_score'] = 0.000
            run.update()
            wandb.finish()

        """ Clean-up Memory """
        del model, trainer, lr_monitor, checkpoint_callback, api, run
        gc.collect()

        """ Load Best Model and Perform Inference to Double-Check Results """
        config_path = [file for file in os.listdir(ckpt_save_path) if '.yaml' in file][0]
        config_path = os.path.join(ckpt_save_path, config_path)

        # Config File
        with open(config_path, 'r') as file:
            CFG_INF = yaml.safe_load(file)
        file.close()
        CFG_INF = dict2obj(CFG_INF)
        CFG_INF.batch_size = 32


        # Tokenizer
        if ("v3" in CFG_INF.model) or ('v2' not in CFG_INF.model):
            tokenizer_path = INPUT_PATH / CFG_INF.model
            CFG_INF.tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
        else:
            CFG_INF.tokenizer = AutoTokenizer.from_pretrained(CFG.model)

        """ Add Abbreviations """
        if CFG_INF.abbreviations:
            CFG_INF.tokenizer.add_tokens(ABBREVIATIONS, special_tokens=False)

        """ Test Dataset """
        test_dataset = TestDataset(CFG_INF, val_folds)
        test_loader = DataLoader(test_dataset,
                                 batch_size=CFG_INF.batch_size,
                                 shuffle=False,
                                 pin_memory=True,
                                 drop_last=False,
                                 )
        print("LOADED VAL DATASETS FOR INFERENCE")

        """ Load Inference Model """
        ckpt_path = [file for file in os.listdir(ckpt_save_path) if '.ckpt' in file][0]
        ckpt_path = os.path.join(ckpt_save_path, ckpt_path)
        autoconfig_path = Path(ckpt_save_path) / 'config.json'
        print(f'autoconfig_path: {autoconfig_path}\n'
              f'automodel_path: {ckpt_path}\n')
        CFG_INF.autoconfig_path = autoconfig_path
        CFG_INF.automodel_path = automodel_path
        nbme_model = BaseLineModel.load_from_checkpoint(ckpt_path,
                                                        autoconfig_path=autoconfig_path,
                                                        automodel_path=ckpt_path)
        if CFG_INF.abbreviations:
            nbme_model.base.resize_token_embeddings(len(CFG_INF.tokenizer))

        nbme_model.to(DEVICE)
        nbme_model.freeze()
        nbme_model.eval()
        print(f'LOADED BASE MODEL: {ckpt_path}')
        predictions = []
        prediction = inference_fn(test_loader, nbme_model, DEVICE)
        prediction = prediction.reshape((len(val_folds), CFG_INF.max_len))
        char_probs = get_char_probs(val_folds['pn_history'].values, prediction, CFG_INF.tokenizer)
        predictions.append(char_probs)
        del nbme_model, prediction
        gc.collect()
        torch.cuda.empty_cache()
        print(f'Complted Inference: {ckpt_path}\n')
        predictions = np.mean(predictions, axis=0)
        val_results = get_results(predictions, th=0.5)
        val_results_ = []
        for val_result in val_results:
            if val_result == "":
                val_results_.append([])
            else:
                val_results_.append([val_result])
        val_preds = scoring.create_labels_for_scoring(pd.DataFrame({'location': val_results_}))
        val_truth = scoring.create_labels_for_scoring(val_folds)
        val_score = scoring.get_score(val_truth, val_preds)
        print(f'Inference Score for Best Model: {val_score}')
        del val_preds, val_results, val_results_

        """ Determine Best Th Value """
        best_th = 0.5
        best_score = 0.
        for th in np.arange(0.45, 0.55, 0.01):
            th = np.round(th, 2)
            val_results = get_results(char_probs, th=th)
            val_results_ = []
            for val_result in val_results:
                if val_result == "":
                    val_results_.append([])
                else:
                    val_results_.append([val_result])
            val_preds = scoring.create_labels_for_scoring(pd.DataFrame({'location': val_results_}))
            val_th_score = scoring.get_score(val_truth, val_preds)
            if best_score < val_th_score:
                best_th = th
                best_score = val_th_score
        del char_probs

        CFG_INF.best_th = float(best_th)
        CFG_INF.best_th_score = float(best_score)
        CFG_INF.f1_inf = float(val_score)
        CFG_INF.autoconfig_path = str(CFG_INF.autoconfig_path)
        CFG_INF.automodel_path = str(CFG_INF.automodel_path)

        # Save inference CFG to disk as yaml
        del CFG_INF.tokenizer, CFG_INF.schedule, CFG_INF.optimizer
        CFG_INF_DICT = CFG_INF.__dict__
        del CFG_INF_DICT['_wandb']
        with open(os.path.join(ckpt_save_path, 'inf_output_config.yaml'), 'w') as file:
            yaml.dump(CFG_INF_DICT, file)
        file.close()

        """ Weights and Biases: save summary metrics and Clean-up Memory """
        if not debug:
            print('Entered into Logging Summary')
            api = wandb.Api()
            run = api.run("/".join([WANDB_USER_NAME, 
                                    wb_logger.name, 
                                    wb_logger.version]))
            run.summary['f1_inf'] = val_score
            run.summary['best_th'] = best_th
            run.summary['best_th_score'] = best_score
            run.update()
            wandb.finish()

        """ Clean-up Memory """
        del train_folds, val_folds, wb_logger, api, run, CFG_INF, CFG_INF_DICT
        gc.collect()
        del train, patient_notes, features
    print(f'Completed Config: {cfg_name}\n')

print('End of Training')
