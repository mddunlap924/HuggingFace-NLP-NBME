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
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from nlp.datasets import TestDataset
from torch.utils.data import DataLoader
import itertools
from helper_fns import scoring

TH = 70
MODELS = ['2kde42r1', '2kxv8o19']


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


def list_model_folds(group_id):
    """ List all models and their files (i.e., *.ckpt and *.config) """
    list_models = []
    filelist = []
    model_group = Path('./input') / 'pseudo_label' / f'nbme-{group_id}'

    for root, dirs, files in os.walk(model_group):
        for file in files:
            # append the file name to the list
            filelist.append(os.path.join(root, file))

    model_folds = []
    for filelist_ in filelist:
        # fold_id = filelist_.split(model_group)[-1].split(f'/{group_id}/')[-1].split('/')[1].split('_')[-1]
        fold_id = filelist_.split('_')[-1].split('/')[0]
        if (fold_id not in model_folds) and ('config' not in fold_id):
            model_folds.append(fold_id)

    config, ckpt = None, None
    for model_fold in model_folds:
        for filelist_ in filelist:
            if (model_fold in filelist_) and ('.yaml' in filelist_):
                if 'inf_output' not in filelist_:
                    config = filelist_
            if (model_fold in filelist_) and ('.ckpt' in filelist_):
                ckpt = filelist_
            if (model_fold in filelist_) and ('.json' in filelist_):
                config_json = filelist_

        if 'config_json' not in locals():
            config_json = None
        list_models.append([config, ckpt, config_json])
    return list_models


def mergeIntervals(arr):
    # Sorting based on the increasing order
    # of the start intervals
    arr.sort(key=lambda x: x[0])

    # array to hold the merged intervals
    m = []
    s = -10000
    max = -100000
    for i in range(len(arr)):
        a = arr[i]
        if a[0] > max:
            if i != 0:
                m.append([s, max])
            max = a[1]
            s = a[0]
        else:
            if a[1] >= max:
                max = a[1]

    # 'max' value gives the last point of
    # that particular interval
    # 's' gives the starting point of that interval
    # 'm' array contains the list of all merged intervals

    if max != -100000 and [s, max] not in m:
        m.append([s, max])
    # print("The Merged Intervals are :", end=" ")
    # for i in range(len(m)):
    #     print(m[i], end=" ")
    return m


# def mergeIntervals(intervals):
#     if len(intervals) == 0 or len(intervals) == 1:
#         return intervals
#     intervals.sort(key=lambda x:x[0])
#     result = [intervals[0]]
#     for interval in intervals[1:]:
#         if interval[0] <= result[-1][1]:
#             result[-1][1] = max(result[-1][1], interval[1])
#         else:
#             result.append(interval)
#     return result


class Solution(object):
   def merge(self, intervals):
      """
      :type intervals: List[Interval]
      :rtype: List[Interval]
      """
      if len(intervals) == 0:
         return []
      self.quicksort(intervals,0,len(intervals)-1)
      #for i in intervals:
         #print(i.start, i.end)
      stack = []
      stack.append(intervals[0])
      for i in range(1,len(intervals)):
         last_element= stack[len(stack)-1]
         if last_element[1] >= intervals[i][0]:
            last_element[1] = max(intervals[i][1],last_element[1])
            stack.pop(len(stack)-1)
            stack.append(last_element)
         else:
            stack.append(intervals[i])
      return stack
   def partition(self,array,start,end):
      pivot_index = start
      for i in range(start,end):
         if array[i][0]<=array[end][0]:
            array[i],array[pivot_index] =array[pivot_index],array[i]
            pivot_index+=1
      array[end],array[pivot_index] =array[pivot_index],array[end]
      return pivot_index
   def quicksort(self,array,start,end):
      if start<end:
         partition_index = self.partition(array,start,end)
         self.quicksort(array,start,partition_index-1)
         self.quicksort(array, partition_index + 1, end)




# The following is necessary if you want to use the fast tokenizer for deberta v2 or v3
# This must be done before importing transformers
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



if __name__ == '__main__':

    # https://huggingface.co/docs/transformers/installation#offline-mode
    """ Load CFG """
    cfg_name = 'cfg0.yaml'
    # cfg_name = sys.argv[1]
    CFG = hfu.load_cfg(filename=cfg_name)
    debug = CFG.debug
    del CFG.debug

    """ Seed Everything """
    hfu.seed_all(CFG.seed)

    """ Load Data from Disk with Some PreProcessing """
    train, features, patient_notes = hfl.load_csv_preprocess(data_path=DATA_PATH)
    train_update = train.copy()

    """ Split Data into Training and Validation Folds """
    if CFG.cv_split == 'groupkfold':
        train = split_data.groupkfold(train=train, n_splits=5)

    # """ Load Best Model and Perform Inference to Double-Check Results """
    # MODEL = MODELS[0]

    """ List all models and files for prediction """
    all_model_files = []
    for MODEL in MODELS:
        model_folds = list_model_folds(group_id=MODEL)
        for model_fold in model_folds:
            all_model_files.append(model_fold)
    cols_to_use = ['inf_annotation', 'inf_location', 'merge_location', 'inf_annotation_length', 'ann_diff']

    """ Loop through each model and make inference """
    predictions = []
    for fold_count, fold in enumerate([0, 1, 2, 3, 4]):
        """ Extract Data into Folds """
        train_folds = train[train['fold'] != fold].copy()
        val_folds = train[train['fold'] == fold].copy()

        model_files_by_fold = []
        for model_files in all_model_files:
            config_path, ckpt_path, autoconfig_path = model_files[0], model_files[1], model_files[2]
            # Config File
            with open(config_path, 'r') as file:
                CFG_INF = yaml.safe_load(file)
            file.close()
            CFG_INF = dict2obj(CFG_INF)
            CFG_INF.batch_size = 32
            if CFG_INF.fold == fold:
                model_files_by_fold.append(model_files)

        for model_files in model_files_by_fold:
            config_path, ckpt_path, autoconfig_path = model_files[0], model_files[1], model_files[2]

            # Config File
            with open(config_path, 'r') as file:
                CFG_INF = yaml.safe_load(file)
            file.close()
            CFG_INF = dict2obj(CFG_INF)
            CFG_INF.batch_size = 32
            # fold = CFG_INF.fold

            # val_folds = train.copy().iloc[0:500]

            # Tokenizer
            if ("v3" in CFG_INF.model) or ('v2' not in CFG_INF.model):
                tokenizer_path = INPUT_PATH / CFG_INF.model
                CFG_INF.tokenizer = DebertaV2TokenizerFast.from_pretrained(tokenizer_path)
            else:
                CFG_INF.tokenizer = AutoTokenizer.from_pretrained(CFG.model)

            """ Do Abbrevisations Exists """
            if 'CFG_INF.abbreviations' in locals():
                abb_exist = CFG_INF.abbreviations
            else:
                abb_exist = False

            """ Add Abbreviations """
            if abb_exist:
                CFG_INF.tokenizer.add_tokens(ABBREVIATIONS, special_tokens=False)

            """ Test Dataset """
            test_dataset = TestDataset(CFG_INF, val_folds)
            test_loader = DataLoader(test_dataset,
                                     batch_size=CFG_INF.batch_size,
                                     shuffle=False,
                                     pin_memory=True,
                                     drop_last=False,
                                     num_workers=8,
                                     )
            print("LOADED VAL DATASETS FOR INFERENCE")

            """ Autoconfig Path for Model """
            if autoconfig_path is None:
                autoconfig_path = Path('./input') / CFG_INF.model

            """ Load Inference Model """
            print(f'autoconfig_path: {autoconfig_path}\n automodel_path: {ckpt_path}\n')
            nbme_model = BaseLineModel.load_from_checkpoint(ckpt_path,
                                                            autoconfig_path=autoconfig_path,
                                                            automodel_path=ckpt_path)
            if abb_exist:
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
            print(f'Completed Inference: {ckpt_path}\n')
        predictions = np.mean(predictions, axis=0)
        inf_results = get_results(predictions, th=float(TH/100))
        val_results_ = []
        for val_result in inf_results:
            if val_result == "":
                val_results_.append([])
            else:
                val_results_.append([val_result])
        train_inf = val_folds.copy()
        inf_locations = [[i] if i != '' else [] for i in inf_results]

        ob1 = Solution()

        inf_annotations = []
        merge_locations = []
        for i, inf_location in enumerate(inf_locations):
            train_location = train_inf.location.iloc[i]
            if inf_location == [] and train_location == []:
                merge_locations.append([])
                inf_annotations.append([])
            else:
                if inf_location != []:
                    pred_idxs = [[int(idx.split(' ')[0]), int(idx.split(' ')[1])] for idx in inf_location[0].split(';')]
                else:
                    pred_idxs = []

                if train_location != []:
                    truth_idxs = []
                    for tl in train_location:
                        if ';' in tl:
                            for idx in tl.split(';'):
                                truth_idxs.append([int(idx.split(' ')[0]), int(idx.split(' ')[1])])
                        else:
                            truth_idxs.append([int(tl.split(' ')[0]), int(tl.split(' ')[1])])
                else:
                    truth_idxs = []
                merge_intervals = ob1.merge(truth_idxs + pred_idxs)
                merge_locations.append([';'.join([str(j[0]) + ' ' + str(j[1]) for j in merge_intervals])])

                merge_anns = []
                for mi in merge_intervals:
                    merge_anns.append(train_inf.pn_history.iloc[i][mi[0]:mi[1]])
                inf_annotations.append(merge_anns)

        train_inf['inf_annotation'] = inf_annotations
        train_inf['inf_location'] = inf_locations
        train_inf['merge_location'] = merge_locations
        train_inf['inf_annotation_length'] = train_inf['inf_annotation'].apply(len)
        train_inf['ann_diff'] = train_inf.inf_annotation_length - train_inf.annotation_length
        if fold_count == 0:
            train_final = train_inf.copy()
        else:
            train_final = pd.concat([train_final, train_inf])
        del train_inf
        # train_update = pd.merge(train_update, train_inf[cols_to_use], left_index=True, right_index=True, how='outer')
        print('check point')
    train_final.sort_index(inplace=True)
    train_final.to_csv(Path('./input') / 'pseudo_label' / f'train_inf_val_folds_{TH}.csv', index=False)
    print('End of Loops')
print('End of Training')
