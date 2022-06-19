import os
import ast
import pandas as pd
from helper_fns.incorrect_annotations import correct_incorrect_annotations
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import numpy as np
from collections import Counter


def preprocess_features(features):
    features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
    return features


class DataCSV:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_train(self, *, mod=True):
        train = pd.read_csv(os.path.join(self.data_dir, 'train.csv'))
        if mod:
            train['annotation'] = train['annotation'].apply(ast.literal_eval)
            train['location'] = train['location'].apply(ast.literal_eval)
        return train

    def load_features(self, *, preprocess=True):
        features = pd.read_csv(os.path.join(self.data_dir, 'features.csv'))
        if preprocess:
            features = preprocess_features(features)
        return features

    def load_patient_notes(self):
        patient_notes = pd.read_csv(os.path.join(self.data_dir, 'patient_notes.csv'))
        return patient_notes


def load_csv_preprocess(data_path):
    data_load = DataCSV(data_dir=data_path)
    train = data_load.load_train()
    features = data_load.load_features()
    patient_notes = data_load.load_patient_notes()
    train = train.merge(features, on=['feature_num', 'case_num'], how='left')
    train = train.merge(patient_notes, on=['pn_num', 'case_num'], how='left')
    train = correct_incorrect_annotations(train=train)
    train['annotation_length'] = train['annotation'].apply(len)
    return train, features, patient_notes


def groupkfold(train, *, n_splits=5):
    Fold = GroupKFold(n_splits=n_splits)
    groups = train['pn_num'].values
    for n, (train_index, val_index) in enumerate(Fold.split(train, train['location'], groups)):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    return train


def stratifiedkfold(patient_notes, SEED, *, n_splits=5):
    """ FOLDS
        There are two possibilities that come to my mind for splitting the data :
        - A k-fold on features stratified by `case_num`
        - A k-fold on features grouped by `case_num`
        From my understanding, clinical cases will be the same in the train and test data, hence I'm going with the
        first option. """
    skf = StratifiedKFold(n_splits=n_splits, random_state=SEED, shuffle=True)
    splits = list(skf.split(X=patient_notes, y=patient_notes['case_num']))
    folds = np.zeros(len(patient_notes), dtype=int)
    for i, (train_idx, val_idx) in enumerate(splits):
        folds[val_idx] = i
        df_val = patient_notes.iloc[val_idx]
        print(f'   -> Fold {i}')
        print('- Number of samples:', len(df_val))
        print('- Case repartition:', dict(Counter(df_val['case_num'])), '\n')
        patient_notes['fold'] = folds
        patient_notes[['case_num', 'pn_num', 'fold']].to_csv('folds.csv', index=False)

    return patient_notes
