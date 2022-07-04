import os
import ast
import pandas as pd
from helper_fns.incorrect_annotations import correct_incorrect_annotations
from pathlib import Path


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


def load_csv_preprocess_pseudo(data_path, filename):
    train, features, _ = load_csv_preprocess(data_path=data_path)
    patient_notes = pd.read_csv(Path('./input') / f'pseudo_label/{filename}')

    return train, features, patient_notes


def additional_training_data(data_path, filename):
    """
    Add additional labeled training data
    https://www.kaggle.com/code/wuyhbb/get-more-training-data-with-exact-match
    """
    train, features, _ = load_csv_preprocess(data_path=data_path)
    patient_notes = pd.read_csv(Path('./input') / f'pseudo_label/{filename}')
    drop_columns = ['annotation_x', 'location_x', 'annotation_length_x',
                    'annotation_y', 'location_y', 'annotation_length_y',
                    'location_xy']
    drop_columns = ['labeled', 'fold']
    patient_notes.drop(columns=drop_columns, inplace=True)
    patient_notes['annotation'] = patient_notes['annotation'].apply(ast.literal_eval)
    patient_notes['location'] = patient_notes['location'].apply(ast.literal_eval)

    return train, features, patient_notes
