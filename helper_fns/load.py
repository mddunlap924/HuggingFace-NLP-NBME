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

    # data_load = DataCSV(data_dir=data_path)
    # train = pd.read_csv(Path('./input') / f'pseudo_label/{filename}')
    # features = data_load.load_features()
    # patient_notes = data_load.load_patient_notes()
    #
    # train.merge_location = train.merge_location.str.replace(';', "', '")

    # train.annotation = train.inf_annotation
    # train.annotation_length = train.inf_annotation_length
    # train.location = train.merge_location
    # drop_columns = ['inf_annotation', 'inf_annotation_length', 'inf_location', 'merge_location', 'ann_diff']
    # train.drop(columns=drop_columns, inplace=True)
    #
    # train['annotation'] = train['annotation'].apply(ast.literal_eval)
    # train['location'] = train['location'].apply(ast.literal_eval)

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
    # patient_notes.rename(columns={"locations_merged": "location",
    #                               "annotation_merged": "annotation",
    #                               'annotation_length_merged': 'annotation_length'},
    #                      inplace=True)
    patient_notes['annotation'] = patient_notes['annotation'].apply(ast.literal_eval)
    patient_notes['location'] = patient_notes['location'].apply(ast.literal_eval)
    # pn = pd.read_csv(Path('./input') / f'pseudo_label/patient_notes_modified.csv')
    # pn = pn[pn.labeled == False]
    # pn_mod = pn.copy()
    #
    # pn_dict = {}
    # for idx, row in pn.iterrows():
    #     pn_dict[row['pn_num']] = row['pn_history']
    #
    # new_annotation = []
    # for case_id in features['case_num'].unique():
    #
    #     all_pn_id = set(pn[pn['case_num'] == case_id]['pn_num'].tolist())
    #
    #     for feature_id in features[features['case_num'] == case_id]['feature_num'].unique():
    #         # get all the pn_num that have already been annotated
    #         # annotated_pn = set(train[train['feature_num'] == feature_id]['pn_num'].tolist())
    #         # get all the pn_num that have NOT been annotated
    #         # pn_to_annotate = all_pn_id - annotated_pn
    #         pn_to_annotate = all_pn_id
    #
    #         # get all current annotations
    #         # we will use them to find more annotations
    #         annotations = train[train['feature_num'] == feature_id]['annotation'].tolist()
    #         annotation_texts = set()
    #         for anns in annotations:
    #             # anns = eval(a)
    #             for at in anns:
    #                 annotation_texts.add(at)
    #
    #         # annotate
    #         for pn_id in pn_to_annotate:
    #             new_annotation_pn, new_location_pn = [], []
    #             pn_text = pn_dict[pn_id]
    #             for at in annotation_texts:
    #                 start = pn_text.find(at)
    #                 if start >= 0:
    #                     new_annotation_pn.append(at)
    #                     new_location_pn.append(f'{start} {start + len(at)}')
    #
    #             id_mod = f'{pn_id:05}_{feature_id:03}'
    #             row_idx = pn_mod.index[pn_mod.id == id_mod].tolist()[0]
    #
    #             if len(new_annotation_pn) > 0:
    #                 new_annotation.append((
    #                     f'{pn_id:05}_{case_id}{feature_id:02}',
    #                     # f'{pn_id:05}_{feature_id:02}',
    #                     case_id,
    #                     pn_id,
    #                     feature_id,
    #                     new_annotation_pn,
    #                     new_location_pn,
    #                 ))
    #                 # id_mod = f'{pn_id:05}_{case_id}{feature_id:02}'
    #                 # row_idx = pn_mod.index[pn_mod.id == id_mod].tolist()[0]
    #                 pn_mod.at[row_idx, 'annotation'] = new_annotation_pn
    #                 pn_mod.at[row_idx, 'location'] = new_location_pn
    #                 pn_mod.at[row_idx, 'annotation_length'] = int(len(new_annotation_pn))
    #             else:
    #                 # id_mod = f'{pn_id:05}_{feature_id:03}'
    #                 row_idx = pn_mod.index[pn_mod.id == id_mod].tolist()[0]
    #                 pn_mod.at[row_idx, 'annotation'] = []
    #                 pn_mod.at[row_idx, 'location'] = []
    #                 pn_mod.at[row_idx, 'annotation_length'] = int(0)
    #             print(f'Completed Feature ID: {feature_id}; PN_ID: {id_mod}')
    #     #     break
    #     # break
    #     print(f'Completed Feature ID: {feature_id}')
    # pn_mod.to_csv(Path('./input') / 'pseudo_label' / f'patient_notes_modified_regex.csv', index=False)
    # new_annotation[:10]
    # len(new_annotation)


    return train, features, patient_notes
