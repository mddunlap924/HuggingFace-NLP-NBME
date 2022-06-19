from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
import numpy as np
from collections import Counter


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