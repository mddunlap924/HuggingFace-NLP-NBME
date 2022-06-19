import pandas as pd
from pathlib import Path


def additional_pn_labels(pn, features, cols_to_include, ids_labeled):
    """ Prepare Patient Notes for Additional Labels """
    # pn_mod = pd.DataFrame(np.nan, index=[0], columns=cols_to_include + ['labeled'])
    for index, row in pn.iterrows():
        feature = features[features.case_num == row.case_num].copy()
        id = [f'{row.pn_num:05}_{row.case_num}{i:02}' for i in range(len(feature))]
        labeled = [True if i in ids_labeled else False for i in id]
        feature['pn_history'] = row.pn_history
        feature['case_num'] = row.case_num
        feature['pn_num'] = row.pn_num
        feature['id'] = id
        feature['labeled'] = labeled
        feature['annotation'] = None
        feature['location'] = None
        feature['annotation_length'] = None
        feature['fold'] = None
        feature = feature[cols_to_include + ['labeled']]
        if index == 0:
            pn_mod = feature.copy()
        else:
            pn_mod = pd.concat([pn_mod, feature])
        del feature
        print(f'index: {index}')
    pn_mod.to_csv(Path('./input') / 'pseudo_label' / f'patient_notes_modified.csv', index=False)
    return pn_mod
