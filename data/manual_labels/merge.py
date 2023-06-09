# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/05 09:59:29
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent


def main():
    mseg = []
    for database in ['aisd', 'apis', 'isles18']:
        db = pd.read_csv(repo_root/f'data/manual_labels/{database}_manual_lab.csv')
        if database == 'aisd':
            db['subject'] = [str(subj).rjust(7, '0') for subj in db.subject.tolist()]
        mseg.append(db)
    mseg = pd.concat(mseg, ignore_index=True)
    mseg.drop(columns=['Unnamed: 9', 'wrongly registered'], inplace=True)
    mseg.to_csv(repo_root/'data/datasets_manual_label.csv')


if __name__ == '__main__':
    main()
