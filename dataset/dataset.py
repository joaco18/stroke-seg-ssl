# -*-coding:utf-8 -*-
'''
@Time    :   2023/02/08 16:35:01
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import logging
import numpy as np
import pandas as pd

from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset


logging.basicConfig(level=logging.INFO)
this_file_path = Path().resolve()
data_path = this_file_path.parent / 'data'

DEFAULT_NORM_CONFIG = {
    'type': 'min_max',
    'max_val': 255,
    'mask': None,
    'percentiles': (1, 99),
    'dtype': np.uint8
}

DEFAULT_RESIZE_CONFIG = {
    'voxel_size': (1, 1, 1),
    'interpolation_order': 3,
    'img_size': None,
}

DEFAULT_PEPROCESSING_CONFIG = {
    'register': 'mni',   # 'mri'
    'skull_strip': True,
    'resizing': DEFAULT_RESIZE_CONFIG,
    'padding': None,
    'intensities': DEFAULT_NORM_CONFIG
}

MODALITIES = [
    'ctp', 'ncct', 'ncct-tilt', 'adc', 'dwi', 'flair',
    'msk', 'pncct', 'cbf', 'cbv', 'tmax', 'mtt', 'bm'
]
DATASETS = ['apis', 'aisd', 'isles18', 'tbi']


class StrokeDataset(Dataset):
    def __init__(
        self,
        datapath: Path = data_path,
        data_csv_path: Path = data_path/'dataset.csv',
        modalities: List[str] = MODALITIES,
        pathology: List[str] = ['ais', 'normal'],
        partitions: List[str] = ['train', 'validation', 'test'],
        fold: int = 0,
        standard: List[str] = ['gold', '-', 'silver'],
        datasets: List[str] = DATASETS,
        case_selection: List[str] = ['all'],
        cases_to_exclude: List[str] = None,
        filter_discard: bool = True
    ) -> None:
        # Set the atributes
        self.datapath = datapath
        self.data_csv_path = data_csv_path
        self.modalities = modalities
        self.pathology = pathology
        self.partitions = partitions
        self.fold = fold
        self.standard = standard
        self.datasets = []
        for dst in datasets:
            dst = dst if ('tbi' not in dst) else 'tbi'
            self.datasets.append(dst)
        self.case_selection = case_selection
        self.filter_discard = filter_discard
        self.cases_to_exclude = cases_to_exclude

        # Load the dataset csv
        self.complete_df = pd.read_csv(self.data_csv_path, index_col=0)
        self.df = self.complete_df.copy()

        # Filter the desired cases
        if self.filter_discard:
            self.filter_by_discard()
        self.filter_by_lesion_presence()
        if 'all' not in self.datasets:
            self.filter_by_dataset_name()
        if 'all' not in self.partitions:
            self.filter_by_partitions()
        if 'all' not in self.case_selection:
            self.filter_by_case_selection()
        if 'all' not in self.standard:
            self.filter_by_standard()
        if self.modalities is not None:
            self.filter_by_modalities()
        if self.cases_to_exclude is not None:
            self.filter_cases_to_exclude()

        # define the fields that the sample will have:
        self.sample_fields = [
            'subject', 'ais', 'hemisphere', 'brain', 'cerebellum', 'stem', 'dataset_name'
        ]

        # avoid request of only masks in sample for normal cases
        if (not(self.df.ais.all())) and ('msk' in self.modalities):
            raise Exception('Requiring mask from normal images without reference image')

        # Generate description of the dataset
        self.description = self.generate_description()

        # Check if filtering left sth in the dataframe
        assert len(self.df) != 0, 'Dataset is empy, check your filtering parameters'

    def generate_description(self):
        datasets_desc = self.df.dataset_name.value_counts().to_dict()
        pathology_desc = self.df.ais.replace(
            {True: 'ais', False: 'healthy'}).value_counts().to_dict()
        partition_desc = self.df[f'partition_{self.fold}'].value_counts().to_dict()
        tilt_desc = self.df.tilt_corr_needed.value_counts().to_dict()

        description = f'Dataset containing {len(self.df)} cases. Composition:\n' \
            f'\tDatasets:\n\t\t{datasets_desc}\n' \
            f'\tPathologies:\n\t\t{pathology_desc}\n' \
            f'\tPartitions:\n\t\t{partition_desc}\n' \
            f'\tTilt status:\n\t\t{tilt_desc}\n'
        return description

    def filter_by_discard(self):
        self.df = self.df.loc[self.df.discard == 'n']
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_lesion_presence(self):
        if 'ais' in self.pathology:
            if 'normal' in self.pathology:
                pass
            else:
                self.df = self.df.loc[self.df.ais]
        elif 'normal' in self.pathology:
            self.df = self.df.loc[~self.df.ais]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_dataset_name(self):
        self.df = self.df.loc[self.df.dataset_name.isin(self.datasets)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_partitions(self):
        self.df = self.df.loc[self.df[f'partition_{self.fold}'].isin(self.partitions)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_case_selection(self):
        self.df = self.df.loc[self.df.subject.isin(self.case_selection)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_standard(self):
        self.df = self.df.loc[self.df.standard.isin(self.standard)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_cases_to_exclude(self):
        self.df = self.df.loc[~self.df.subject.isin(self.cases_to_exclude)]
        self.df.reset_index(drop=True, inplace=True)

    def filter_by_modalities(self):
        # If pncct and ncct-tilt allowed, use them instead of ncct
        if 'pncct' in self.modalities:
            if 'ncct' in self.modalities:
                selection = (self.df.pncct != '-') & (self.df.ncct == '-')
                logging.warning(
                    'Be aware that since since pncct modality was included cases '
                    'with pncct image and without ncct, the image provided in '
                    'ncct field will be the pncct.'
                )
            else:
                selection = (self.df.pncct != '-')
                logging.warning(
                    'Be aware that since since pncct the image provided in '
                    'ncct field will be the pncct.'
                )
            self.df.loc[selection, 'ncct'] = self.df.loc[selection, 'pncct']
        if 'ncct-tilt' in self.modalities:
            selection = (self.df['ncct-tilt'] != '-')
            logging.warning(
                'Be aware that since since ncct-tilt the image provided in '
                'ncct field will be the ncct-tilt. GT masks will also be adjusted.'
            )

            self.df.loc[selection, 'ncct'] = self.df.loc[selection, 'ncct-tilt'].values
            selection = selection & (self.df.gt_space == 'ncct')
            self.df.loc[selection, 'msk'] = self.df.loc[selection, 'msk-tilt'].values

        self.modalities = [
            mod for mod in self.modalities if mod not in ['pncct', 'ncct-tilt', 'msk-tilt']]

        for modality in self.modalities:
            self.df = self.df.loc[self.df[modality] != '-', :]
        self.df.reset_index(drop=True, inplace=True)

    def __str__(self) -> str:
        logging.info(self.description)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> Dict:
        df_row = self.df.loc[idx, :].squeeze()
        dataset_name = df_row.dataset_name
        dataset_path = self.datapath / dataset_name
        clean_dataset_path = self.datapath / 'clean' / dataset_name

        # fill all the sample fields with the dataframe data
        sample = {}
        for field in self.sample_fields:
            sample[field] = df_row.get(field)
        for field in self.modalities:
            sample[field] = dataset_path / df_row.get(field)

        # Get all the filenames
        clean_names = ['ncct', 'ncct-pp', 'ncct-pp-flip', 'adc', 'adc-pp', 'dwi',
                       'dwi-pp', 'msk', 'msk-pp', 'stseg', 'vasc-pp', 'bm', 'diff-pp']
        sample['clean'], sample['crop'] = {}, {}
        for name in clean_names:
            ses = '0000'
            path = f'{df_row.get("subject")}/ses-{ses}'
            path = f'{path}/sub-{df_row.get("subject")}_ses-{ses}_{name}.nii.gz'
            # if 'msk' in name:
            #     print(clean_dataset_path / path)
            if (clean_dataset_path / path).exists():
                sample['clean'][name] = clean_dataset_path / path
            path = path.replace('.nii', '-c.nii')
            if (clean_dataset_path / path).exists():
                sample['crop'][name] = clean_dataset_path / path

        return sample
