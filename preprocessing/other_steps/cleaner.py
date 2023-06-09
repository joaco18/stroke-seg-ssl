# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import shutil
from pathlib import Path
from typing import List
from preprocessing.preprocessing_utils import check_available_modalities


class Cleaner():
    def __init__(self, cfg: dict) -> None:
        self.clean_base_path = cfg['clean_datasets_path']
        self.match_dict = cfg['cleaning_matching']
        self.match_dict = {
            f'{k}.nii.gz': f'{v}.nii.gz' for k, v in self.match_dict.items()}

    def __call__(self, ncct_path: Path, mri_path: Path) -> List[Path]:
        new_files = []
        ncct_path, adc_path, dwi_path = check_available_modalities(ncct_path, mri_path)
        if ncct_path is not None:
            subject_path = ncct_path.parent.parent
        elif adc_path is not None:
            subject_path = adc_path.parent.parent
        else:
            subject_path = dwi_path.parent.parent
        subject = subject_path.name
        dataset_name = subject_path.parent.name
        base_path = Path(self.clean_base_path) / dataset_name / subject
        file_paths = list(subject_path.rglob('*.nii.gz'))
        for file_path in file_paths:
            _, session, modality = (file_path.name).split('_')
            if modality in [key for key in self.match_dict]:
                new_name = f'sub-{subject}_{session}_{self.match_dict[modality]}'
                clean_path = base_path / session
                clean_path.mkdir(exist_ok=True, parents=True)
                shutil.copyfile(file_path, clean_path / new_name)
                new_files.append(clean_path)
        return new_files
