# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import argparse
import logging
import yaml
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

from preprocessing.preprocessor import Preprocessor
from dataset.dataset import StrokeDataset


def multi_thread_preprocessor(paths_pair: Tuple, preprocessor: Preprocessor):
    ncct_path, mri_path = paths_pair[0], paths_pair[1]
    failure = None
    try:
        metrics, new_files = preprocessor(ncct_path, mri_path)
    except Exception as e:
        del e
        metrics, new_files = None, None
        failure = ncct_path.parent.parent.name
    return metrics, new_files, failure


def main(preprocessing_config_path: Path):

    # Load cfg file
    with open(preprocessing_config_path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)

    # Initilize the dataset
    datapath = Path(cfg['general_config']['datapath'])
    if 'mri_expected' not in cfg['general_config'].keys():
        cfg['general_config']['mri_expected'] = True
    if 'modalities' not in cfg['general_config'].keys():
        if cfg['general_config']['datasets'] in ['tbi']:
            cfg['general_config']['modalities'] = ['ncct', 'ncct-tilt']
        else:
            cfg['general_config']['modalities'] = ['ncct', 'ncct-tilt', 'msk', 'msk-tilt']
    if 'pathology' not in cfg['general_config'].keys():
        cfg['general_config']['pathology'] = ['ais', 'normal']

    dataset = StrokeDataset(
        datapath=datapath,
        data_csv_path=Path(cfg['general_config']['data_csv_path']),
        datasets=[cfg['general_config']['datasets']],
        partitions=['all'],
        modalities=cfg['general_config']['modalities'],
        case_selection=cfg['general_config']['case_selection'],
        filter_discard=True
    )

    ncct_paths, mri_paths = [], []
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        base = 'ncct-tilt' if 'tilt' in sample['ncct'].name else 'ncct'
        ncct_path = sample['ncct']
        mri_path = ncct_path.parent / ncct_path.name.replace(f'{base}.nii.gz', 'dwi.nii.gz')
        if sample['dataset_name'] == 'tum':
            mri_path = Path(str(mri_path).replace('ses-0000', 'ses-0001'))
        if not mri_path.exists():
            mri_path = Path(str(mri_path).replace('dwi', 'adc'))
        if (not mri_path.exists()):
            mri_path = None
            if cfg['general_config']['mri_expected']:
                logging.warning(f'No ADC or DWI found for {sample["subject"]}')
                continue
        ncct_paths.append(ncct_path)
        mri_paths.append(mri_path)

    # ncct_paths = ncct_paths[:10]
    # mri_paths = mri_paths[:10]
    # With the ongoing changes of the code, it is now better to run each step
    # for all the cases and keep goign on, to exploit multithreading in some steps
    original_steps = cfg['general_config']['steps']
    multi_threaded_steps = ['read_and_write_sitk', 'resample', 'intensity_clipping',
                            'brain_region_crop', 'cleaning', 'difference_image', 'registration']
    # Initilize the Preprocessor
    metrics, new_files, failures = [], [], []
    for step in original_steps:
        print(f'Performing step: {step}...')
        cfg['general_config']['steps'] = [step]
        preprocessor = Preprocessor(cfg)
        if step in multi_threaded_steps:
            mp_preprocessor = partial(multi_thread_preprocessor, preprocessor=preprocessor)
            # for pair in tqdm(zip(ncct_paths, mri_paths), total=len(dataset)):
            #     result = mp_preprocessor(pair)
            #     if (result[0] is not None):
            #         metrics.append(result[0])
            #     if (result[1] is not None):
            #         new_files.append(result[1])
            #     if (result[2] is not None):
            #         failures.append(result[2])
            with mp.Pool(mp.cpu_count()) as pool:
                pairs = [i for i in zip(ncct_paths, mri_paths)]
                for result in tqdm(pool.imap(mp_preprocessor, pairs), total=len(pairs)):
                    if (result[0] is not None):
                        metrics.append(result[0])
                    if (result[1] is not None):
                        new_files.append(result[1])
                    if (result[2] is not None):
                        failures.append(result[2])
        else:
            for ncct_path, mri_path in tqdm(zip(ncct_paths, mri_paths), total=len(ncct_paths)):
                try:
                    result = preprocessor(ncct_path, mri_path)
                    metrics.append(result[0])
                    new_files.append(result[1])
                except Exception as e:
                    print(e)
                    failures.append(result[2])
        if step == 'brain_region_crop':
            bbox_meta = {}
            temp_files_path = Path(cfg['brain_region_crop']['bboxes_info_path'])
            for file in temp_files_path.iterdir():
                with open(file, 'r') as jfile:
                    bbox_meta.update(json.load(jfile))
            filename = cfg['brain_region_crop']['bboxes_info_file']
            with open(temp_files_path.parent / filename, 'w') as jfile:
                json.dump(bbox_meta, jfile, indent=4)
    print(list(set(failures)))
    # TODO: Reorganize metrics and new_files file
    # if (cfg['general_config']['metrics_path'] is not None) and bool(metrics):
    #     metrics.append(pd.DataFrame(metric))
    #     metrics_df = pd.concat(metrics, ignore_index=True)
    #     metrics_df.to_csv(cfg['general_config']['metrics_path'])


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-ppcfg', dest='pp_cfg_path', help='Path to the source data')
    args = parser.parse_args()

    # run preprocessing
    main(Path(args.pp_cfg_path))
