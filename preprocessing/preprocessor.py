# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import yaml
import pickle
import subprocess
from pathlib import Path
from typing import Dict, Union, List, Tuple

from preprocessing.registration.registrator import Registrator
from preprocessing.resample.resampler import Resampler
from preprocessing.super_resolution.super_resolutor import SuperResolutor
from preprocessing.skull_stripping.skull_stripper import SkullStripper
from preprocessing.tissue_segmentation.tissue_segmenter import TissueSegmenter
from preprocessing.preprocessing_utils import read_write_sitk
from preprocessing.other_steps.intensity_clipper import IntensityClipper
from preprocessing.other_steps.brain_region_cropper import BrainRegionCropper
from preprocessing.other_steps.difference_image import DifferenceImageGenerator
from preprocessing.other_steps.cleaner import Cleaner


PREPROCESSING_PATH = Path(__file__).parent.resolve()


class Preprocessor():
    def __init__(
        self,
        config_file: Union[Path, Dict],
    ) -> None:
        # Read the configuration file
        if isinstance(config_file, Path):
            self.config_file_path = config_file
            with open(config_file, 'r') as ymlfile:
                self.cfg = yaml.safe_load(ymlfile)
        else:
            self.cfg = config_file

        # Instantiate each preprocessing step
        self.gcfg = self.cfg['general_config']
        self.command_line = self.gcfg['command_line']

        if 'resample' in self.gcfg['steps']:
            self.resampler = Resampler(self.cfg['resample'])

        if self.command_line:
            # If gpu is used, in order to tf to free memory we need
            # to run them as external scripts.
            self.super_resolutor_file = PREPROCESSING_PATH/'super_resolution/super_resolutor.py'
            self.skull_stripper_file = PREPROCESSING_PATH/'skull_stripping/skull_stripper.py'
            self.tissue_segmenter_file = PREPROCESSING_PATH/'tissue_segmentation'
            self.tissue_segmenter_file = self.tissue_segmenter_file/'tissue_segmenter.py'
            # generate a temporary file to get the metrics
            self.temp_file = str(PREPROCESSING_PATH / 'tmp.pkl')
        else:
            if 'super_resolution' in self.gcfg['steps']:
                self.super_resolutor = SuperResolutor(self.cfg['super_resolution'])
            if 'skull_stripping' in self.gcfg['steps']:
                self.skull_stripper = SkullStripper(self.cfg['skull_stripping'])
            if 'tissue_segmentation' in self.gcfg['steps']:
                self.tissue_segmenter = TissueSegmenter(self.cfg['tissue_segmentation'])

        if 'registration' in self.gcfg['steps']:
            self.registrator = Registrator(self.cfg['registration'])

        if 'intensity_clipping' in self.gcfg['steps']:
            self.intensity_clipper = IntensityClipper(self.cfg['intensity_clipping'])

        if 'brain_region_crop' in self.gcfg['steps']:
            self.brain_region_cropper = BrainRegionCropper(self.cfg['brain_region_crop'])

        if 'difference_image' in self.gcfg['steps']:
            self.difference_img_gen = DifferenceImageGenerator(self.cfg['difference_image'])

        if 'cleaning' in self.gcfg['steps']:
            self.cleaner = Cleaner(self.cfg['cleaning'])

    def read_temp_file(self,):
        with open(self.temp_file, 'rb') as pfile:
            content = pickle.load(pfile)
        return content

    def __call__(self,
                 ncct_path: Path,
                 mri_path: Path,
                 new_files: Dict = None,
                 metrics: List = None) -> Tuple[Dict, List]:
        # initiate output variables
        metrics = {} if metrics is None else metrics
        new_files = [] if new_files is None else new_files
        if 'read_and_write_sitk' in self.gcfg['steps']:
            # There was a problem with some images and sitk generated a warning
            # when loading them, if loaded and re-saved the problem was fixed
            read_write_sitk(ncct_path, mri_path)

        if 'resample' in self.gcfg['steps']:
            new_files.extend(self.resampler(ncct_path, mri_path))
            # new_files.self.resampler(ncct_path, mri_path)

        if 'super_resolution' in self.gcfg['steps']:
            if self.command_line:
                command = f'python {str(self.super_resolutor_file)} '\
                    f'-ncct {str(ncct_path)} -mri {str(mri_path)} '\
                    f'-cfg {str(self.config_file_path)} -temp {self.temp_file} '
                subprocess.run(command, shell=True)
                new_files.extend(self.read_temp_file())
            else:
                new_files.extend(self.super_resolutor(ncct_path, mri_path))

        if 'skull_stripping' in self.gcfg['steps']:
            if self.command_line:
                command = f'python {str(self.skull_stripper_file)} '\
                    f'-ncct {str(ncct_path)} -mri {str(mri_path)} '\
                    f'-cfg {str(self.config_file_path)} -temp {self.temp_file} '
                subprocess.run(command, shell=True)
                (metric, files) = self.read_temp_file()
            else:
                metric, files = self.skull_stripper(ncct_path, mri_path)
            metrics.update(metric)
            new_files.extend(files)

        if 'tissue_segmentation' in self.gcfg['steps']:
            if self.command_line:
                command = f'python {str(self.tissue_segmenter_file)} '\
                    f'-ncct {str(ncct_path)} -mri {str(mri_path)} '\
                    f'-cfg {str(self.config_file_path)} -temp {self.temp_file} '
                subprocess.run(command, shell=True)
                (metric, files) = self.read_temp_file()
            else:
                metric, files = self.tissue_segmenter(ncct_path, mri_path)
            metrics.update(metric)
            new_files.extend(files)

        if 'registration' in self.gcfg['steps']:
            metric, files = self.registrator(ncct_path, mri_path)
            metrics.update(metric)
            new_files.extend(files)

        if 'intensity_clipping' in self.gcfg['steps']:
            files = self.intensity_clipper(ncct_path, mri_path)
            new_files.extend(files)

        if 'brain_region_crop' in self.gcfg['steps']:
            files = self.brain_region_cropper(ncct_path, mri_path)
            new_files.extend(files)

        if 'difference_image' in self.gcfg['steps']:
            files = self.difference_img_gen(ncct_path, mri_path)
            new_files.extend(files)

        if 'cleaning' in self.gcfg['steps']:
            files = self.cleaner(ncct_path, mri_path)

        return metrics, new_files
