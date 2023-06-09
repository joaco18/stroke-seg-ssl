# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

SIMPLIFY_SYNTHSEG = {
    0: 0,    # background

    2: 1,    # left hem - left cerebral white matter
    3: 1,    # left hem - left cerebral cortex

    4: 2,    # ventricles - left lateral ventricle
    5: 2,    # ventricles - left inferior lateral ventricle

    7: 3,    # left cerebellum - left cerebellum white matter
    8: 3,    # left cerebellum - left cerebellum cortex

    10: 4,   # left basal_nuclei - left thalamus
    11: 4,   # left basal_nuclei - left caudate
    12: 4,   # left basal_nuclei - left putamen
    13: 4,   # left basal_nuclei - left pallidum

    14: 2,   # ventricles
    15: 2,   # ventricles
    16: 6,   # brain-stem

    17: 1,   # left hem - left hippocampus
    18: 1,   # left hem - left amygdala
    24: 11,  # csf
    26: 1,   # left hem - left accumbens area
    28: 1,   # left hem - left ventral DC

    41: 7,   # right hem - right cerebral white matter
    42: 7,   # right hem - right cerebral cortex

    43: 2,   # ventricles - right lateral ventricle
    44: 2,   # ventricles - right inferior lateral ventricle

    46: 9,   # right cerebellum - right cerebellum white matter
    47: 9,   # right cerebellum - right cerebellum cortex

    49: 10,  # right basal_nuclei - right thalamus
    50: 10,  # right basal_nuclei - right caudate
    51: 10,  # right basal_nuclei - right putamen
    52: 10,  # right basal_nuclei - right pallidum

    53: 7,   # right hem - right hippocampus
    54: 7,   # right hem - right amygdala
    58: 7,   # right hem - right accumbens area
    60: 7,   # right hem - right ventral DC
}
