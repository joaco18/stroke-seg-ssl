# -*-coding:utf-8 -*-
'''
@Time    :   2023/06/09 12:28:28
@Author  :   Joaquin Seia
@Contact :   joaquin.seia@icometrix.com
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def modified_bland_altman_plot(
    data1: np.ndarray, data2: np.ndarray, title: str = '',
    unit='mm3', y_lim=None, x_lim=None, *args, **kwargs
) -> None:
    # Get the value
    title = f' {title}' if (title != '') else ''
    diff = data1 - data2
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    ci_low = md - 1.96 * sd
    ci_high = md + 1.96 * sd

    # Generate scatter plot
    _, ax = plt.subplots()
    ax.scatter(mean, diff, *args, **kwargs)
    # sns.scatterplot(data=None, x=mean, y=diff, size=9, ax=ax)
    ax.set_xlabel(f'Means [{unit}]')
    ax.set_ylabel(f'Difference [{unit}]')
    ax.set_title(f"Volumes Bland-Altman Plot GT vs. Prediction\n{title}")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    if x_lim is not None:
        ax.set_xlim(x_lim)

    # Add reference lines and text
    ax.axhline(md, color='gray', linestyle='--')
    ax.axhline(ci_high, color='gray', ls='--')
    ax.axhline(ci_low, color='gray', ls='--')
    text_x_loc = np.min(mean) + (np.max(mean)-np.min(mean))*1.14
    ax.text(text_x_loc, ci_low, f'-1.96SD:\n{ci_low:.2f}{unit}', ha="center", va="center")
    ax.text(text_x_loc, ci_high, f'+1.96SD:\n{ci_high:.2f}{unit}', ha="center", va="center")
    ax.text(text_x_loc, md, f'MD:\n{md:.2f}{unit}', ha="center", va="center")
    plt.subplots_adjust(right=0.85)
    sns.despine()
    plt.show()
    return ax
