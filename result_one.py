# Replicable script for the first section of Results
#
# Created on 18 Oct 2023, by Yi
#

import json
import pandas as pd

from pathlib import Path


# env variables
MAPPINGFILE = Path('data') / 'mapping.json'
MAPPING = json.loads(MAPPINGFILE.read_text(encoding='utf8'))


# functions
def get_cities(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform cities in Chinese into English

    :param data: the raw dataframe
    :return: an updated dataframe
    """
    city = MAPPING['city']
    data['prefecture_en'] = data['prefecture'].apply(lambda x: city.get(x, 'other'))
    return data


def get_percap(data: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Transform selected data into percap data

    :param data: the raw dataframe
    :param columns: columns for percap transformation
    :return: an updated dataframe
    """
    for c in columns:
        data[f'{c}_percap'] = data[c] / data['size']

    return data


def get_energy(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform energy consumption data into a well-shaped dataset

    :param data: the raw energy consumption dataframe
    :return: the pivot-processed dataframe
    """
    data['type'] = data['type'].apply(lambda x: f'en_{x}')
    merged = data[['id', 'type', 'use']].groupby(['id', 'type']).sum().reset_index()
    pivot = merged.pivot(index='id', columns='type', values='use').fillna(0)
    pivot['en_total'] = pivot.iloc[:, 1:].sum(axis=1)  # sum by rows
    return pivot


def analyse(data: pd.DataFrame) -> pd.DataFrame:


    return


if __name__ == '__main__':
    # merging data and test
    datafile = Path('data') / 'vardata-1025.xlsx'
    data = pd.read_excel(datafile, engine='openpyxl')
    engfile = Path('data') / 'energyuse-1024.xlsx'
    eng = pd.read_excel(engfile, engine='openpyxl')
    eng = get_energy(eng)

    merged = data.merge(eng, on='id', how='left')
    # preprocess
    merged = get_cities(merged)
    merged = get_percap(merged, columns=['en_total'])


