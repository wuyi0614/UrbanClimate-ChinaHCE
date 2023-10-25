# Replicable script for the first section of Results
#
# Created on 18 Oct 2023, by Yi
#

import pandas as pd

from pathlib import Path

# env variables
DATAFILE = Path('data') / 'rawdata-0125.csv'
CALCULATE_DATAFILE = Path('data') / 'CGSS-calculate-20231019.xlsx'


# functions
def get_cities(data: pd.DataFrame) -> list:
    """
    Return a list of city names

    :param data: the raw dataframe
    :return: list
    """

    return




if __name__ == '__main__':
    # test
    data = pd.read_csv(DATAFILE)
    pass