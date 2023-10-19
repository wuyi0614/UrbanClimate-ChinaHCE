# Replicable script for the data integration
#
# Created on 19 Oct 2023, by Yi
#


if __name__ == '__main__':
    import pandas as pd
    from pathlib import Path

    # load calculated data
    cal_datafile = Path('data') / 'CGSS-calculate-20231019.xlsx'
    calculate = pd.read_excel(cal_datafile, engine='openpyxl')


    pass
