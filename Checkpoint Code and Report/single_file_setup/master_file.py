import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.magics import extension

from common_setup import build_common_setup
from baseline import run_baseline
import argparse
from pathlib import Path
import shutil
import dbscan_full_func
import Isolation_Forest
from time import perf_counter

'''
Example usage:
master_file.py data.csv
'''

def master(file, type):
    start = perf_counter()
    def elapsed(start) -> float:
        return perf_counter() - start
    print('preparing space for data...')
    data_dir = Path.cwd() / "data"
    if data_dir.is_dir():
        shutil.rmtree(data_dir)
    print(f'processing data ({elapsed(start):.2f}s)')
    build_common_setup(raw_path=file, type=type) #add progress updates here
    print(f"done. starting analysis...".ljust(40), flush=True)

    print(f'running BASELINE (1/3) ({elapsed(start):.2f}s)')
    run_baseline() #add progress updates here
    print(f'running DBSCAN (2/3) ({elapsed(start):.2f}s)')
    dbscan_full_func.main()
    print(f'running ISOLATION FOREST (3/3) ({elapsed(start):.2f}s)')
    Isolation_Forest.main()
    print(f'\nanalysis finished. ({elapsed(start):.2f}s)')

parser = argparse.ArgumentParser()
parser.add_argument("input_file", type=Path, help="Path to csv or parquet file.")
args = parser.parse_args()

extension = args.input_file.suffix.lower()

if extension == ".csv":
    print('csv file recognized')
    master(args.input_file, "csv")
elif extension == ".parquet":
    print('parquet file recognized.')
    master(args.input_file, "parquet")
else:
    parser.error('Expected a .csv or .parquet file.')


