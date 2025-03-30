from pytask import task
from pytask import Product
from typing import Annotated
from pathlib import Path
from model.config import SRC
import subprocess


@task
def task_etl(path: Annotated[Path, Product] = SRC / "11_etl.py"):
    subprocess.run(["python",path], check=True)

@task
def task_dataprocessing(path: Annotated[Path, Product] = SRC / "25_process_data.py",after=task_etl):
    subprocess.run(["python", path], check=True)

@task
def task_optimize_slitting(path: Annotated[Path, Product] = SRC / "36_optimize_slitting.py",after=task_dataprocessing):
    subprocess.run(["python", path], check=True)

@task
def task_optimize_slitting_ind(path: Annotated[Path, Product] = SRC / "36_optimize_slitting_ind.py",after=task_dataprocessing):
    subprocess.run(["python", path], check=True)
    
@task
def task_merge_results(path: Annotated[Path, Product] = SRC / "41_merged_excel.py",after=task_optimize_slitting):
    subprocess.run(["python", path], check=True)

@task
def task_whole_flow_coil_slitting():
    """Run all three scripts sequentially."""
    task_etl()
    task_dataprocessing()
    task_optimize_slitting()
    task_merge_results()
