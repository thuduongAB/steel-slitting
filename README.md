# Column Generation for the Cutting Stock Problem

The Cutting Stock Problem deals with the problem of cutting stock material with the same, fixed width into smaller pieces, according to a set of orders specifying both the widths and the demand (weight) requirements, so as to minimize the amount of wasted material.

This repository demonstrates how to set up a Python environment using an `environment.yml` file and run Python tasks with `pytask`.

## Project Directory

steel-slitting
│
├───.pytask
│
├───data
│   ├───SlitRequestSRO-1117.csv
│   └───Stock End Details_Jan.xlsx
│
├───src
│   ├───model
│   │   ├───__init__.py
│   │   ├───config.py
│   │   ├───O41_dual_solver.py
│   │   ├───<...>.py
│   │   └───config.py
│   │
│   ├───11_etl.py
│   ├───<...>.py
│   └───task_coil_steel_slitting.py
│
└───pyproject.toml

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setting Up the Environment](#setting-up-the-environment)
- [Running PyTask](#running-pytask)
---
## Prerequisites

Before setting up the environment, make sure you have the following installed:

- **[PuLP](https://coin-or.github.io/pulp/)** (for managing optimization problem)
- **[Python](https://www.python.org/downloads/)** (version 3.12)
- **[PyTask](pytask-dev.readthedocs.io/)** (Python task automation)

---

## Setting Up the Environment and Run the project
1. **Create a Conda environment** using the provided `environment.yml` file.

   This file contains the necessary dependencies for the project, including `pytask` and any other libraries you may need.

   ```bash
   conda env create -f environment.yml
   ```

### Run the project:
1. **Save data**
- Input data in the `data` folder.

2. **Save ENV VARIABLES**
- Ensure that you have a correct env  `.env` if there is changed parameters [UAT]

3. **Run Task(s)**
   ```bash
   pytask -k task_etl
   ```

## Running PyTask

1. **PyTask Installation**
   You can verify that pytask is installed by running:
   ```bash
   pytask --version
   pytask -h | --help
   ```

   The build command is the default command, meaning the following two calls are identical:
   ```bash
   pytask
   pytask build
   ```
2. **PyTask File**
   The task, task_optimize_slitting and task_whole_flow_coil_slitting, are be defined in src/task_coil_steel_slitting.py, and it will generate a optimization results stored in /results.

   The task_ prefix for modules and task functions is important so that pytask automatically discovers them.

3. **Run PyTask**
   Run tasks with pytask:
   ```bash
   pytask -k task_run_coil_slitting
   ```
   Or run individual tasks:
   ```bash
   pytask -k task_etl
   pytask -k task_optimize_slitting
   ```

