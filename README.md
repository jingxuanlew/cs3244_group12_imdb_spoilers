# Installation and Running of Project

## Pre-Installation
This project uses python 3.13. If you do not already have python installed, please feel free to follow the following instructions

1. (If not already installed) Install [homebrew](https://brew.sh/) using the instructions at the link

2. (Optional) Verify homebrew is installed
```bash
brew --version
```

3. Install python using homebrew
```bash
brew install python@3.13
```

4. (Optional) Verify python 3.13 is installed
```bash
python3.13 --version
```

## First Installation

The packages we used are stored in [requirements.txt](requirements.txt). To install these packages, after cloning this repository, follow the steps:  

1. Create a virtual environment using python 3.13
```bash
python3.13 -m venv .venv
```

2. Activate virtual environment (note: once environment is activated, we use python instead of python3)
```bash
source .venv/bin/activate
```

3.  Install necessary packages based on requirements.txt
```bash
python -m pip install -r requirements.txt
```

4. (Optional) Check all the packages you have installed by running
```bash
python -m pip list
```

5. (Optional) Verify that your python is pointing to your virtual environment by running
```bash
which python
```
This should point to something ending with `.venv/bin/python`

## Adding New Package(s)

If you would like to add a new package e.g. pytest, follow these steps (on a new branch)

1. Activate virtual environment (if not already activated)
```bash
source .venv/bin/activate
```

2.  Install latest packages based on requirements.txt
```bash
python -m pip install -r requirements.txt
```

3. Install new package
```bash
python -m pip install pytest
```

4. Freeze current packages into requirements.txt
```bash
python -m pip freeze > requirements.txt
```

Then create a pull request for approval.

## Running Project Normally

For normal development (post-installation), feel free to just run this before beginning work:

1. Activate virtual environment
```bash
source .venv/bin/activate
```

2. (Optional) If you know new packages have been added by others, run this:
```bash
python -m pip install -r requirements.txt
```
