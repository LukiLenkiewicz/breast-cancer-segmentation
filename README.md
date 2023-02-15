# machine-learning/BreastCancer-ML
## Installation

1. Create new virtual environment:
    If you use _conda_:
    ```
    conda create --name your-environment-name python=3.9
    ```
    Alternatively use any other virtual enviroment manager of your choice.

2. Activate environment
    ```
    conda activate your-environment-name
    ```
3. Make sure you use recent _pip_ version:
    ```
    python -m pip install --upgrade pip
    ```
4. Install packages:

    ```
    python -m pip install -e .[dev] --extra-index-url https://download.pytorch.org/whl/cu116
    ```
5. Enable pre-commit
    ```
    pre-commit install
    ```
6. Enjoy