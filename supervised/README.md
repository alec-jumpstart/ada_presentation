0. test your python version `python --version`. Fairly sure it'll work for all versions > 3.0, but I'm on `3.7.8` for reference.

if your python version is less than 3.0, replace all `python` commands with `python3`
1. set up a virtualenv `python -m venv venv`
2. activate your virtualenv `source ./venv/bin/activate`
3. upgrade pip `python -m pip install --upgrade pip`
4. install the necessary libraries `pip install -r requirements.txt` (may be slow)
5. `python guess_digits.py`
