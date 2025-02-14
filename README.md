# rllm

Get to O3 performance

## Install

<<<<<<< HEAD
Install Verl Submodule:
=======

> > > > > > > 1af5909 (pytests added and addresses comments)

```bash
git submodule init
git submodule update
```

<<<<<<< HEAD
Install Python Dependencies:
=======
New:

> > > > > > > 1af5909 (pytests added and addresses comments)

```bash
pip install -e ./verl
pip install -e .
pip install google-cloud-aiplatform latex2sympy2 pylatexenc sentence_transformers
```

### Download training/testing data from Google Drive

```bash
gdown "https://drive.google.com/uc?id=1q5Z0Xi98f1Zt-x4R3ubWLxkIIHsZPlum" -O "rllm/data/train/coding/apps.json"
gdown "https://drive.google.com/uc?id=1tAG36FB32ZLeUUckB6AHyEROkQ8lFhJ6" -O "rllm/data/train/coding/code_contests.json"
gdown "https://drive.google.com/uc?id=1K2kP8r8_jjGDbdwvTsRo2TEFEpJjMJxp" -O "rllm/data/train/coding/taco.json"
gdown "https://drive.google.com/uc?id=1ek936L0N57jVaF1YA0vCPv4GRWae4R5C" -O "rllm/data/train/coding/codeforces.json"
```

### WandB

````bash
wandb login
wandb init
### Run Unit Tests

Unit tests are in the `test/` folder and uses pytest. To run them, you can do something like this.

```bash
PYTHONPATH=. pytest tests/rllm/rewards/tests.py
````
