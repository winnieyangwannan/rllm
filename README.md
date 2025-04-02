# RLLM - Democratizing Reinforcement Learning for Language Models

## Installation

```bash
# Fetch both rllm and verl.
git clone --recurse-submodules https://github.com/deepscaler/rllm.git
# Install dependencies.
cd rllm
pip install -e ./verl
pip install -e .
pip install -r requirements.txt
```

## Dataset Setup

```bash
cd scripts/data
# Download large datasets.
python download_datasets.py
# Math Datasets
python deepscaler_dataset.py
# Code Ddatasets
python code_dataset.py
```

Install Python Dependencies:

```bash
pip install -r ./verl/requirements.txt
pip install -e ./verl
pip install google-cloud-aiplatform latex2sympy2 pylatexenc sentence_transformers
pip install -e .
# Don't forget to install this, or training run will crash!
```


## Train
```bash
cd scripts/train
# 8k training run, for example.
./run_deepscaler_1.5b_8k.sh --model agentica-org/DeepScaleR-1.5B-Preview
```


### WandB

````bash
wandb login
wandb init
### Run Unit Tests

Unit tests are in the `test/` folder and uses pytest. To run:

```bash
PYTHONPATH=. pytest tests/rllm/rewards/tests.py
````

### WebAgent 

#### BrowserGym setup
Setup playwright by running
```bash
pip install playwright
playwright install chromium
```

Then follow instruction for each specific environment
##### MiniWob
```bash
git clone git@github.com:Farama-Foundation/miniwob-plusplus.git
git -C "./miniwob-plusplus" reset --hard 7fd85d71a4b60325c6585396ec4f48377d049838
export MINIWOB_URL="file://<PATH_TO_MINIWOB_PLUSPLUS_CLONED_REPO>/miniwob/html/miniwob/"

cd rllm
python scripts/data/miniwob_dataset.py --local_dir ~/data/rllm-miniwob
```