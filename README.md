# rllm

Get to O3 performance

## Install

Install Verl Submodule:

```bash
git submodule init
git submodule update
```

Install Python Dependencies:

```bash
pip install -r ./verl/requirements.txt
pip install -e ./verl
pip install google-cloud-aiplatform latex2sympy2 pylatexenc sentence_transformers
pip install -e .
# Don't forget to install this, or training run will crash!
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


Train your Math/Coding Agents
```bash
cd scripts/data
python3 deepscaler_dataset.py
python3 code_dataset.py
```