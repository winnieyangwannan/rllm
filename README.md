# RLLM - Democratizing Reinforcement Learning for Language Models

## Installation

```bash
# Fetch both rllm and verl.
git clone --recurse-submodules https://github.com/agentica-project/rllm-internal.git
# Install dependencies.
cd rllm
pip install -e ./verl[vllm,gpu,sglang]
pip install -e .
pip install -r requirements.txt
```
```

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


## ToDO for release
- [ ] Check all training scripts in the examples to make sure they are all runnable