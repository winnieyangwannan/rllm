# rllm
Get to O3 performance

## Install
```bash
pip install -e .
pip install -r requirements.txt
pip install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4/
```
New: 
```bash
cd rllm/verl
pip install -e .
pip install -r requirements.txt
pip install google-cloud-aiplatform latex2sympy2 pylatexenc sentence_transformers
```

### Download training/testing data from Google Drive
```bash
gdown "https://drive.google.com/uc?id=1q5Z0Xi98f1Zt-x4R3ubWLxkIIHsZPlum" -O "rllm/data/train/coding/apps.json"
gdown "https://drive.google.com/uc?id=1tAG36FB32ZLeUUckB6AHyEROkQ8lFhJ6" -O "rllm/data/train/coding/code_contests.json"
gdown "https://drive.google.com/uc?id=1K2kP8r8_jjGDbdwvTsRo2TEFEpJjMJxp" -O "rllm/data/train/coding/taco.json"
gdown "https://drive.google.com/uc?id=1ek936L0N57jVaF1YA0vCPv4GRWae4R5C" -O "rllm/data/train/coding/codeforces.json"
```
