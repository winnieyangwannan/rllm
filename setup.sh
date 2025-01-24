#!/bin/bash

# download coding datasets for training
gdown "https://drive.google.com/uc?id=1q5Z0Xi98f1Zt-x4R3ubWLxkIIHsZPlum" -O "rllm/data/train/coding/apps.json"
gdown "https://drive.google.com/uc?id=1tAG36FB32ZLeUUckB6AHyEROkQ8lFhJ6" -O "rllm/data/train/coding/code_contests.json"
gdown "https://drive.google.com/uc?id=1K2kP8r8_jjGDbdwvTsRo2TEFEpJjMJxp" -O "rllm/data/train/coding/taco.json"