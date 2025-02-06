set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_eval \
    data.path=$HOME/aime.parquet \
