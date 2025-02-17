export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

bash scripts/train/debug_code_dataloader.sh --model ~/models/DeepSeek-R1-Distill-Qwen-1.5B  > code_contests.log 2>&1 