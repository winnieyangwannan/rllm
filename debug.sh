export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

bash scripts/train/debug_code_dataloader.sh --model ~/models/DeepSeek-R1-Distill-Qwen-1.5B  > apps.log 2>&1 


# ps -aux | grep "actor_rollout" | grep -v grep | awk '{print $2}' | xargs kill -9


# ps -aux | grep "main_ppo" | grep -v grep | awk '{print $2}' | xargs kill -9