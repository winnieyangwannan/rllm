export RAY_TMPDIR=/data/xiaoxiang/rllm/tmp
#DeepSeek-R1-Distill-Qwen-1.5B livecodebench
bash ./scripts/eval/eval_livecodebench.sh --model /data/xiaoxiang/DeepSeek-R1-Distill-Qwen-1.5B  --datasets livecodebench --output-dir /data/xiaoxiang/eval/DeepSeek-R1-Distill-Qwen-1.5B > eval.log 2>&1

#DeepScaleR-1.5B-Preview livecodebench
#./scripts/eval/eval_livecodebench.sh --model /data/xiaoxiang/DeepScaleR-1.5B-Preview --datasets livecodebench --output-dir /data/xiaoxiang/eval/DeepScaleR-1.5B-Preview
