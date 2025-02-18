bash ./scripts/eval/eval_lcb_model.sh  > eval_lcb.log 2>&1  &

# # ps -aux | grep "main_generation" | grep -v grep | awk '{print $2}' | xargs kill -9

# ps -aux | grep "main_task" | grep -v grep | awk '{print $2}' | xargs kill -9