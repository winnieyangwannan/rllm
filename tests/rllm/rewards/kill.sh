ps -aux | grep "tests_taco.py" | grep -v grep | awk '{print $2}' | xargs kill -9

ps -aux | grep "coder1.py" | grep -v grep | awk '{print $2}' | xargs kill -9


# nohup python3 examples/data_preprocess/coder1.py > coder1.log 2>&1 &

# nohup python3 coder1.py > coder1.log 2>&1 &