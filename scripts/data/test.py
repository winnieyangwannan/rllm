from pathlib import Path

# 获取当前脚本的路径
current_script_path = Path(__file__).resolve()

# 推导出项目根目录（假设项目根目录是当前脚本的父目录的父目录）
project_root = current_script_path.parent.parent

# 替换路径
original_path = "/data/xiaoxiang/rrllm/data"
new_path = str(project_root / "rrllm/data")

print(f"Original Path: {original_path}")
print(f"New Path: {new_path}")