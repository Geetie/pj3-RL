
"""
找到最新的训练模型路径
"""
import os
import argparse


def get_latest_model_path(exp_name: str = "MsPacman-v5"):
    """
    找到最新的模型路径
    """
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        print(f"[错误] runs/ 目录不存在")
        return None
    
    # 列出所有子目录
    subdirs = []
    for d in os.listdir(runs_dir):
        full_path = os.path.join(runs_dir, d)
        if os.path.isdir(full_path) and exp_name in d:
            subdirs.append(full_path)
    
    if not subdirs:
        print(f"[错误] 找不到包含 '{exp_name}' 的实验目录")
        return None
    
    # 按修改时间排序，最新的在最后
    subdirs.sort(key=lambda x: os.path.getmtime(x))
    latest_dir = subdirs[-1]
    
    # 在该目录下找 .pth 文件
    for f in os.listdir(latest_dir):
        if f.endswith(".pth"):
            model_path = os.path.join(latest_dir, f)
            print(f"[成功] 找到最新模型:")
            print(f"       {model_path}")
            return model_path
    
    print(f"[错误] 在 {latest_dir} 中找不到 .pth 模型文件")
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="MsPacman-v5", help="实验名称")
    args = parser.parse_args()
    
    latest_model = get_latest_model_path(args.exp_name)
    
    if latest_model:
        print(f"\n录制视频命令:")
        print(f"python record_video.py --model-path {latest_model} --num-episodes 3")

