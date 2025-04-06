import os


def list_all_dirs(base_path, level=0):
    try:
        for entry in os.listdir(base_path):
            full_path = os.path.join(base_path, entry)
            if os.path.isdir(full_path):
                print("    " * level + f"📁 {entry}")
                list_all_dirs(full_path, level + 1)
    except PermissionError:
        print("    " * level + f"🚫 无权限访问: {base_path}")
    except FileNotFoundError:
        print(f"❌ 目录不存在: {base_path}")
