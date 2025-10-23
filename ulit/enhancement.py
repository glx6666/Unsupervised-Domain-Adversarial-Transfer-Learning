import os
import shutil

# 根目录
root_path = r"E:\数据\dataset\new_shanke_zhoucheng"

# 要删除的文件夹名称
folders_to_delete = {'2', '3', '5', '6', '8', '9'}

for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
    # 检查当前路径下是否包含要删除的文件夹
    for folder in list(dirnames):
        if folder in folders_to_delete:
            folder_path = os.path.join(dirpath, folder)
            print(f"删除文件夹: {folder_path}")
            shutil.rmtree(folder_path, ignore_errors=True)

# 再次遍历，用于重命名排序
for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
    # 只处理有多个数字命名的子文件夹的目录
    numeric_dirs = [d for d in dirnames if d.isdigit()]
    if numeric_dirs:
        numeric_dirs.sort(key=lambda x: int(x))  # 排序
        for idx, d in enumerate(numeric_dirs):
            old_path = os.path.join(dirpath, d)
            new_path = os.path.join(dirpath, str(idx))
            if old_path != new_path:
                print(f"重命名: {old_path} -> {new_path}")
                os.rename(old_path, new_path)
