import ast
import json
import os
import pickle
import shutil
import time
from typing import Any, List, Union
import numpy as np
import pandas as pd


def save_jsonl(data: List[Any], filename: str, mode='w') -> None:
    """
    将数据保存为JSONL格式的文件。

    :param data: 要保存的数据，列表中的每个元素都应该是可序列化为JSON的对象。
    :param filename: 保存文件的名称。
    """
    with open(filename, mode, encoding='utf-8') as file:
        for entry in data:
            json_line = json.dumps(entry, ensure_ascii=False)  # 将对象转换为JSON字符串
            file.write(json_line + '\n')  # 写入文件，每个对象后换行


def read_jsonl(filename: str) -> List[Any]:
    """
    从JSONL格式的文件中读取数据。

    :param filename: JSONL文件的名称。
    :return: 包含文件中所有JSON对象的列表。
    """
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))  # 读取每行并转换为Python对象
    return data


def save_json(data: Any, filename: str) -> None:
    """
    将数据保存为JSON格式的文件。

    :param data: 要保存的数据，应该是可序列化为JSON的对象。
    :param filename: 保存文件的名称。
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)  # 将对象转换为JSON并保存到文件


def read_json(filename: str) -> Any:
    """
    从JSON格式的文件中加载数据。

    :param filename: JSON文件的名称。
    :return: 文件中的JSON对象。
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return json.load(file)  # 读取并转换整个文件的JSON内容


def save_text(data: str, filename: str, append: bool = False) -> None:
    """
    将字符串数据保存到文本文件中，可选择追加或覆盖写入。

    :param data: 要保存的字符串数据。
    :param filename: 保存文件的名称。
    :param append: 如果为 True，则以追加模式写入；否则覆盖写入。
    """
    mode = 'a' if append else 'w'
    with open(filename, mode, encoding='utf-8') as file:
        file.write(data)


def read_text(filename: str) -> str:
    """
    从文本文件中读取字符串数据。

    :param filename: 文本文件的名称。
    :return: 文件内容的字符串。
    """
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()  # 读取整个文件内容并返回


def save_numpy(data: np.ndarray, filename: str) -> None:
    """
    将NumPy数组保存到文件中（.npy格式）。

    :param data: 要保存的NumPy数组。
    :param filename: 保存文件的名称。
    """
    np.save(filename, data)


def read_numpy(filename: str) -> np.ndarray:
    """
    从文件中读取NumPy数组（.npy格式）。

    :param filename: 文件的名称。
    :return: 读取的NumPy数组。
    """
    return np.load(filename)


def save_feather(data: pd.DataFrame, filename: str) -> None:
    """
    将Pandas DataFrame保存为Feather格式。

    :param data: 要保存的Pandas DataFrame。
    :param filename: 保存文件的名称。
    """
    data.to_feather(filename)


def read_feather(filename: str) -> pd.DataFrame:
    """
    从Feather文件中读取Pandas DataFrame。

    :param filename: Feather文件的名称。
    :return: 读取的Pandas DataFrame。
    """
    return pd.read_feather(filename)


def save_pickle(data: Any, filename: str) -> None:
    """
    将数据保存为Pickle格式。

    :param data: 要保存的Python对象。
    :param filename: 保存文件的名称。
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def read_pickle(filename: str) -> Any:
    """
    从Pickle文件中加载数据。

    :param filename: Pickle文件的名称。
    :return: 文件中的Python对象。
    """
    with open(filename, 'rb') as file:
        return pickle.load(file)


def rmdir(fpath):
    """删除目录（如果存在）"""
    if os.path.exists(fpath):
        shutil.rmtree(fpath)


def remove_file(fname):
    """删除文件或目录"""
    if os.path.exists(fname):
        os.remove(fname)  # 删除文件


def wait_until_unlocked(action, waiting_msg, success_msg, fail_msg, max_wait=10):
    print(waiting_msg, end='\n→ ', flush=True)
    for _ in range(max_wait):
        try:
            action()
            print(f"\n✅ {success_msg}")
            return True
        except PermissionError:
            print('.', end='', flush=True)
            time.sleep(1)
        except FileNotFoundError as e:
            print(f"\n⚠️ 文件不存在：{e.filename}")
            return False
    print(f"\n❌ {fail_msg}")
    return False


def move_file_when_unlocked(fpath_src, fpath_dest, max_wait=10):
    return wait_until_unlocked(
        action=lambda: shutil.move(fpath_src, fpath_dest),
        waiting_msg=f"⏳ 正在尝试移动 \nFrom {fpath_src}\nTO {fpath_dest}",
        success_msg="文件转移成功！",
        fail_msg="超时，文件无法移动",
        max_wait=max_wait
    )


def delete_file_when_unlocked(fpath_target, max_wait=10):
    return wait_until_unlocked(
        action=lambda: os.remove(fpath_target),
        waiting_msg=f"⏳ 正在尝试删除 {fpath_target}",
        success_msg="文件删除成功！",
        fail_msg="超时，文件无法删除",
        max_wait=max_wait
    )


if __name__ == '__main__':
    pass

    # def get_all_functions_from_file(file_path):
    #     with open(file_path, 'r', encoding='utf-8') as f:
    #         source = f.read()
    #     tree = ast.parse(source)
    #     return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    #
    #
    # res = get_all_functions_from_file(__file__)
