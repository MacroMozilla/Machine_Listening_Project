import os
import concurrent.futures
from functools import partial
from typing import Any, Callable, Dict, List
from tqdm import tqdm

from f_utility.const import TypeR
from f_utility.watch import watch_time, Watch

# 定义常量
K_args = "args"
K_kwargs = "kwargs"



@watch_time
def batch_execute_forloop(func: Callable[..., TypeR], args_kwargs_s: List[Dict[str, Any]]) -> List[TypeR]:
    """
    使用普通 for 循环逐个执行函数，并显示进度条。
    """
    results: List[TypeR] = []
    for arg_kwargs in tqdm(args_kwargs_s, total=len(args_kwargs_s), desc=f"forloop {func.__name__}"):
        args = arg_kwargs.get(K_args, ())
        kwargs = arg_kwargs.get(K_kwargs, {})
        results.append(func(*args, **kwargs))
    return results


def _execute_with_args_kwargs(func: Callable[..., TypeR], arg_kwargs: Dict[str, Any]) -> TypeR:
    """
    辅助函数，用于多进程环境下执行 func。
    """
    args = arg_kwargs.get(K_args, ())
    kwargs = arg_kwargs.get(K_kwargs, {})
    return func(*args, **kwargs)


@watch_time
def batch_execute_process(func: Callable[..., TypeR], args_kwargs_s: List[Dict[str, Any]], workers=None) -> List[TypeR]:
    """
    使用多进程池并行执行函数，并显示进度条。
    """
    if workers is None:
        workers = os.cpu_count() - 2

    factor = 2
    chunksize = max(1, len(args_kwargs_s) // (workers * factor))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        partial_func = partial(_execute_with_args_kwargs, func)
        results_iterator = executor.map(partial_func, args_kwargs_s, chunksize=chunksize)
        results: List[TypeR] = list(tqdm(results_iterator, total=len(args_kwargs_s), desc=f"process {func.__name__}"))
    return results


@watch_time
def batch_execute_thread(func: Callable[..., TypeR], args_kwargs_s: List[Dict[str, Any]], workers=None) -> List[TypeR]:
    """
    使用多线程池并行执行函数，并显示进度条。
    """
    if workers is None:
        workers = os.cpu_count() - 2

    factor = 4
    chunksize = max(1, len(args_kwargs_s) // (workers * factor))

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        partial_func = partial(_execute_with_args_kwargs, func)
        results_iterator = executor.map(partial_func, args_kwargs_s, chunksize=chunksize)
        results: List[TypeR] = list(tqdm(results_iterator, total=len(args_kwargs_s), desc=f"thread {func.__name__}"))
    return results


if __name__ == "__main__":
    # 定义三种执行方式
    pass
