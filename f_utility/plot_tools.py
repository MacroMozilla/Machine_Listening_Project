import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

# 设置 matplotlib 后端和风格
matplotlib.use('TkAgg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('Solarize_Light2')


def plt_img(img_XxHxW: torch.Tensor, title: str = '', ax=None):
    """
    显示单张图片。支持 torch.Tensor 或 numpy 数组，
    若图像为 3D tensor，则假定通道维度在最前面，自动转换为 (H, W, C)。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if len(img_XxHxW.shape) == 3:
        img_XxHxW = img_XxHxW.permute(1, 2, 0)

    if isinstance(img_XxHxW, torch.Tensor):
        ax.imshow(img_XxHxW.detach().cpu().numpy())
    else:
        ax.imshow(img_XxHxW)

    ax.set_title(title)
    ax.grid(False)


def plt_imgs(imgs: list, titles: list = None, row_col: tuple = None):
    """
    使用 plt_imgshow 显示多张图片。

    参数:
    - imgs: 图片列表（每个元素为 torch.Tensor 或 numpy 数组）
    - titles: 每张图的标题列表，若未提供则默认为空字符串
    - row_col: 网格布局 (rows, cols)。若为 None，则根据图片数量自动选择：
        1 张图  -> 1×1
        2 张图  -> 2×1
        3 张图  -> 3×1
        4 张图  -> 2×2
        5 张图  -> 3×2
        其它数量 -> 近似正方形的布局
    """
    n = len(imgs)
    if row_col is None:
        if n == 1:
            row_col = (1, 1)
        elif n == 2:
            row_col = (2, 1)
        elif n == 3:
            row_col = (3, 1)
        elif n == 4:
            row_col = (2, 2)
        elif n == 5:
            row_col = (3, 2)
        else:
            rows = int(np.ceil(np.sqrt(n)))
            cols = int(np.ceil(n / rows))
            row_col = (rows, cols)

    rows, cols = row_col
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 10, rows * 10))

    # 如果只有一个子图则转换为列表，便于后续统一处理
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    else:
        axes = axes.flatten()

    # 准备标题列表
    if titles is None:
        titles = [''] * n
    elif len(titles) < n:
        titles = titles + [''] * (n - len(titles))

    # 显示每张图片
    for i, img in enumerate(imgs):
        if img is not None:
            plt_img(img, title=titles[i], ax=axes[i])

    # 隐藏多余的子图
    for j in range(n, len(axes)):
        axes[j].axis('off')

    # plt.tight_layout()


plt_show = lambda: plt.show(block=True)
