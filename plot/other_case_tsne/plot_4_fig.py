import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob


def combine_images_with_labels(folder_path, output_path):
    """
    读取文件夹中的所有PNG图片，水平排列并在每张图正下方添加带括号的标签

    参数:
    folder_path: 字符串，包含PNG图片的文件夹路径
    output_path: 字符串，输出图片的保存路径
    """
    # 1. 读取文件夹中的所有PNG图片
    png_files = glob.glob(os.path.join(folder_path, "*.png"))

    # 如果没有找到PNG图片，则提示并退出
    if not png_files:
        print(f"在文件夹 {folder_path} 中未找到PNG图片")
        return

    # 按文件名排序，确保顺序一致
    png_files.sort()

    # 2. 创建子图布局
    n = len(png_files)
    fig, axes = plt.subplots(1, n, figsize=(n * 3, 4))  # 宽度根据图片数量调整

    # 如果只有一张图片，axes不是数组，需要转换为数组
    if n == 1:
        axes = [axes]

    # 3. 处理每张图片
    for i, (ax, img_path) in enumerate(zip(axes, png_files)):
        # 打开图片并转换为RGB数组
        img = Image.open(img_path)
        img_array = np.array(img)

        # 显示图片
        ax.imshow(img_array)

        # 移除坐标轴
        ax.axis('off')

        # 在图片正下方添加带括号的标签 (a), (b), (c)...
        label = f"({chr(97 + i)})"  # 97是'a'的ASCII码

        # 计算标签位置（图片底部中心）
        # 使用annotate方法将标签放在图片正下方
        ax.annotate(label,
                    xy=(0.5, 0),  # 图片中心底部
                    xycoords='axes fraction',  # 使用坐标轴比例
                    xytext=(0, -20),  # 向下偏移20像素
                    textcoords='offset points',
                    ha='center', va='top',  # 水平居中，垂直顶部对齐
                    fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))

    # 4. 调整布局
    plt.tight_layout()

    # 5. 保存合并后的图片
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"成功处理 {n} 张PNG图片")
    print(f"图片已生成并保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 指定包含PNG图片的文件夹路径
    folder_path = "D:\MYsource\diffusion_building_elec_load_exp\plot\other_case_tsne"  # 替换为你的实际路径

    # 指定输出图片路径
    output_path = "combined_images_with_labels.png"

    # 调用函数处理图片
    combine_images_with_labels(folder_path, output_path)