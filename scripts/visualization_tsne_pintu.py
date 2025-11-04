import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ==================== 配置参数 ====================
# 存放4张图的文件夹路径（请替换为你的文件夹路径）
image_folder = "../results/tsne/chosen"  # 与生成图的output_dir一致
# 输出拼接图的路径和文件名
output_path = "../results/tsne/merged_tsne_fig.png"
# 图片排序方式（按文件名排序，确保顺序为a→b→c→d）
# 若文件名包含数字（如sparsity 90→70→50→30），会自动按数字降序排列
# 若需自定义顺序，可手动指定图片文件名列表，如：image_names = ["img1.png", "img2.png", "img3.png", "img4.png"]
image_names = None  # 设为None则自动读取文件夹内图片

# 标注文字配置（a/b/c/d）
label_font = {
    "fontsize": 20,
    "weight": "bold",
    "color": "black",
    "bbox": dict(facecolor="white", edgecolor="black", pad=3, boxstyle="round,pad=0.3")
}


# ==================== 拼接函数 ====================
def merge_images_vertical(image_folder, output_path, image_names=None):
    # 1. 获取图片路径列表
    if image_names is None:
        # 自动读取文件夹内所有图片（支持png/jpg/jpeg格式）
        valid_ext = (".png", ".jpg", ".jpeg")
        image_names = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_ext)]
        # 按文件名排序（确保顺序正确，可根据实际文件名调整排序逻辑）
        image_names.sort(key=lambda x: int(''.join(filter(str.isdigit, x))), reverse=True)  # 按数字降序

    # 检查是否有4张图
    if len(image_names) != 4:
        raise ValueError(f"文件夹内图片数量不为4，当前为{len(image_names)}张")
    print(f"按顺序拼接图片：{image_names}")

    # 2. 读取所有图片
    images = []
    for name in image_names:
        img_path = os.path.join(image_folder, name)
        img = Image.open(img_path).convert("RGB")  # 转为RGB格式
        images.append(np.array(img))

    # 3. 创建拼接画布（4行1列）
    # 计算总高度（单张高度×4）和宽度（与单张宽度一致）
    h, w, _ = images[0].shape
    fig, axes = plt.subplots(4, 1, figsize=(w / 100, h * 4 / 100), dpi=300)  # 按像素比例设置尺寸
    fig.subplots_adjust(hspace=0.05)  # 调整子图间距

    # 4. 绘制图片并添加标注
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for i, (ax, img, label) in enumerate(zip(axes, images, labels)):
        ax.imshow(img)
        ax.axis("off")  # 关闭坐标轴
        # 在左上角添加标注（位置可根据需要调整）
        ax.text(0.03, 0.97, label, transform=ax.transAxes,
                ha="left", va="top", **label_font)

    # 5. 保存拼接图
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    print(f"拼接图已保存至：{output_path}")
    plt.close()


# ==================== 执行拼接 ====================
if __name__ == "__main__":
    merge_images_vertical(image_folder, output_path, image_names)