import os
from PIL import Image
import sys

def split_image_with_overlap(image_path, output_folder):
    """将一张1024x1024的图片切割成25张256x256的图片，每张图片之间有64像素的重叠，并保存到输出文件夹"""
    img = Image.open(image_path)
    img_width, img_height = img.size

    if img_width != 1024 or img_height != 1024:
        print(f"跳过图片 {image_path}，因为它不是1024x1024大小")
        return
    
    base_name = os.path.basename(image_path).split('.')[0]  # 获取图片的基础名称（无扩展名）

    step_size = 192  # 每个窗口的步长
    patch_size = 256  # 每张子图的尺寸
    count = 0

    # 切割图片，步长为192，生成25张子图
    for i in range(0, img_width - patch_size + 1, step_size):
        for j in range(0, img_height - patch_size + 1, step_size):
            # 定义切割的区域
            box = (i, j, i + patch_size, j + patch_size)
            part = img.crop(box)
            
            # 保存切割后的图片
            output_path = os.path.join(output_folder, f"{base_name}_{count}.png")
            part.save(output_path)
            count += 1

    print(f"图片 {image_path} 切割完成，保存到 {output_folder}")

def process_folder(input_folder, output_folder):
    """处理输入文件夹中的所有1024x1024 PNG图片"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(input_folder, filename)
            split_image_with_overlap(image_path, output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法: python split_images_with_overlap.py <输入文件夹路径> <输出文件夹路径>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    process_folder(input_folder, output_folder)
