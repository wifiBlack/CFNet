#!/bin/bash

# 检查是否提供了两个参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_txt>"
    exit 1
fi

input_folder="$1"
output_txt="$2"

# 确保输入文件夹存在
if [ ! -d "$input_folder" ]; then
    echo "Error: Directory '$input_folder' does not exist."
    exit 1
fi

# 清空或创建输出txt文件
> "$output_txt"

# 遍历文件夹中的所有.png文件并写入文件名（不含后缀）到txt
for file in "$input_folder"/*.png; do
    if [ -f "$file" ]; then
        filename=$(basename "$file" .png)
        echo "$filename" >> "$output_txt"
    fi
done

echo "File names written to $output_txt."
