import os
import json
import random

# ===== 配置 =====
src_file = r'./data/train/new_train.json'  # 你的训练集文件
output_dir = r'./calib_texts'              # 输出文件夹
sample_size = 300                          # 抽样数量

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 读取数据
with open(src_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 打乱并抽样
random.shuffle(data)
samples = data[:sample_size]

# 保存成一个txt文件，每行一句
output_file = os.path.join(output_dir, 'calib_sentences.txt')
with open(output_file, 'w', encoding='utf-8') as f:
    for item in samples:
        text = item['content'].strip().replace('\n', ' ')
        f.write(text + '\n')

print(f'✅ 已生成校准数据：{output_file}（共 {len(samples)} 条）')
