# save_labels.py
# import json
# from transformers import AutoConfig
# cfg = AutoConfig.from_pretrained("./runs/distilbert_balanced")  # 你的已微调目录
# labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
# with open("labels.json", "w", encoding="utf-8") as f:
#     json.dump(labels, f, ensure_ascii=False)
# print("labels.json =", labels)

# save_labels.py
import json
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained(r"./runs/distilbert_balanced")
labels = [cfg.id2label[i] for i in range(cfg.num_labels)]
print("id2label from model:", labels)
with open("labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)