from collections import Counter
import pandas as pd
df = pd.read_csv("./external/simplifyweibo_4_moods.csv", encoding="utf-8")
print(Counter(df["label"].astype(str)))  # 看是否严重失衡或只剩一种
print("unique labels:", sorted(df["label"].astype(str).unique()))
