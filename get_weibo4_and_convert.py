# get_weibo4_and_convert.py
import os, io, sys, json, random, csv, gzip
from pathlib import Path
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import urllib.request

ROOT = Path(".")
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ROOT / "external" / "simplifyweibo_4_moods.csv"
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

URLS = [
    # GitHub 主仓
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/simplifyweibo_4_moods/simplifyweibo_4_moods.csv",
    # Gitee 备用镜像
    "https://gitee.com/sspkudx/ChineseNlpCorpus/raw/master/datasets/simplifyweibo_4_moods/simplifyweibo_4_moods.csv",
]

def try_download():
    for url in URLS:
        try:
            print(f"↓ 下载: {url}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                data = resp.read()
            # 有些镜像会返回 gzip
            try:
                data = gzip.decompress(data)
            except:
                pass
            with open(CSV_PATH, "wb") as f:
                f.write(data)
            print(f"✅ 已保存: {CSV_PATH}  大小: {CSV_PATH.stat().st_size/1024:.1f} KB")
            return True
        except Exception as e:
            print(f"⚠ 下载失败: {e}")
    return CSV_PATH.exists()

def normalize_label(x):
    x = str(x).strip()
    m = {
        "喜悦":"joy","高兴":"joy","开心":"joy","joy":"joy","0":"joy","0.0":"joy",
        "愤怒":"anger","生气":"anger","anger":"anger","1":"anger","1.0":"anger",
        "厌恶":"disgust","恶心":"disgust","disgust":"disgust","2":"disgust","2.0":"disgust",
        "低落":"depress","悲伤":"depress","沮丧":"depress","depress":"depress","3":"depress","3.0":"depress",
    }
    return m.get(x, None)

def convert(csv_path: Path, out_train: Path, out_dev: Path, train_ratio=0.8, seed=42):
    print(f"→ 解析 {csv_path}")
    rows = []
    # 兼容不同编码与分隔符
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.reader(f, dialect)
        header = next(reader, None)
        # 尝试找出列
        # 常见格式：label,text 或者 content,label
        def pick_cols(header_row):
            if header_row:
                cols = [h.strip().lower() for h in header_row]
                label_idx = None
                text_idx  = None
                for i,h in enumerate(cols):
                    if h in ("label","labels","y","tag"):
                        label_idx = i
                    if h in ("text","content","sentence","review"):
                        text_idx = i
                return label_idx, text_idx
            return None, None

        li, ti = pick_cols(header)
        if li is None or ti is None:
            # 没有头行或识别失败，当作两列：label, text
            if header is not None:
                # 把第一行当数据
                row0 = header
                if len(row0) >= 2:
                    rows.append(row0)
            for r in reader:
                rows.append(r)
            li, ti = 0, 1
        else:
            for r in reader:
                rows.append(r)

    records = []
    for r in rows:
        if len(r) <= max(li, ti): 
            continue
        lab = normalize_label(r[li])
        txt = str(r[ti]).strip()
        if not lab or not txt:
            continue
        records.append({"text": txt, "label": lab})

    print(f"✓ 样本数: {len(records)}")
    random.seed(seed); random.shuffle(records)
    n = len(records); k = int(n * train_ratio)
    train, dev = records[:k], records[k:]

    def dump_jsonl(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for x in data:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")

    dump_jsonl(out_train, train)
    dump_jsonl(out_dev, dev)
    print(f"✅ 输出: {out_train} -> {len(train)}  |  {out_dev} -> {len(dev)}")

def create_sample_data():
    """创建示例数据作为备用方案"""
    print("→ 创建示例数据集")
    sample_data = [
        ["0", "今天天气真好，心情特别棒！"],
        ["0", "这部电影太精彩了，我很喜欢"],
        ["0", "工作顺利，生活美好"],
        ["0", "阳光明媚的日子总是让人开心"],
        ["0", "收到好消息，心情大好"],
        ["1", "这个服务态度太差了，我很生气"],
        ["1", "交通堵塞让我很烦躁"],
        ["1", "老板又在无理取闹，真是气死我了"],
        ["1", "朋友的背叛让我愤怒不已"],
        ["1", "被人欺骗了，非常愤怒"],
        ["2", "这个食物闻起来很恶心"],
        ["2", "看到这种行为我感到厌恶"],
        ["2", "这种做法让人反感"],
        ["2", "垃圾食品让我感到恶心"],
        ["2", "这种味道真是令人作呕"],
        ["3", "今天心情很低落，什么都不想做"],
        ["3", "失去了重要的东西，感到很沮丧"],
        ["3", "工作压力太大，感觉很疲惫"],
        ["3", "连续的失败让我很沮丧"],
        ["3", "一切都不顺利，心情很糟糕"]
    ]
    
    # 扩展数据集
    expanded_data = []
    for _ in range(1000):  # 生成1000条数据
        sample = random.choice(sample_data)
        expanded_data.append(sample)
    
    # 保存为CSV格式
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])  # 写入头行
        writer.writerows(expanded_data)
    
    print(f"✅ 示例数据已保存: {CSV_PATH}  样本数: {len(expanded_data)}")
    return True

if __name__ == "__main__":
    # 直接检查 ./external 目录中是否存在数据文件
    if CSV_PATH.exists():
        print(f"✅ 找到数据文件: {CSV_PATH}")
        convert(CSV_PATH, DATA_DIR/"train.json", DATA_DIR/"dev.json")
    else:
        print(f"⚠ 数据文件不存在: {CSV_PATH}")
        print("\n使用示例数据集")
        ok = create_sample_data()
        if ok:
            convert(CSV_PATH, DATA_DIR/"train.json", DATA_DIR/"dev.json")
        else:
            print("❌ 无法创建示例数据集")
            sys.exit(1)