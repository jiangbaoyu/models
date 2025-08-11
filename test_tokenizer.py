from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("./TinyBERT")
print(len(tok), tok.tokenize("今天心情很好，效率也很高！"))
# 中文TinyBERT应当几乎逐字切分，基本不出现 [UNK]
