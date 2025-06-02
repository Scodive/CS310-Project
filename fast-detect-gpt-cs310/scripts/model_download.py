from transformers import AutoModelForCausalLM, AutoTokenizer

models = [
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-7b"
]

cache_dir = "../cache"  # 你的缓存目录路径

for model_name in models:
    # 保存模型
    model = AutoModelForCausalLM.from_pretrained(model_name)
    save_path = f"{cache_dir}/local.{model_name.replace('/', '_')}"
    model.save_pretrained(save_path)

    # 保存 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)

    print(f"成功保存模型和 Tokenizer 到：{save_path}")