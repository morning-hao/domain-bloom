import os
import argparse
import json
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_model(base_model):
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            load_in_8bit=True,
            device_map="auto",
        )
        if os.path.exists(args.lora_weights):
            print('==========LOAD LORA==========')
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if os.path.exists(args.lora_weights):
            model = PeftModel.from_pretrained(
                model,
                args.lora_weights,
                device_map={"": device},
            )

    return model


def load_dev_data(dev_file_path=''):
    dev_data = []
    with open(dev_file_path) as f:
        lines = f.readlines()
        for line in lines:
            dev_data.append(json.loads(line.strip()))
    return dev_data


def generate_text(dev_data, batch_size, tokenizer, model, skip_special_tokens=True, clean_up_tokenization_spaces=True):
    res = []
    for i in tqdm(range(0, len(dev_data), batch_size), total=len(dev_data), unit="batch"):
        batch = dev_data[i:i + batch_size]
        batch_text = []
        for item in batch:
            input_text = "用户:" + item['instruction'] + item['input'] + "\n\n助手:"
            batch_text.append(tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text)
            # 改输入

        with torch.autocast("cuda"):
            features = tokenizer(batch_text, padding=True, return_tensors="pt", truncation=True,
                                 max_length=args.max_length)
            input_ids = features['input_ids'].to("cuda")
            attention_mask = features['attention_mask'].to("cuda")

            output_texts = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=4,
                do_sample=False,
                min_new_tokens=1,
                max_new_tokens=512,
                early_stopping=True
            )
        output_texts = tokenizer.batch_decode(
            output_texts.cpu().numpy().tolist(),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
        for i in range(len(output_texts)):
            input_text = batch_text[i]
            input_text = input_text.replace(tokenizer.bos_token, "")
            predict_text = output_texts[i][len(input_text):]
            res.append({"input": input_text, "predict": predict_text, "target": batch[i]["output"]})
    return res


def main(args):
    dev_data = load_dev_data(args.dev_file)
    res = generate_text(dev_data, batch_size, tokenizer, model)
    with open(args.output_file, 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--dev_file", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True, help="pretrained language model")
    parser.add_argument("--max_length", type=int, default=512, help="max length of dataset")
    parser.add_argument("--dev_batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lora_weights", default="", type=str, help="use lora")
    parser.add_argument("--output_file", type=str, default="output_dir/predictions.json")

    args = parser.parse_args()
    batch_size = args.dev_batch_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = get_model(args.model_name_or_path)
    main(args)
