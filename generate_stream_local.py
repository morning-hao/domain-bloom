import os
import argparse
import json
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--model_name_or_path", type=str, default="/home/hpn/down_model/bloomz-560m", help="pretrained language model")
    parser.add_argument("--max_length", type=int, default=512, help="max length of dataset")
    parser.add_argument("--lora_weights", default="trained_models/bloomz_560m_domain_大量测试", type=str, help="use lora")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = get_model(args.model_name_or_path)
    streamer = TextIteratorStreamer(tokenizer)

    while True:
        instruct = input('请输入您的问题:')

        input_text = "用户:" + instruct + "\n\n助手:"
        input_text_ = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
        inputs = tokenizer([input_text_], return_tensors="pt")

        generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=args.max_length)
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            generated_text += new_text
            print(new_text, end='')
        print()