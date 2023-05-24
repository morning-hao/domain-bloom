from flask import Flask, request, Response
import os
import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

app = Flask(__name__)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_model(base_model):
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # torch_dtype=torch.float16,
            # load_in_8bit=True,
            device_map="auto",
        )
        if os.path.exists(lora_model):
            model = PeftModel.from_pretrained(
                model,
                lora_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        if os.path.exists(lora_model):
            model = PeftModel.from_pretrained(
                model,
                lora_model,
                device_map={"": device},
            )

    return model


def generate_response(stream):
    for new_text in stream:
        if '\n' in new_text:
            new_text = new_text.replace('\n', '<FN>')
        yield new_text + '\n'


@app.route('/', methods=['POST'])
def answer_question():
    streamer = TextIteratorStreamer(tokenizer)
    data = request.get_json()
    instruct = data['question']

    input_text = "用户:" + instruct + "\n\n助手:"
    input_text_ = tokenizer.bos_token + input_text if tokenizer.bos_token != None else input_text
    inputs = tokenizer([input_text_], return_tensors="pt")

    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=max_length)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    return Response(generate_response(streamer), mimetype='text/plain')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--model_name_or_path", type=str, default="/xx/bloomz-xx", help="pretrained language model")
    parser.add_argument("--lora_weights", default="trained_models/xx", type=str,
                        help="use lora")
    args = parser.parse_args()

    base_model = args.model_name_or_path
    lora_model = args.lora_weights

    max_length = 512
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    model = get_model(base_model)

    app.run(host='0.0.0.0', port=5000)
