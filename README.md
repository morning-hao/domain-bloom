# 一个简单的可商用的LLM微调项目

> 虽然复现和开源通用LLM很有意思，但没卡没数据没人力的我就做做简单的LLM微调吧

# 一、安装环境

```bash
conda create -n domain-bloom python=3.10
conda activate domain-bloom
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```



# 二、配置

 [Bloom_config.json](run_config/Bloom_config.json) 

```json
{
    。。。
    "model_name_or_path": "/xxx/bloomz-7b1-mt",  # 这个是重点，下载模型后改成自己的原始模型路径
	。。。

}
```



# 三、训练

```bash
python finetune.py --model_config_file run_config/Bloom_config.json --lora_hyperparams_file run_config/lora_hyperparams_bloom.json  --use_lora
```

# 四、预测

```bash
python generate.py --dev_file data_dir/train_data_家庭教育.jsonl --model_name_or_path /xxx/bloomz-7b1-mt  --lora_weights trained_models/xxx/
```

