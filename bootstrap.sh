mkdir -p pretrained/qwen-3b
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./pretrained/qwen-3b/

mkdir -p pretrained/qwen-7b
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir ./pretrained/qwen-7b/

mkdir -p pretrained/blip3o-8b
huggingface-cli download BLIP3o/BLIP3o-Model-8B --local-dir ./pretrained/blip3o-8b/