# ChatGLM-webui
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarkSchmidty/ChatGLM-webui-EN/blob/main/ChatGLM-webui-EN.ipynb) <-- press here to test it in Colab

![image](https://user-images.githubusercontent.com/36563862/226165277-acc5ba44-8041-4c30-a6aa-2746c87f8475.png)

## Features

- Original Chat like [chatglm-6b](https://huggingface.co/THUDM/chatglm-6b)'s demo, but use Gradio Chatbox for better user experience.
- One click install script (but you still must install python)
- More parameters that can be freely adjusted
- Convenient save/load dialog history, presets
- Custom maximum context length
- Save to Markdown
- Use program arguments to specify model and caculation accuracy

## Install

### requirements

python3.10

```shell
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install --upgrade -r requirements.txt
```

or

```shell
bash install.sh
```

## Run

```shell
python webui.py
```

### Arguments

`--model-path`: specify model path. If this parameter is not specified manually, the default value is `THUDM/chatglm-6b`. Transformers will automatically download model from huggingface.

`--listen`: launch gradio with 0.0.0.0 as server name, allowing to respond to network requests

`--port`: webui port

`--share`: use gradio to share

`--precision`: fp32(CPU only), fp16, int4(CUDA GPU only), int8(CUDA GPU only)

`--cpu`: use cpu
