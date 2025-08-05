## LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation

This is the code repository for our paper:  ```LoopLLM: Transferable Energy-Latency Attacks in LLMs via Repetitive Generation```

## Abstract

As large language models (LLMs) scale, their inference incurs substantial computational resources, exposing them to energy-latency attacks, where crafted prompts induce high energy and latency cost. Existing attack methods aim to prolong output by delaying the generation of termination symbols. However, as the output grows longer, controlling the termination symbols through input becomes difficult, making these methods less effective. Therefore, we propose \textbf{LoopLLM}, an energy-latency attack framework based on the observation that repetitive generation can trigger low-entropy decoding loops, reliably compelling LLMs to generate until their output limits. LoopLLM introduces (1) a repetition-inducing prompt optimization that exploits autoregressive vulnerabilities to induce repetitive generation, and (2) a token-aligned ensemble optimization that aggregates gradients to improve cross-model transferability. Extensive experiments on 12 open-source and 2 commercial LLMs show that LoopLLM significantly outperforms existing methods, achieving over 90\% of the maximum output length, compared to 20\% for baselines, and improving transferability by around 40\% to DeepSeek-V3 and Gemini 2.5 Flash.

![overview](https://github.com/neuron-insight-lab/LoopLLM/raw/main/assets/overview.png)

## Installation 

### Environment Preparation

```bash
pip install - r requirements.txt
```

### Model Preparation

You can download the required LLMs from the [huggingface](https://huggingface.co/)  (such as [LLama2-7B](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)) and fill in your model path in the corresponding position of the ```utils/__init__.py``` file

## Usage

Run the following command to generate LoopLLM prompt to induce  Llama2-7b into repetitive generation. 

```bash
python main.py --model_name "llama2-7b"
```

Or you can run the `ensemble.py` file to ensemble optimization on multiple models to construct more generalized prompt and use the `transfer.py` file to conduct transfer experiments.

## The Examples of Real-World LLMs
Screenshots of repetitive generation triggered by our LoopLLM on real-world LLMs. \
Top left: Deepseek-V3. Top
right: Gemini 2.5 Flash. Bottom left: Mistrial. Bottom right: Meta LLaMA2-7B
![overview](https://github.com/neuron-insight-lab/LoopLLM/raw/main/assets/real_world.png)