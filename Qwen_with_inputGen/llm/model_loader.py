# llm/model_loader.py
from vllm import LLM, SamplingParams

def load_model():
    llm = LLM(
        model="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
        dtype="auto",
        device="cuda"
    )
    return llm

def chat(llm, prompt, temperature=0.2):
    params = SamplingParams(
        temperature=temperature,
        max_tokens=1024,
        top_p=0.9
    )
    output = llm.generate([prompt], params)[0].outputs[0].text
    return output
