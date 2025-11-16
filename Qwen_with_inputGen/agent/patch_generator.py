# agent/patch_generator.py

from llm.model_loader import chat

def generate_patch(llm, function_code, failing_examples):
    example_text = "\n".join([
        f"Input: {f['input']}\nError: {f['error']}"
        for f in failing_examples[:5]
    ])

    prompt = f"""
You are an expert Python bug fixer.

Here is a buggy function:
{function_code}

Here are failing inputs and errors:
{example_text}

Fix the bug. Return ONLY patched code. Keep same signature.
"""

    response = chat(llm, prompt)
    return response
