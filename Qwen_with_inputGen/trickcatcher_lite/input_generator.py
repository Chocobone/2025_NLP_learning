# trickcatcher_lite/input_generator.py
from llm.model_loader import chat

def generate_test_inputs(llm, function_code):
    prompt = f"""
    You are an expert at generating adversarial test inputs.
    Given the following function, generate 20 test inputs as JSON list.
    Include:
    - null / None
    - empty values
    - long strings
    - negative numbers
    - zero
    - very large numbers
    - wrong types
    - boundary values
    
    Function:
    {function_code}
    Return ONLY JSON list.
    """
    
    raw = chat(llm, prompt)
    import json, re

    json_str = re.search(r'\[.*\]', raw, re.S)
    return json.loads(json_str.group(0)) if json_str else []
