# agent/react_loop.py
from llm.model_loader import chat
from agent.patch_generator import generate_patch
from trickcatcher_lite.input_generator import generate_test_inputs
from trickcatcher_lite.executor import run_function

def run_agent(llm, function_code, fn):
    # 1) edge-case input generation
    inputs = generate_test_inputs(llm, function_code)

    # 2) 실행
    results = run_function(fn, inputs)
    failing = [r for r in results if r["error"]]

    if not failing:
        return {"patch": None, "reason": "No failures found."}

    # 3) 실패 입력을 기반으로 패치 생성
    patch = generate_patch(llm, function_code, failing)

    return {
        "patch": patch,
        "failing_examples": failing
    }
