# trickcatcher_lite/executor.py
import traceback

def run_function(fn, inputs):
    results = []
    for inp in inputs:
        try:
            out = fn(inp)
            results.append({"input": inp, "output": out, "error": None})
        except Exception as e:
            results.append({
                "input": inp,
                "output": None,
                "error": traceback.format_exc()
            })
    return results
