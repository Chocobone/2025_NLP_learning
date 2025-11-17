# examples/run_agent.py

from llm.model_loader import load_model
from agent.react_loop import run_agent
from example.target_code import buggy

if __name__ == "__main__":
    llm = load_model()
    with open("./examples/target_code.py") as f:
        code = f.read()

    result = run_agent(llm, code, buggy)

    print("Failing Cases:", result["failing_examples"])
    print("Patch:")
    print(result["patch"])
