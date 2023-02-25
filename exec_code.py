from IPython.core.getipython import get_ipython

def evaluate_prompt(prompt: str) -> str:
    kernel = get_ipython()
    result = kernel.run_cell(prompt)
    return str(result)
