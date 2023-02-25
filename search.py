import os
import requests

BING_API_KEY = os.environ["BING_API_KEY"]
BING_ENDPOINT = os.environ["BING_API_ENDPOINT"] + "/v7.0/search"

def evaluate_prompt(prompt: str) -> str:
    mkt = 'en-US'
    params = { 'q': prompt, 'mkt': mkt }
    headers = { "Ocp-Apim-Subscription-Key": BING_API_KEY }

    response = requests.get(BING_ENDPOINT, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()

    md = ""
    for result in search_results["webPages"]["value"]:
        md += f"[{result['name']}]({result['url']})\n\n"
        md += f"{result['snippet']}\n\n"

    return md