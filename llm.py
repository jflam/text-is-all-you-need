import openai
import os

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_key = os.environ[f"{ENVIRONMENT}_API_KEY"]
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = 'azure'
openai.api_version = '2022-12-01' # this may change in the future
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

def evaluate_prompt(prompt: str) -> str:
    response = openai.Completion.create(
        engine=DEPLOYMENT_ID,
        prompt=prompt,
        temperature=0.0,
        max_tokens=1000,
    )
    return response.choices[0].text