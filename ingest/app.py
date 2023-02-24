import json, openai, os, re, tiktoken
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain.llms import AzureOpenAI

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

MODEL_NAME = "text-davinci-003"

# I like the dedent concept - this is copied from summ
def dedent(text: str) -> str:
    """A more lenient version of `textwrap.dedent`."""
    return "\n".join(map(str.strip, text.splitlines())).strip()

def compute_token_count(doc: str) -> int:
    enc = tiktoken.encoding_for_model(MODEL_NAME)
    tokens = enc.encode(doc)
    return len(tokens) 

# Parse the results into context and facts
def parse_list(results: list[str], prefix: str = ""):
    return [
        g.strip()
        for r in results
        for p in [re.search(prefix + r"(?:\s*)(?P<res>.*)", r)]
        for g in [p and p.group("res")]
        if p and g
    ]

def extract_facts_and_context_from_result(result: str):
    try:
        idx = result.lower().index("context")
        facts_raw, context_raw = result[:idx], result[idx:]
        context = "\n".join(context_raw.splitlines()[1:])
    except ValueError:
        facts_raw, context = result, DEFAULT_CONTEXT

    facts = parse_list(facts_raw.splitlines(), prefix=r"-+")
    return facts, context

DEFAULT_CONTEXT = "This is the start of the conversation."

# Few shot examples for factifier
FEW_SHOT_EXAMPLES = [
    {
        "context": ("The conversation so far has covered the backround of "
            "the speaker. He is in sales at UiPath."),
        "chunk": ("We had a client where they would, they had like a huge "
            "database legacy database of like their inventory in the store. "
            "Whenever they would whenever they would do any type of like "
            "inventory accounts, they would shut down for like eight hours "
            "but they wouldn't go in there and see the differences between "
            "like the database and it will take them 16 hours to do. Yes, "
            "insane. We built a bot that will go in there and do like we "
            "like to call it, auditing and reconciliation of all the "
            "inventories, as long as they gave us like a spreadsheet, and "
            "you could do it in an hour."),
        "facts": [
            ("A client had a large legacy database for inventory in their "
             "store."),
            ("The inventory reconciliation process would shut down the store "
             "for 8 hours."),
            ("The process of reconciling the database would take 16 hours "
             "to complete."),
            ("A bot was built to perform inventory auditing and "
             "reconciliation."),
            ("The bot can complete the process in an hour as long as a "
             "spreadsheet is provided."),
        ],
        "new_context": ("An RPA developer talks about a bot he made. The bot "
        "was created to reconcile a client's inventory database which used "
        "to take 16 hours to complete and shut down the store for 8 hours, "
        "and can now be done in an hour."),
    }
]

few_shot_examples = PromptTemplate(
    input_variables=["context", "chunk", "facts", "new_context"],
    template_format="jinja2",
    template=dedent(
        """
        {% for item in items %}
        ---
        Context:
        {{ item.context }}

        Paragraph:
        {{ item.chunk }}

        Facts:
        - {{ item.facts | join("\n- ") }}

        Context:
        {{ item.new_context }}
        ---
        {% endfor %}
        """
    )
)

factify_template = FewShotPromptTemplate(
    examples=FEW_SHOT_EXAMPLES,
    example_prompt=few_shot_examples,
    input_variables=["context", "chunk"],
    prefix=dedent(
        """ 
        Your task is to take the context of a conversation, and a
        paragraph, and extract any pertinent facts from it. The facts
        should only cover new information introduced in the paragraph. The
        context is only for background; do not use it to generate facts.

        You will also generate a new context, by taking the old context
        and modifying it if needed to account for the additional
        paragraph. You do not need to change the old context if it is
        suitable; simply return it again.

        Here is an example:
        """
    ),
    suffix=dedent(
        """ 
        Now the real one:

        ---
        Context:
        {context}

        Paragraph:
        {chunk}

        Facts:
        -
        """
    )
)

# Read document into a string
loader = TextLoader("ch1.txt")
doc = loader.load()

splitter = NLTKTextSplitter(chunk_size=8000)
docs = splitter.split_documents(doc)

context = DEFAULT_CONTEXT
chunk = docs[0].page_content
prompt = factify_template.format(context=context, chunk=chunk)

llm = AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
    openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
    model_name=MODEL_NAME, temperature=0.0, max_tokens=1000)

print(f"Calling factify...")

print(f"Calling factify with prompt: {prompt}")
print(f"Token count: {compute_token_count(prompt)}")
print(f"There are {len(docs)} chunks in the document.")

result = llm(prompt)

# TODO: implement a similar mechanism for caching results from
# LLM calls given the expense of doing these calls. Make sure
# to persist the cache to disk.
facts, context = extract_facts_and_context_from_result(result)

str_facts = "- " + str.join("\n- ", facts)
print(f"Facts:\n{str_facts}")
print(f"Context:\n{context}")

question_template = PromptTemplate(
    input_variables=["facts", "context"],
    template_format="jinja2",
    template=dedent(
        """ 
        The context below is a summary of a section of a document. Below the
        summary is a list of facts. For each fact, create a question that is
        answered by the fact using the context to inform your answer. Do not
        mention the context in your questions.

        {% for fact in facts %}
        Fact {{ loop.index }}: {{ fact }}
        {% endfor %}

        Context: {{ context }}

        Return your answer encoded in JSON:
        {
            "questions": [
                "Question 1",
                "Question 2",
                ...
            ]
        }
        """
    )
)

question_prompt = question_template.format(facts=facts, context=context)
print(f"Calling question generator with prompt:\n{question_prompt}")

result = llm(question_prompt)
questions = json.loads(result)
print(f"Questions:\n{questions}")