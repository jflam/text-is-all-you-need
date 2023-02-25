import hashlib, json, openai, os, pickle, re, tiktoken
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter, TextSplitter, TokenTextSplitter
from langchain.llms import AzureOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM

from textual.app import App

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
LLM_MODEL = "text-davinci-003"
MAX_TOKENS = 4096

# I like the dedent concept - this is copied from summ
def dedent(text: str) -> str:
    """A more lenient version of `textwrap.dedent`."""
    return "\n".join(map(str.strip, text.splitlines())).strip()

def compute_token_count(doc: str) -> int:
    enc = tiktoken.encoding_for_model(LLM_MODEL)
    tokens = enc.encode(doc)
    return len(tokens) 

# Parse the results into context and facts
# TODO: Not going to need this once I refactor the code to ask the LLM to
# generate JSON as its response for the factify part.
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
# TODO: update this to be more relevant to non-interview tasks (might not
# matter either)
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
        Your task is to take the context of document and a chunk of text, and
        extract any pertinent facts from it. The facts should only cover new
        information introduced in the chunk. The context is only for
        background; do not use it to generate facts.

        You will also generate a new context, by taking the old context and
        modifying it if needed to account for the additional chunk. You do not
        need to change the old context if it is suitable; simply return it
        again.

        Here is an example:
        """
    ),
    suffix=dedent(
        """ 
        Now the real one:

        Context:
        {context}

        Paragraph:
        {chunk}

        Return your response as JSON:
        {{
            "facts": ["fact 1", "fact 2", ...],
            "context": "new context"
        }}
        """
    )
)

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

def process_document(document: Document, 
                     llm: BaseLLM, 
                     splitter: TextSplitter, 
                     embedding_model: Embeddings, 
                     cache: dict) -> list[Document]:
    
    # TODO: need to dynamically split a document based on token size limit
    # for LLM and the increasing size of the context. Also the context
    # seems to be repeated quite a bit as well - need some language to
    # constrain how it summarizes into the context.
    # General strategy:
    # Split into smaller chunks and then assemble those chunks until we
    # reach a token size limit that is a constraint that we precalc using 
    # the known size of the context. What we don't know is the size of the
    # response, so that needs to be constrained over time. 
    # From docs:
    # There is a CharacterTextSplitter.from_tiktoken_encoder that can be used
    # to split text into chunks of a fixed size chunk_size is now in units
    # of tokens. But the challenge here is that we need to have different 
    # sized chunks over time.
    docs = splitter.split_documents(document)
    context = DEFAULT_CONTEXT
    for doc in docs:
        chunk = doc.page_content
        chunk_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:8]
        if chunk_hash in cache:
            print(f"Cache hit for {chunk_hash}")
            doc.metadata = cache[chunk_hash]
            context = doc.metadata["context"]
        else:
            print(f"Cache miss for {chunk_hash}")
            factify_prompt = factify_template.format(context=context, 
                                                     chunk=chunk)
            print(f"Calling LLM with prompt:\n{factify_prompt}")
            result = llm(factify_prompt)
            facts, new_context = extract_facts_and_context_from_result(result)

            print(f"Extracted facts: {facts}")
            doc.metadata["facts"] = facts

            print(f"Evaluating new context: {new_context}")
            doc.metadata["context"] = new_context

            question_prompt = question_template.format(facts=facts, 
                                                       context=new_context)

            print(f"Calling LLM with prompt:\n{question_prompt}")
            result = llm(question_prompt)
            questions = json.loads(result)["questions"]

            print(f"Extracted questions: {questions}")
            doc.metadata["questions"] = questions

            question_embeddings = embedding_model.embed_documents(questions)
            fact_embeddings = embedding_model.embed_documents(facts)

            doc.metadata["question_embeddings"] = question_embeddings
            doc.metadata["fact_embeddings"] = fact_embeddings

            cache[chunk_hash] = doc.metadata
            context = new_context
    return docs

llm_model = AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
    openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
    model_name=LLM_MODEL, temperature=0.0, max_tokens=1000)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

cache = {}
if os.path.exists("cache.pkl"):
    with open("cache.pkl", "rb") as f:
        cache = pickle.load(f)

loader = TextLoader("ch1.txt")
doc = loader.load()
splitter = TokenTextSplitter.from_tiktoken_encoder(encoding_name=LLM_MODEL, chunk_size=1000)
# splitter = NLTKTextSplitter(chunk_size=8000)

docs = process_document(doc, llm_model, splitter, embedding_model, cache)

# TODO: add to FAISS

with open("cache.pkl", "wb") as f:
    pickle.dump(cache, f)

# A function that, given a document, extracts statistics from it and displays
# a simple summary using the textual library.
def summarize_document(document: Document, splitter: TextSplitter) -> None:
    docs = splitter.split_documents(document)
    for doc in docs:
        print(f"Page {doc.page_number}: {doc.page_content[:100]}...")

        if "facts" in doc.metadata:
            print(f"Facts: {doc.metadata['facts']}")

        if "questions" in doc.metadata:
            print(f"Questions: {doc.metadata['questions']}")

        print()

class DocumentProcessing(App):
    """A Textual app for processing documents for question answer."""
    pass

if __name__ == "__main__":
    app = DocumentProcessing()
    app.run()