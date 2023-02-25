from ast import Tuple
from typing import Any
from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter, TextSplitter
from langchain.llms import AzureOpenAI
from langchain.llms.base import BaseLLM
from langchain.embeddings import HuggingFaceEmbeddings
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import TextLog, Header, Footer, DirectoryTree, Static
from tiktoken import encoding_for_model, Encoding
from factify import factify_template, question_template
import asyncio, hashlib, json, openai, os, pickle

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
LLM_MODEL = "text-davinci-003"
MAX_TOKENS = 4096
TEMPERATURE = 0
RESPONSE_TOKEN_LIMIT = 1000
CACHE_FILENAME = "cache.pkl"
ERROR_FILENAME = "errors.txt"
LOG_FILENAME = "log.txt"

DEFAULT_CONTEXT = "This is the start of the document."

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
splitter = NLTKTextSplitter(chunk_size=4000)
tokenizer = encoding_for_model(LLM_MODEL)

def load_file(filename: str) -> Tuple(str, int):
    """Loads filename and returns the content and token count."""
    loader = TextLoader(filename)
    doc = loader.load()
    assert len(doc) == 1
    content = " ".join(doc[0].page_content.splitlines())
    encoder = encoding_for_model(LLM_MODEL)
    tokens = encoder.encode(content)
    return content, len(tokens)

def split_document(content: str,
                   splitter: TextSplitter,
                   tokenizer: Encoding):
    """Split content into chunks and return the chunks and token counts. """
    chunks = splitter.split_text(content)
    tokens = [len(tokenizer.encode(chunk)) for chunk in chunks]
    return chunks, tokens

def get_llm() -> AzureOpenAI:
    return AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
        openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
        model_name=LLM_MODEL, temperature=TEMPERATURE,
        max_tokens=RESPONSE_TOKEN_LIMIT)

def dump_error(e: Exception, message: str) -> None:
    with open(ERROR_FILENAME, "a") as file:
        file.write(f"Error: {e}\n\n")
        file.write(f"{message}\n\n")

def dump_output(message: str) -> None:
    with open(LOG_FILENAME, "a") as file:
        file.write(f"{message}\n\n")

async def call_llm(llm: BaseLLM, prompt: str, text_log: TextLog) -> Any:
    """Async call the LLM and return the response as stripped string."""
    try:
        result = await llm.agenerate([prompt])
        return result.generations[0][0].text.strip()
    except openai.error.InvalidRequestError as e:
        text_log.write(f"[red]InvalidRequestError: {e}[/]")
        dump_error(e, prompt)
        return None

def json_to_obj(response: str, text_log: TextLog) -> Any:
    """Parse the response as JSON and return object."""
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        text_log.write(f"[red]JSONDecodeError on: {e}[/]")
        dump_error(e, response)
        return None

def log_chunks(chunks: list[str], tokens: list[int], text_log: TextLog) -> None:
    """Log the last 40 characters of each chunk."""
    text_log.write(f"I have split the document into "
                   f"[bold yellow]{len(chunks)}[/] chunks.")
    text_log.write("Last 40 characters of each chunk prefixed by character "
                   "count and token count:")
    for i, chunk in enumerate(chunks):
        trimmed = " ".join(chunk[-40:].splitlines())
        text_log.write(f"{i+1} [{len(chunk)}][{tokens[i]}]: "
                            f"[green]... {trimmed}[/]")

async def process_document(filename: str, text_log: TextLog, 
                           cache: dict) -> None:
    text_log.clear()
    text_log.write(f"Processing: [bold yellow]{filename}[/]")
    content, token_count = load_file(filename)
    text_log.write(f"Document has [bold yellow]{token_count}[/] tokens.")
    chunks, tokens = split_document(content, splitter, tokenizer)
    log_chunks(chunks, tokens, text_log)

    llm_model = get_llm()
    context = DEFAULT_CONTEXT
    for i in range(4):
        chunk = chunks[i]
        chunk_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:8]
        if chunk_hash in cache:
            text_log.write(f"Cache hit for chunk {i+1}!")
            metadata = cache[chunk_hash]
            context = metadata["context"]
            facts = metadata["facts"]
        else:
            metadata = {}
            text_log.write(f"Sending chunk {i+1} to LLM...")
            factify_prompt = factify_template.format(context=context, chunk=chunk)
            text = await call_llm(llm_model, factify_prompt, text_log)
            if text is None:
                continue
            obj = json_to_obj(text, text_log)
            if obj is None:
                continue

            facts, context = obj["facts"], obj["new_context"]
            metadata["facts"], metadata["context"] = facts, context
            cache[chunk_hash] = metadata

        if "questions" in metadata:
            questions = metadata["questions"]
        else:
            questions_prompt = question_template.format(context=context, 
                                                        facts=facts)
            text = await call_llm(llm_model, questions_prompt, text_log)
            if text is None:
                continue

            obj = json_to_obj(text, text_log)

            questions = obj["questions"]
            metadata["questions"] = questions

        text_log.write("Questions and Facts extracted by the model:")
        for i, fact in enumerate(facts):
            text_log.write(f"Question {i+1}: [green]{questions[i]}[/]")
            text_log.write(f"Fact {i+1}: [green]{fact}[/]")
        text_log.write(f"New context: [green]{context}[/]")

class DocumentProcessingApp(App):
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit")
    ]
    TITLE = "Document Processing App"
    SUB_TITLE = "Import a document into question answer database"
    CSS_PATH = "doc.css"

    def __init__(self) -> None:
        super().__init__()
        self.cache = {}
        if os.path.exists(CACHE_FILENAME):
            with open(CACHE_FILENAME, "rb") as file:
                self.cache = pickle.load(file)
        if os.path.exists(ERROR_FILENAME):
            os.remove(ERROR_FILENAME)
        if os.path.exists(LOG_FILENAME):
            os.remove(LOG_FILENAME)

    def compose(self) -> ComposeResult:
        yield Header()
        yield DirectoryTree(path=".", classes="box")
        yield TextLog(highlight=True, markup=True, wrap=True, classes="box", 
                      id="log")
        yield Static("Bob", classes="box")
        yield Footer()
    
    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_quit(self) -> None:
        with open(CACHE_FILENAME, "wb") as file:
            pickle.dump(self.cache, file)
        self.exit()

    async def on_directory_tree_file_selected(self, msg: Message) -> None:
        text_log = self.query_one("#log", TextLog)
        asyncio.create_task(process_document(msg.path, text_log, self.cache))

if __name__ == "__main__":
    app = DocumentProcessingApp()
    app.run()