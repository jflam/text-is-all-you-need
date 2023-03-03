from ast import Tuple
from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter, TextSplitter
from langchain.llms.base import LLM
from langchain.llms import AzureOpenAI, OpenAIChat
from langchain.embeddings import HuggingFaceEmbeddings
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import TextLog, Header, Footer, DirectoryTree
from tiktoken import encoding_for_model
from factify import one_pass_factify_template
from nltk.tokenize.punkt import PunktSentenceTokenizer
import asyncio, hashlib, openai, os, pickle

# ENVIRONMENT="EAST_AZURE_OPENAI"
ENVIRONMENT="OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
# openai.api_type = "azure"
# openai.api_version = "2022-12-01"
# DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
# LLM_MODEL = "text-davinci-003"

# The turbo model released 3/1/2023
LLM_MODEL = "gpt-3.5-turbo"
COST_PER_1K_TOKENS = 0.002 

MAX_TOKENS = 4096
TEMPERATURE = 0
CACHE_FILENAME = "cache.pkl"
ERROR_FILENAME = "errors.txt"
LOG_FILENAME = "log.txt"

# These parameters are used to manage the LLM token window
MAX_CONTEXT = 1000
RESPONSE_TOKEN_LIMIT = 2000
CHUNK_SIZE = 2500

DEFAULT_CONTEXT = "This is the start of the document."

class Cache:
    """Stores CacheEntries and manages sequence"""
    pass

class CacheEntry:
    """Cache entry for a chunk of text in a document."""
    def __init__(self, 
                 questions: list[str],
                 facts: list[str],
                 chunk: str,
                 cost: int):
        self.questions = questions # List of questions to be embedded
        self.facts = facts # List of facts to be embedded
        self.chunk = chunk # Chunk of text from document
        self.cost = cost # Number of tokens it cost to generate this entry

# TODO: delay load this when needed!
# embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
splitter = NLTKTextSplitter(chunk_size=CHUNK_SIZE)

# TODO: add UI to toggle this
COST_MONITORING = False

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
                   splitter: TextSplitter) -> Tuple(list[str], list[int]):
    """Split content into chunks and return the chunks and token counts. """
    chunks = splitter.split_text(content)
    if COST_MONITORING:
        tokenizer = encoding_for_model(LLM_MODEL)
        tokens = [len(tokenizer.encode(chunk)) for chunk in chunks]
    else:
        tokens = [0] * len(chunks)
    return chunks, tokens

def dump_error(e: Exception, message: str) -> None:
    with open(ERROR_FILENAME, "a") as file:
        file.write(f"Error: {e}\n\n")
        file.write(f"{message}\n\n")

def dump_output(message: str) -> None:
    with open(LOG_FILENAME, "a") as file:
        file.write(f"{message}\n\n")

def get_llm(max_tokens: int = RESPONSE_TOKEN_LIMIT) -> LLM:
    return OpenAIChat(openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"], 
                      temperature=TEMPERATURE,
                      max_tokens=-1,
                      model_name=LLM_MODEL)
    # return AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
    #     openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
    #     model_name=LLM_MODEL, temperature=TEMPERATURE,
    #     max_tokens=-1) # TODO: check this value

def extract_facts_questions_and_context(response: str) -> list[str]:
    facts = []
    questions = []
    context = []
    context_mode = False
    for line in response.splitlines():
        if line.startswith("New Context:"):
            context.append(line[12:].strip())
            context_mode = True
        else:
            if context_mode:
                context.append(line)
            elif line.startswith("F:"):
                facts.append(line[2:].strip())
            elif line.startswith("Q:"):
                questions.append(line[2:].strip())

    return facts, questions, " ".join(context)

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

def strip_sentences(text: str, limit: int = MAX_CONTEXT) -> str:
    """Strip sentences from text until it is less than limit."""
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    while len(text) > limit:
        text = " ".join(sentences[:-1])
        sentences = tokenizer.tokenize(text)
    return text

async def process_document_one_shot(filename: str, 
                                    text_log: TextLog, 
                                    stats_log: TextLog,
                                    cache: dict) -> None:
    text_log.clear()
    content, token_count = load_file(filename)
    stats_log.write(f"File [bold yellow]{filename}[/] has "
                    f"[bold yellow]{token_count} tokens[/].")
    chunks, tokens = split_document(content, splitter)
    log_chunks(chunks, tokens, stats_log)

    factify_model = get_llm()
    context = DEFAULT_CONTEXT
    cumulative_tokens = 0
    chunk_count = 3
    for i in range(chunk_count):
        try:
            chunk_id = f"{i+1}/{chunk_count}"
            dump_output(f"Processing chunk {chunk_id}...")
            dump_output(f"Context: {context}")
            chunk = chunks[i]
            chunk_hash = hashlib.sha1(chunk.encode("utf-8")).hexdigest()[:8]
            if chunk_hash in cache:
                text_log.write(f"Cache hit for chunk {chunk_id}!")
                metadata = cache[chunk_hash]
                context = metadata["context"]
                facts = metadata["facts"]
                questions = metadata["questions"]
            else:
                metadata = {}
                text_log.write(f"Sending chunk {chunk_id} to LLM...")
                factify_prompt = one_pass_factify_template.format(
                    context=context, 
                    chunk=chunk)
                dump_output(">>> FACTIFY PROMPT")
                dump_output(factify_prompt)
                dump_output("==============================")

                try:
                    result = await factify_model.agenerate([factify_prompt])
                except openai.error.InvalidRequestError as e:
                    text_log.write(f"[red]InvalidRequestError: {e}[/]")
                    dump_error(e, factify_prompt)
                    continue

                text = result.generations[0][0].text.strip()
                token_count = result.llm_output['token_usage']['total_tokens']
                stats_log.write(f"Chunk {chunk_id} used "
                                f"[bold yellow]{token_count} tokens")
                cumulative_tokens += token_count

                dump_output(">>> FACTIFY MODEL OUTPUT")
                dump_output(text)
                dump_output("==============================")

                facts, questions, context = extract_facts_questions_and_context(text)

            context = strip_sentences(context)
            metadata["facts"], metadata["context"] = facts, context
            metadata["questions"] = questions
            cache[chunk_hash] = metadata

            assert len(facts) == len(questions)
            text_log.write("Questions and Facts extracted by the model:")
            for i in range(len(questions)):
                text_log.write(f"Question {i+1}: [green]{questions[i]}[/]")
                text_log.write(f"Fact {i+1}: [green]{facts[i]}[/]")
            text_log.write(f"New context: [green]{context}[/]")
        except BaseException as e:
            text_log.write(f"[red]Exception Name: {type(e)}[/]")
            text_log.write(f"[red]Exception Message: {e}[/]")
    text_log.write("[bold yellow]Done![/]")
    stats_log.write(f"Total tokens used: [bold yellow]{cumulative_tokens}[/]")
    dollar_cost = cumulative_tokens / 1000.0 * COST_PER_1K_TOKENS
    stats_log.write(f"Total Cost = $[bold yellow]{dollar_cost}[/]")

def dump_questions_and_answers(path: str, text_log: TextLog) -> None:
    with open(path, "rb") as file:
        cache = pickle.load(file)

    for key in cache:
        metadata = cache[key]
        facts = metadata["facts"]
        questions = metadata["questions"]
        context = metadata["context"]

        text_log.write(f"Context\n:{context}")
        if len(facts) == len(questions):
            for i in range(len(questions)):
                text_log.write(f"Question {i+1}: [green]{questions[i]}[/]")
                text_log.write(f"Fact {i+1}: [green]{facts[i]}[/]")
        else:
            text_log.write(f"[red]Mismatch in facts and questions for chunk[/]")
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
        yield TextLog(markup=True, wrap=True, id="stats", classes="box")
        yield Footer()
    
    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_quit(self) -> None:
        with open(CACHE_FILENAME, "wb") as file:
            pickle.dump(self.cache, file)
        self.exit()

    async def on_directory_tree_file_selected(self, msg: Message) -> None:
        text_log = self.query_one("#log", TextLog)
        stats_log = self.query_one("#stats", TextLog)
        path = msg.path
        if path.endswith(".pkl"):
            dump_questions_and_answers(path, text_log)
        else:
            asyncio.create_task(process_document_one_shot(path, 
                                                          text_log, 
                                                          stats_log, 
                                                          self.cache))

if __name__ == "__main__":
    app = DocumentProcessingApp()
    app.run()