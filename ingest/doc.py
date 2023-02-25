from langchain.document_loaders import TextLoader
from langchain.text_splitter import NLTKTextSplitter, TextSplitter
from langchain.llms import AzureOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import TextLog, Header, Footer, DirectoryTree, Static
from tiktoken import encoding_for_model, Encoding
from factify import factify_template
import asyncio, json, openai, os

ENVIRONMENT="EAST_AZURE_OPENAI"
openai.api_base = os.environ[f"{ENVIRONMENT}_ENDPOINT"]
openai.api_type = "azure"
openai.api_version = "2022-12-01"
DEPLOYMENT_ID = os.environ[f"{ENVIRONMENT}_DEPLOYMENT"]

EMBEDDING_MODEL = "sentence-transformers/gtr-t5-large"
LLM_MODEL = "text-davinci-003"
MAX_TOKENS = 4096

DEFAULT_CONTEXT = "This is the start of the document."

llm_model = AzureOpenAI(deployment_name=DEPLOYMENT_ID, 
    openai_api_key=os.environ[f"{ENVIRONMENT}_API_KEY"],
    model_name=LLM_MODEL, temperature=0.0, max_tokens=1000)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
splitter = NLTKTextSplitter(chunk_size=4000)
tokenizer = encoding_for_model(LLM_MODEL)

def load_file(filename: str) -> str:
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

async def process_document(filename: str, text_log: TextLog) -> None:
    text_log.clear()
    text_log.write(f"Processing: [bold yellow]{filename}[/]")
    content, tokens = load_file(filename)
    chunks, tokens = split_document(content, splitter, tokenizer)
    text_log.write(f"Document has [bold yellow]{tokens}[/] tokens.")
    text_log.write(f"I have split the document into [bold yellow]{len(chunks)}[/] chunks.")
    text_log.write("Last 40 characters of each chunk prefixed by character count and token count:")
    for i, chunk in enumerate(chunks):
        trimmed = " ".join(chunk[-40:].splitlines())
        text_log.write(f"{i+1} [{len(chunk)}][{tokens[i]}]: "
                            f"[green]... {trimmed}[/]")

    context = DEFAULT_CONTEXT
    for i in range(2):
        chunk = chunks[i]
        text_log.write(f"Sending chunk {i+1} to LLM...")

        factify_prompt = factify_template.format(context=context, chunk=chunk)
        response = await llm_model.agenerate([factify_prompt])
        text = response.generations[0][0].text.strip()
        text_log.write(f"Raw response: {text}")
        obj = json.loads(text)
        facts = obj["facts"]
        new_context = obj["new_context"]
        text_log.write("Facts extracted by the model:")
        for i, fact in enumerate(facts):
            text_log.write(f"Fact {i+1}: [green]{fact}[/]")
        text_log.write(f"New context: [green]{new_context}[/]")
        context = new_context

class DocumentProcessingApp(App):
    BINDINGS = [
        ("d", "toggle_dark", "Toggle dark mode"),
        ("q", "quit", "Quit")
    ]
    TITLE = "Document Processing App"
    SUB_TITLE = "Import a document into question answer database"
    CSS_PATH = "doc.css"

    def compose(self) -> ComposeResult:
        yield Header()
        yield DirectoryTree(path=".", classes="box")
        yield TextLog(highlight=True, markup=True, wrap=True, classes="box", id="log")
        yield Static("Bob", classes="box")
        yield Footer()
    
    def action_toggle_dark(self) -> None:
        self.dark = not self.dark

    def action_quit(self) -> None:
        self.exit()

    async def on_directory_tree_file_selected(self, msg: Message) -> None:
        text_log = self.query_one("#log", TextLog)
        asyncio.create_task(process_document(msg.path, text_log))

if __name__ == "__main__":
    app = DocumentProcessingApp()
    app.run()