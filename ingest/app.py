# Experiments in doing a summ-like ingestion pipeline
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

# I like the dedent concept - this is copied from summ
def dedent(text: str) -> str:
    """A more lenient version of `textwrap.dedent`."""
    return "\n".join(map(str.strip, text.splitlines())).strip()

# Few shot examples for factifier
FEW_SHOT_EXAMPLES = [
    {
        "context": "The conversation so far has covered the backround of the speaker. He is in sales at UiPath.",
        "chunk": "We had a client where they would, they had like a huge database legacy database of like their inventory in the store. Whenever they would whenever they would do any type of like inventory accounts, they would shut down for like eight hours but they wouldn't go in there and see the differences between like the database and it will take them 16 hours to do. Yes, insane. We built a bot that will go in there and do like we like to call it, auditing and reconciliation of all the inventories, as long as they gave us like a spreadsheet, and you could do it in an hour.",
        "facts": [
            "A client had a large legacy database for inventory in their store.",
            "The inventory reconciliation process would shut down the store for 8 hours.",
            "The process of reconciling the database would take 16 hours to complete.",
            "A bot was built to perform inventory auditing and reconciliation.",
            "The bot can complete the process in an hour as long as a spreadsheet is provided.",
        ],
        "new_context": " An RPA developer talks about a bot he made. The bot was created to reconcile a client's inventory database which used to take 16 hours to complete and shut down the store for 8 hours, and can now be done in an hour.",
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

print(few_shot_examples.format(items=FEW_SHOT_EXAMPLES))