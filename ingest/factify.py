from langchain.prompts import PromptTemplate
from utils import dedent

factify_template = PromptTemplate(
    input_variables=["context", "chunk"],
    template=dedent(
        """ 
        Your task is to take the context of document and a chunk of text, and
        extract up to ten pertinent facts from it. The facts should only cover
        new information introduced in the chunk. The context is only for
        background; do not use it to generate facts.

        You will also generate a new context, by taking the old context and
        modifying it if needed to account for the additional chunk. You do not
        need to change the old context if it is suitable; simply return it
        again.

        Here is an example:

        Context: 
        
        The conversation so far has covered the backround of the speaker. He
        is in sales at UiPath.

        Chunk:

        We had a client where they would, they had like a huge database legacy
        database of like their inventory in the store. Whenever they would
        whenever they would do any type of like inventory accounts, they would
        shut down for like eight hours but they wouldn't go in there and see
        the differences between like the database and it will take them 16
        hours to do. Yes, insane. We built a bot that will go in there and do
        like we like to call it, auditing and reconciliation of all the
        inventories, as long as they gave us like a spreadsheet, and you could
        do it in an hour.

        {{
            "facts": [
                "A client had a large legacy database for inventory in their store.",
                "The inventory reconciliation process would shut down the store for 8 hours.",
                "The process of reconciling the database would take 16 hours to complete.",
                "A bot was built to perform inventory auditing and reconciliation.",
                "The bot can complete the process in an hour as long as a spreadsheet is provided.",
            ],
            "new_context": "An RPA developer talks about a bot he made. The bot was created to reconcile a client's inventory database which used to take 16 hours to complete and shut down the store for 8 hours, and can now be done in an hour."
        }}

        Now the real one:

        Context:
        {context}

        Chunk:
        {chunk}

        Only return JSON in your response based on the example above.
        """
    )
)