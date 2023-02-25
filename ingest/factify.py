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
        
        This is the start of the document.

        Chunk:

        CHAPTER 1: THE LADDER There is a great deal of human nature in people.

        - Mark Twain  THE TUG-OF-WAR IN OUR HEADS The animal world is a
        stressful place to be.

        The issue is that the animal world isn't really an animal world—it's a
        world of trillions of strands of genetic information, each one
        hell-bent on immortality.

        Most gene strands don't last very long, and those still on Earth today
        are the miracle outliers, such incredible survival specialists that
        they're hundreds of millions of years old and counting.

        Animals are just a hack these outlier genes came up with—temporary
        containers designed to carry the genes and help them stay immortal.

        Genes can't talk to their animals, so they control them by having them
        run on specialized survival software I call the Primitive Mind:   The
        Primitive Mind is a set of coded instructions for how to be a
        successful animal in the animal's natural habitat.

        {{
            "facts": [
                "There is a great deal of human nature in people.",
                "The animal world is a stressful place.",
                "The animal world is made up of trillions of strands of genetic information.",
                "Most gene strands don't last very long.",
                "Animals are temporary containers designed to carry the genes.",
                "The Primitive Mind is a set of coded instructions for how to be a successful animal in the animal's natural habitat."
            ],
            "new_context": "The document discusses the Primitive Mind, a set of coded instructions for how to be a successful animal in the animal's natural habitat. It is part of a larger discussion about the animal world, which is made up of trillions of strands of genetic information, and the great deal of human nature in people."
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

question_template = PromptTemplate(
    input_variables=["facts", "context"],
    template_format="jinja2",
    template=dedent(
        """ 
        The context below is a summary of a section of a document. Below the
        summary is a list of facts. For each fact, create a question that is
        answered by the fact using the context to inform your answer. Do not
        mention the context in your questions.

        Context: 
        {{ context }}

        Facts:
        {% for fact in facts %}
        Fact {{ loop.index }}: {{ fact }}
        {% endfor %}

        Return your answer encoded in JSON:
        {
            "questions": [
                "Question for Fact 1",
                "Question for Fact 2",
                ...
            ]
        }
        """
    )
)
