from langchain.prompts import PromptTemplate
from utils import dedent

factify_template = PromptTemplate(
    input_variables=["context", "chunk"],
    template=dedent(
        """ 
        Your task is to take the context of document and a chunk of text, and
        extract five pertinent facts from it. The facts should only cover new
        information introduced in the chunk. The context is only for
        background; do not use it to generate facts.

        You will also generate a new context, by taking the old context and
        modifying it if needed to account for the new facts. You do not need
        to change the old context if it is suitable; simply return it again.
        Make sure the new context is as short as possible.

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

        Facts:
        F: There is a great deal of human nature in people.
        F: The animal world is a stressful place.
        F: The animal world is made up of trillions of strands of genetic information.
        F: Most gene strands don't last very long.
        F: Animals are temporary containers designed to carry the genes.
        F: The Primitive Mind is a set of coded instructions for how to be a successful animal in the animal's natural habitat.

        New Context:
        The document discusses the Primitive Mind, a set of coded instructions
        for how to be a successful animal in the animal's natural habitat. It
        is part of a larger discussion about the animal world, which is made
        up of trillions of strands of genetic information, and the great deal
        of human nature in people.

        Now the real one:

        Context:
        {context}

        Chunk:
        {chunk}

        Facts: 
        """
    )
)

question_template = PromptTemplate(
    input_variables=["facts", "context"],
    template_format="jinja2",
    template=dedent(
        """ 
        The context below is a summary of a section of a document. Below the
        context is a list of facts. For each fact, create a question that is
        answered by the fact using the context to inform your answer. Do not
        mention the context in your questions. 

        Here is an example:

        Context:

        The document discusses the Primitive Mind and Higher Mind, two parts
        of the human brain. The Primitive Mind is a set of coded instructions
        for how to be a successful animal in the animal's natural habitat,
        while the Higher Mind is the part of you that can think outside itself
        and self-reflect and get wiser with experience. People form beliefs by
        settling on a portion of the Idea Spectrum where they suspect the
        truth may lie, and scientists actively seek out dissent to test their
        hypotheses. Thinking like a Scientist involves being aware of what you
        do and don't know, and when the Primitive Mind infiltrates the
        reasoning process, people start thinking like an Attorney or a Zealot.
        The Attorney's hypothesis formation stage is a belief-strengthening
        process, and the result of thinking like an Attorney is that the
        brain's ability to learn new things is mostly shut down.

        Facts:

        F: Motivated reasoning becomes obligated reasoning when the Primitive Mind infiltrates the reasoning process.
        F: The Attorney's hypothesis formation stage is a belief-strengthening process.
        F: The Attorney's opponents will feel like they're arguing with a brick wall.
        F: The result of thinking like an Attorney is that the brain's ability to learn new things is mostly shut down.
        F: Beliefs held by the Primitive Mind can be so strong that the Higher Mind has no influence over how they are thought about.

        Questions:

        Q: What happens when the Primitive Mind infiltrates the reasoning process?
        Q: What is the result of the Attorney's hypothesis formation stage?
        Q: What is the experience of the Attorney's opponents?
        Q: What impact does thinking like an Attorney have on the brain's ability to learn?
        Q: How strong can beliefs held by the Primitive Mind be?

        Now the real one:

        Context:
        {{ context }}

        Facts:
        {% for fact in facts %}
        F: {{ fact }}
        {% endfor %}

        Questions:

        """
    )
)
