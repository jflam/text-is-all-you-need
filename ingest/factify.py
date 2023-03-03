from langchain.prompts import PromptTemplate
from utils import dedent

# TODO: trim the examples to be more concise
one_pass_factify_template = PromptTemplate(
    input_variables=["context", "chunk"],
    template=dedent(
        """
        Below is a chunk of text from a document and a context that summarizes
        the document up to that point.

        Your task is to generate five pertinent facts from the chunk of text.
        The facts should only cover new information introduced in the chunk.
        The context is only for background; do not use it to generate facts.

        For each fact, I want you to generate a question that is answered by
        the fact using the context to inform your answer. Do not mention the
        context in your questions.
        
        You will also generate a new context, by taking the old context and
        modifying it if needed to account for the new facts. You do not need
        to change the old context if it is suitable; simply return it again.
        Make sure the new context is as short as possible.

        Here is an example:

        Context: 
        
        Idea Labs are places where humility is highly valued, and ideas are
        treated like hypotheses. People with a reputation for bias or
        arrogance or dishonesty will be met with a high degree of skepticism,
        while arguments are not taken personally. It might even be optimal to
        be a little over-confident in our intellectual lives, as this allows
        us to really give our ideas a try. Echo Chambers, on the other hand,
        are cultures of groupthink and conformity, where falling in line with
        the rest of the group is socially rewarded and humility is looked down
        upon. Conviction is social currency in an Echo Chamber, and ideas are
        equated with a person's identity.

        Chunk:

        But Echo Chambers equate a person’s ideas with their identity, so
        respecting a person and respecting their ideas are one and the same.

        Disagreeing with someone in an Echo Chamber is seen not as
        intellectual exploration but as rudeness, making an argument about
        ideas indistinguishable from a fight.

        This moral component provides Echo Chambers with a powerful tool for
        cultural law enforcement: taboo.

        Those who challenge the sacred ideas are seen not just as wrong but as
        bad people.

        As such, violators are slapped with the social fines of status
        reduction or reputation damage, the social jail time of ostracism, and
        even the social death penalty of permanent excommunication.

        Express the wrong opinion on God, abortion, patriotism, immigration,
        race, or capitalism in the wrong group and you may be met with an
        explosive negative reaction.

        Echo Chambers are places where you must watch what you say.

        An Echo Chamber can be the product of a bunch of people who all hold
        certain ideas to be sacred.

        Other times, it can be the product of one or a few “intellectual
        bullies” who everyone else is scared to defy.

        Even in the smallest group—a married couple, say—if one person knows
        that it’s never worth the fight to challenge their spouse’s strongly
        held viewpoints, the spouse is effectively imposing Echo Chamber
        culture on the marriage.

        Intellectual cultures have a major impact on the individuals within
        them.

        While Idea Lab culture encourages intellectual and moral growth, Echo
        Chamber culture discourages new ideas, curbs intellectual innovation,
        and removes knowledge-acquisition tools like debate—all of which
        repress growth.

        Spending too much time in an Echo Chamber makes people feel less
        humble and more sure of themselves, all while limiting actual learning
        and causing thinking skills to atrophy.

        In a broader sense, both primitive-mindedness and high-mindedness tend
        to be contagious.

        While Idea Lab culture is a support group that helps keep people’s
        minds up on the high rungs, Echo Chamber culture pumps out Primitive
        Mind pheromones and exerts a general downward pull on the psyches of
        its members.

        Given the obvious benefits of Idea Lab culture, it’s odd that we ever
        go for the alternative.

        We eat Skittles because our Primitive Minds are programmed to want
        sugary, calorie-dense food.

        But why do our Primitive Minds want us to build Echo Chambers?

        Let’s zoom out further.

        GIANTS Billions of years ago, some single-celled creatures realized
        that being just one cell left your options pretty limited.

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

        Facts and Questions:

        F: Motivated reasoning becomes obligated reasoning when the Primitive Mind infiltrates the reasoning process.
        Q: What happens when the Primitive Mind infiltrates the reasoning process?

        F: The Attorney's hypothesis formation stage is a belief-strengthening process.
        Q: What is the result of the Attorney's hypothesis formation stage?

        F: The Attorney's opponents will feel like they're arguing with a brick wall.
        Q: What is the experience of the Attorney's opponents?

        F: The result of thinking like an Attorney is that the brain's ability to learn new things is mostly shut down.
        Q: What impact does thinking like an Attorney have on the brain's ability to learn?

        F: Beliefs held by the Primitive Mind can be so strong that the Higher Mind has no influence over how they are thought about.
        Q: How strong can beliefs held by the Primitive Mind be?

        New Context:

        Idea Labs and Echo Chambers are two different intellectual cultures.
        In Idea Labs, humility is highly valued and ideas are treated like
        hypotheses, while in Echo Chambers, ideas are equated with a person's
        identity and disagreeing with someone is seen as rudeness. Taboo is a
        powerful tool for cultural law enforcement in Echo Chambers, and
        violators of sacred ideas are met with social fines, jail time, and
        even death penalty. Echo Chambers limit intellectual innovation and
        cause thinking skills to atrophy, while billions of years ago,
        single-celled creatures realized that being just one cell left their
        options limited.

        Now the real one:

        Context:
        {context}

        Chunk:
        {chunk}

        Facts and Questions: 
        """
    )
)