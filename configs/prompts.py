#CORE RAG APP
SYS_TEMPLATE = """
    You are a helpful assistant with access to storage notes.

    Use only the provided context to answer the user's questions. If you reference a specific piece of context, include a reference number like [1], [2], etc.

    If the context is missing, irrelevant, or insufficient, say "I don’t know based on the provided information."

    Do not guess or make assumptions. If the user is just making conversation, respond conversationally.

    Below is the context:
"""

#GENERATE QUESTION ANSWER PAIRS
QA_GENERATION_PROMPT = """
    You’re a data generation assistant that outputs only json array objects. Your task is to produce only a **valid JSON list** of exactly 3 question-answer pairs about the following passage.

    **Each question must be:**
    - Fact-based
    - Non-trivial
    - Diverse in content

    **Your response must folow these strict rules**
    - Only return a valid JSON list
    - Do not include any text outside of the JSON list
    - Do not wrap the JSON in Markdown or code blocks
    - Do not repeat these instructions
    - The keys in each object must be "question" and "answer"

    ###Required output format:###
    [
    {{
        "question": "What is the primary goal of reinforcement learning?",
        "answer": "The primary goal of reinforcement learning is to learn a policy that maximizes cumulative reward through trial-and-error interactions with an environment."
    }},
    ...
    ]

    ###Passage:###
    {context}
"""

#GENERATE QUESTION ANSWER PARIS WITHOUT DUPLICATION
DEDUP_QA_GENERATION_PROMPT = """
    You’re a data generation assistant that outputs only json array objects. Your task is to produce only a **valid JSON list** of exactly 3 question-answer pairs about the following passage.

    **Each question must be:**
    - Fact-based
    - Non-trivial
    - Diverse in content

    **Your response must folow these strict rules**
    - Only return a valid JSON list
    - Do not include any text outside of the JSON list
    - Do not wrap the JSON in Markdown or code blocks
    - Do not repeat these instructions
    - The keys in each object must be "question" and "answer"
    - Do not repeat, rephrase, paraphrase any of the following existing QA pairs
    - Each QA pair must be unique

    Do **not** repeat, rephrase, or paraphrase any of these existing QA pairs:
    {pairs}

    ###Required output format:###
    [
    {{
        "question": "What is the primary goal of reinforcement learning?",
        "answer": "The primary goal of reinforcement learning is to learn a policy that maximizes cumulative reward through trial-and-error interactions with an environment."
    }},
    ...
    ]

    ###Passage:###
    {context}
"""

#JUDGE WHETHER CHUNKS ARE RELEVANT TO A QUESTION-ANSWER PAIR
RELEVANCE_DETERMINATION_PROMPT = """
    You are a relevance determination assistant that outputs **only** JSON array objects.

    Given the following question-answer pair and chunked document, your task is to create a list of indexes of the chunks required to answer the question.

    **Strict rules for your response**
    - Only return a valid JSON list
    - Do not include any text outside of the JSON list
    - Do not wrap the JSON in Markdown or code blocks
    - Do not repeat these instructions
    - The keys in each object must be "question", "answer", "relevant_chunks"

    **Important note**
    - Some text will have indexes of sources that look like [index], the chunk indexes have been annotated as [CHUNK-index].

    ###Required output format:###
    {{
        "question": question,
        "answer": answer,
        "relevant_chunks" index_list

    Question: {question}
    Answer: {answer}

    Chunks:
    {chunked_doc}
"""
