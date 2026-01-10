

Two datasets:

code_qa.json  and history_qa.json in this directory. DO NOT read them as they are very large. Completely ignore data.json

code_qa.json snippet:

{
        "_id": "66fa208bbb02136c067c5fc1",
        "domain": "Code Repository Understanding",
        "sub_domain": "Code repo QA",
        "difficulty": "easy",
        "length": "long",
        "question": "In the function that calculates the derivative of given functions, which of the following keyword arguments are all recognized?",
        "choice_A": "singular, addprec, function",
        "choice_B": "h, method, direction",
        "choice_C": "relative, fc, y",
        "choice_D": "radius, x, step",
        "answer": "B",
        "context": "... (very long)"
}


history_qa.json sample:

{
        "_id": "671b1335bb02136c067d4e88",
        "domain": "Long-dialogue History Understanding",
        "sub_domain": "Agent history QA",
        "difficulty": "easy",
        "length": "short",
        "question": "Which player won the most times in the game?",
        "choice_A": "player_2",
        "choice_B": "player_4",
        "choice_C": "player_6",
        "choice_D": "player_8",
        "answer": "C",
        "context": "..."
    },