"""Token variant lists for token-level answer probability marginalization.

When computing the probability that a model selected a particular answer (e.g. "A"),
we sum the probabilities across all formatting variants the model might produce
(e.g. "A", " a", "[A]", "(A)"). This accounts for variations in casing, leading
whitespace, and markdown formatting that differ across models.

Each list below is generated from one of a few repeating shapes (case, bracket/
markdown formatting, leading whitespace) rather than typed out by hand, so adding
a new letter/word/phrase is a one-line call instead of a new manually-enumerated
list."""


def _with_leading_space_variants(tokens: list[str]) -> list[str]:
    """Add a leading-space copy of each token (some tokenizers encode a token
    differently right after whitespace than at the start of a string)."""
    return tokens + [f" {t}" for t in tokens]


def letter_tokens(letter: str) -> list[str]:
    """
    Token variants for a single-letter answer (e.g. "A"): bare upper/lower case,
    plus bracket/parenthesis/markdown formatting of the upper-case letter, each
    with and without a leading space.
    """
    upper = letter.upper()
    lower = letter.lower()
    formatted = []
    for l in [upper, lower]:
        formatted.extend([f"{l}", f"[{l}]", f"({l})", f"*{l}*", f"**{l}**"])  # options cannot share prefixes because otherwise probabilities would be double-counted
    return _with_leading_space_variants(formatted)


def word_tokens(word: str) -> list[str]:
    """
    Token variants for a word answer (e.g. "Yes"): bare title/lower/upper case,
    plus bracket/parenthesis/markdown formatting of the title-case word, each
    with and without a leading space.
    """
    bare = [word, word.lower(), word.upper()]
    formatted = [f"[{word}]", f"({word})", f"*{word}*", f"**{word}**"]
    return _with_leading_space_variants(bare + formatted)


def contraction_phrase_tokens(word: str) -> list[str]:
    """
    Token variants for an "It's <word>" answer: contraction vs. spelled-out
    ("It's"/"It is"), each in title and lower case, each with and without a
    leading space.
    """
    bare = [f"It's {word}", f"It is {word}", f"it's {word}", f"it is {word}"]
    return _with_leading_space_variants(bare)


def verdict_letter_tokens(letter: str) -> list[str]:
    """
    Token variants for a "Verdict: <letter>" answer, used by RedditPrompter's
    per-scenario randomized letter mapping. Unlike letter_tokens(), this includes
    the "Verdict: " prefix, because the reddit answer format asks for
    "Verdict: <letter>" followed by reasoning -- the letter is not the model's
    first generated token, so scoring it bare (against position-0 logits) would
    almost always read as ~0 regardless of the model's actual confidence.

    Also covers bracket/markdown formatting of the letter itself (e.g.
    "Verdict: **D**"), mirroring letter_tokens() -- models (observed: Gemma)
    sometimes bold the letter, and omitting those variants silently undercounts
    probability mass for every row where that happens.
    """
    letter_variants = letter_tokens(letter)
    return [f"Verdict: {v}" for v in letter_variants] + [f"verdict: {v}" for v in letter_variants]


A_TOKENS = letter_tokens("A")
B_TOKENS = letter_tokens("B")
C_TOKENS = letter_tokens("C")
D_TOKENS = letter_tokens("D")
E_TOKENS = letter_tokens("E")
# LETTER_TOKENS[n] gives the ordered token-variant lists for the first n letters,
# used by NormBankPrompter to score a per-scenario random letter mapping (see that
# module's docstring for why the mapping must be randomized). Valid where the letter
# is expected to be the model's very first generated token (e.g. "Respond only with
# a single letter...").
LETTER_TOKENS = [A_TOKENS, B_TOKENS, C_TOKENS, D_TOKENS, E_TOKENS]

YES_TOKENS = word_tokens("Yes")
NO_TOKENS = word_tokens("No")

GOOD_TOKENS = contraction_phrase_tokens("good")
OK_TOKENS = contraction_phrase_tokens("acceptable")
BAD_TOKENS = contraction_phrase_tokens("wrong")
