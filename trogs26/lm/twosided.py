"""A language model that scores a text from both sides.

The decoder's search reads a row left to right, so a letter is proposed from the letters before it; a row of a
squeeze is a fragment, and its first letters have nothing before them. Every proposed text is scored once more for
the final choice (`decode.search.candidate_pool`), and there the whole text is known. `TwoSided` scores each letter
there from the letters before it and from the letters after it, taking the two as independent evidence about the
letter:

    log P(letter | before) + log P(letter | after) - log P(letter),

summed over the text. A row's first letter is then scored from what follows it alone, its last from what precedes
it alone. The search itself still reads the forward model.
"""
from __future__ import annotations


class TwoSided:
    """`forward` scores a letter from the text before it; `backward` from the text after it, given in reverse order
    (`lm.ppm.Backward`, or a mixture or cache of such). Both have `cond_logprob` and `text_logprob`."""

    def __init__(self, forward, backward):
        self.forward, self.backward = forward, backward

    def cond_logprob(self, letter: str, context: str) -> float:
        """log P(letter | context) of the forward model, which the search reads."""
        return self.forward.cond_logprob(letter, context)

    def text_logprob(self, text: str) -> float:
        """The two-sided score of `text` (module docstring)."""
        alone = sum(self.forward.cond_logprob(letter, '') for letter in text)
        return float(self.forward.text_logprob(text) + self.backward.text_logprob(text[::-1]) - alone)
