"""Removes what the LLM copies from its input into its reply, voice readings and tagged game snapshots or client inputs, so it is neither spoken nor kept in the context or the study log."""

import re

from loguru import logger
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
from pipecat.utils.text.pattern_pair_aggregator import MatchAction, PatternMatch, PatternPairAggregator

from agent.game import TAG as GAME_TAG
from agent.inputs import CLIENT_INPUTS

# A reading as expressive conversation writes it: bracketed categories in parentheses, such as ([tone neutral=0.99],[arousal value=0.30]).
READING = re.compile(r"\(\[\w+ [^\[\]()]*\](?:,\s*\[\w+ [^\[\]()]*\])*\)\s*")


async def strip_readings(text: str, text_type: str) -> str:
    """Remove every reading from a sentence of the LLM's reply, logging the sentence it was found in."""
    stripped = READING.sub("", text)
    if stripped != text:
        logger.warning(f"Stripped a voice reading the LLM wrote into its reply: {text!r}")
    return stripped


class _Aggregator(PatternPairAggregator):
    """Pipecat's pattern pair aggregator, except that the last sentence of a reply reaches the context without the space before it."""

    async def flush(self) -> PatternMatch | None:
        """Flush what is left of the reply, with the text the context records trimmed like the text that is spoken."""
        remaining = await super().flush()
        if remaining is None:
            return None
        return PatternMatch(content=remaining.text, type=remaining.type, full_match=remaining.full_match.strip())


async def _log_block(match: PatternMatch):
    """Log a tagged block the LLM wrote into its reply, which the aggregator has dropped."""
    logger.warning(f"Stripped a <{match.type}> block the LLM wrote into its reply: {match.full_match!r}")


def reply_strip() -> LLMTextProcessor:
    """Build the processor between the LLM and the TTS: it aggregates the reply by sentence, drops every tagged block, even one spanning sentences, and strips every reading."""
    aggregator = _Aggregator()
    for tag in (GAME_TAG, *CLIENT_INPUTS):
        aggregator.add_pattern(tag, f"<{tag}>", f"</{tag}>", MatchAction.REMOVE)
        aggregator.on_pattern_match(tag, _log_block)
    return LLMTextProcessor(text_aggregator=aggregator, text_transforms=[("*", strip_readings)])
