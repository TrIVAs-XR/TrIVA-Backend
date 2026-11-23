"""Removes voice readings the LLM copies into its reply, so a reading is neither spoken nor kept in the context or the study log."""

import re

from loguru import logger

# A reading as expressive conversation writes it: bracketed categories in parentheses, such as ([tone neutral=0.99],[arousal value=0.30]).
READING = re.compile(r"\(\[\w+ [^\[\]()]*\](?:,\s*\[\w+ [^\[\]()]*\])*\)\s*")


async def strip_readings(text: str, text_type: str) -> str:
    """Remove every reading from a sentence of the LLM's reply, logging the sentence it was found in."""
    stripped = READING.sub("", text)
    if stripped != text:
        logger.warning(f"Stripped a voice reading the LLM wrote into its reply: {text!r}")
    return stripped
