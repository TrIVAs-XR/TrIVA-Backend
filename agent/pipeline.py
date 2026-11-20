"""The frame pipeline. Receives its transport, its services, its prompt and its tools, and constructs none of them."""

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_transport import BaseTransport
from pipecat.turns.user_mute import MuteUntilFirstBotCompleteUserMuteStrategy
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import StrategyMode, UserTurnStrategies, UserTurnStrategiesParams
from pipecat.utils.types import NOT_GIVEN

from agent.frame_saver import FrameSaver
from agent.inputs import ContextCapture
from agent.vad_fallback import VADFallback, VADFallbackStopStrategy


def build(
    transport: BaseTransport,
    stt: STTService,
    llm: LLMService,
    tts: TTSService,
    system_prompt: str,
    tools: list[FunctionSchema],
    context_capture: ContextCapture,
    frame_saver: FrameSaver | None = None,
) -> Pipeline:
    """Assemble the pipeline. The only place it is assembled. The context capture sits on both sides of the LLM; a frame saver, when given, sees the transport's input first."""
    context = LLMContext(messages=[{"role": "system", "content": system_prompt}], tools=tools or NOT_GIVEN)

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3()),
                    VADFallbackStopStrategy(),
                ]
            ),
            # Either stop strategy ends the turn on its own, rather than Pipecat's default of waiting for both.
            user_turn_strategies_params=UserTurnStrategiesParams(stop_mode=StrategyMode.FIRST),
            # Held muted from connect until the greeting finishes, so nothing said before or during it reaches the LLM.
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        ),
    )

    return Pipeline(
        [
            transport.input(),
            *([frame_saver] if frame_saver is not None else []),
            stt,
            VADFallback(),
            aggregators.user(),
            context_capture.before_llm(),
            llm,
            context_capture.after_llm(),
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
