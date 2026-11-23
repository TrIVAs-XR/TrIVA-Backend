"""The frame pipeline. Receives its transport, its services, its tools and the processors the session reads, and constructs none of them."""

from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.expressive.processors.voice_reader import PipecatVoiceReader
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_text_processor import LLMTextProcessor
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
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.utils.types import NOT_GIVEN

from agent.camera import Camera
from agent.frame_saver import FrameSaver
from agent.inputs import ContextCapture
from agent.reading_strip import strip_readings
from agent.vad_fallback import VADFallback


def build(
    transport: BaseTransport,
    stt: STTService,
    llm: LLMService,
    tts: TTSService,
    tools: list[FunctionSchema],
    camera: Camera,
    context_capture: ContextCapture,
    frame_saver: FrameSaver | None = None,
) -> Pipeline:
    """Assemble the pipeline. The only place it is assembled. A frame saver, when given, and then the camera see the transport's input first, the VAD fallback and the voice reader sit between STT and the user aggregator, and the context capture on both sides of the LLM."""
    context = LLMContext(tools=tools or NOT_GIVEN)

    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]),
            # Held muted from connect until the greeting finishes, so nothing said before or during it reaches the LLM.
            user_mute_strategies=[MuteUntilFirstBotCompleteUserMuteStrategy()],
        ),
    )

    return Pipeline(
        [
            transport.input(),
            *([frame_saver] if frame_saver is not None else []),
            camera,
            stt,
            VADFallback(),
            # Hears how each user turn sounded and puts the reading at the head of its transcription.
            PipecatVoiceReader(),
            aggregators.user(),
            context_capture.before_llm(),
            llm,
            context_capture.after_llm(),
            # Takes a voice reading the LLM copied into its reply out of the speech, the context and the study log.
            LLMTextProcessor(text_transforms=[("*", strip_readings)]),
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
