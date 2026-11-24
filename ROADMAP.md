# Roadmap

Upcoming work only. What is built is described by the code and ARCHITECTURE.md.

## Lab tutor

The lab scenario runs with the Unity client. Still open:

- Whether Iva talks too much, tuned from the `game_event` and turn lines in the study logs. The reaction table in `agent/scenarios/lab/__init__.py` and the prompt are the two places to turn it down.
- Whether the 0.3 s grace and the check that a participant's own turn already answered a snapshot hold up against real pipeline timing.
- A participant's self-talk still triggers a reply every time.
- A prose history of past events, if the newest snapshot proves too little.
- A reply whose text never becomes audio, such as a failing TTS, leaves the agent busy for good, and no reaction runs after it.
- So does a reply that is only emoji: the TTS leaves nothing to say, but `Inputs` and `Game` took the reply strip's words and wait for speech, so the input record stays open and a held reaction waits until the next turn or interruption.

## Later

- Conditions and the balancing of participants across them, once the next study defines them: interaction modality, explanation behaviour, embodiment and nonverbals, voice and speech style. A demo condition that skips the study, once there is something to skip.
- The microphone gated on the handshake. Audio input starts on its own today (`audio_in_stream_on_start`); once the Unity client sends `client-ready` that flips, so no participant audio is processed before the scene is up.
- Camera frames for the LLM. Unity publishes its camera, `--vision` decides whether the agent subscribes, and `Camera` holds the newest frame, but no frame reaches the LLM yet: the input selection in `agent/inputs.py` is to decide per call whether to attach `Camera.latest`.
- A PII filter, should a study need one. Its placeholders, such as `[name]`, are filled back in for speech only, as a text transform on the TTS, so the context and the study log keep the placeholder; the reply strip removes text everywhere and is not the place.
- Transport comparison, deferred: `agent/transports/websocket.py` using Pipecat's `WebsocketServerTransport`, behind the existing `--transport` flag. Only meaningful if the Unity side logs its timeline identically on both paths, and only worth doing if a transport problem turns up.
