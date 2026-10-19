"""Voice assistant (optional, off by default).

Listens on the microphone, transcribes with faster-whisper and turns phrases
such as "volume up", "open notepad" or "type hello world" into the same action
dicts that ``main.py`` handles in ``_on_voice_command``.

Everything heavy (faster-whisper, sounddevice, numpy audio) is imported lazily,
and :meth:`VoiceAssistant.is_available` reports whether the stack is present so
the engine can skip starting the listener.
"""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from utils import deep_merge, module_available, try_import

logger = logging.getLogger(__name__)

DEFAULT_VOICE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "model_size": "tiny.en",
    "device": "cpu",
    "compute_type": "int8",
    "language": "en",
    "command_cooldown": 1.0,
    "listen_timeout": 3.0,
    "sample_rate": 16000,
    "block_seconds": 0.5,
    "silence_seconds": 0.7,
    "energy_threshold": 0.012,      # RMS of normalised float32 audio
    "max_chunk_seconds": 6.0,
    "input_device": None,           # sounddevice device id / name, None = default
    "wake_word": "",                # optional: only act after this word
}

#: Phrase -> action grammar.  Order matters: longest / most specific first.
COMMAND_PATTERNS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    (r"\bright click\b", {"action": "right_click"}),
    (r"\bdouble click\b", {"action": "double_click"}),
    (r"\btriple click\b", {"action": "select_paragraph"}),
    (r"\b(single )?click\b|\btap\b", {"action": "click"}),
    (r"\bscroll (up|down|left|right)\b", {"action": "scroll", "direction": 1}),
    (r"\bvolume (up|louder|higher)\b", {"action": "volume_up"}),
    (r"\bvolume (down|lower|quieter)\b", {"action": "volume_down"}),
    (r"\bmute\b|\bunmute\b", {"action": "mute"}),
    # GestureOS control must be matched before the generic "play/pause" rule.
    (r"\bpause gesture ?os\b|\bstop tracking\b|\bfreeze\b", {"action": "pause"}),
    (r"\bresume gesture ?os\b|\bstart tracking\b|\bunfreeze\b", {"action": "resume"}),
    (r"\b(play|pause|resume playback)\b", {"action": "play_pause"}),
    (r"\bnext (track|song)\b", {"action": "next_track"}),
    (r"\bprevious (track|song)\b|\bgo back\b", {"action": "prev_track"}),
    (r"\balt tab\b|\bswitch (window|app)\b|\bnext window\b", {"action": "alt_tab"}),
    (r"\btask view\b|\bshow all windows\b", {"action": "task_view"}),
    (r"\bshow desktop\b|\bminimize everything\b", {"action": "show_desktop"}),
    (r"\bminimize\b", {"action": "minimize"}),
    (r"\bmaximize\b", {"action": "maximize"}),
    (r"\bclose (window|this|it)\b|\bclose\b", {"action": "close"}),
    (r"\bstart menu\b", {"action": "start_menu"}),
    (r"\bbrightness (up|down)\b", {"action": "brightness", "direction": 1}),
    (r"\bzoom (in|out)\b", {"action": "zoom", "direction": 1}),
    (r"\bscreenshot\b|\bsnap\b", {"action": "screenshot"}),
    (r"\bpause( gestureos)?\b|\bstop tracking\b", {"action": "pause"}),
    (r"\bresume( gestureos)?\b|\bstart tracking\b", {"action": "resume"}),
    (r"\btype (.+)\b", {"action": "type", "text": 1}),
    (r"\bopen (.+)\b|\blaunch (.+)\b|\bstart (.+)\b", {"action": "open", "app": 1}),
    (r"\bsearch (for )?(.+)\b", {"action": "search", "query": 2}),
)

_DIRECTION_MAP = {"up": +1, "louder": +1, "higher": +1, "in": +1, "down": -1, "lower": -1, "quieter": -1, "out": -1,
                  "left": -1, "right": +1}


def parse_command(text: str, wake_word: str = "") -> Optional[Dict[str, Any]]:
    """Turn a transcript into an action dict (``None`` when nothing matched)."""
    if not text:
        return None
    cleaned = re.sub(r"[^\w\s]", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        return None

    if wake_word:
        marker = wake_word.lower().strip()
        if marker and marker not in cleaned:
            return None
        cleaned = cleaned.split(marker, 1)[-1].strip()
        if not cleaned:
            return None

    for pattern, template in COMMAND_PATTERNS:
        match = re.search(pattern, cleaned)
        if not match:
            continue
        command: Dict[str, Any] = {}
        params: Dict[str, Any] = {}
        for key, value in template.items():
            if isinstance(value, int) and value > 0:
                captured = ""
                for group in match.groups():
                    if group:
                        captured = group.strip()
                        break
                if value > 1 and match.lastindex and match.lastindex >= value:
                    captured = (match.group(value) or "").strip()
                if key == "direction":
                    params["direction"] = "up" if _DIRECTION_MAP.get(captured, 1) > 0 else "down"
                else:
                    params[key] = captured
            else:
                command[key] = value
        if command.get("action") in ("scroll", "zoom", "brightness"):
            direction = params.get("direction", "up")
            command["action"] = f"{command['action']}_{direction}"
        command["params"] = params
        command["text"] = cleaned
        command["confidence"] = 1.0
        return command
    return None


class VoiceAssistant:
    """Microphone listener + Whisper transcription on a worker thread."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config: Dict[str, Any] = deep_merge(dict(DEFAULT_VOICE_CONFIG), config or {})
        self._model = None
        self._thread: Optional[threading.Thread] = None
        self._stream = None
        self._running = False
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=64)
        self._last_command_at = 0.0
        self._lock = threading.Lock()
        self.transcripts: List[Dict[str, Any]] = []
        self.missing: List[str] = []

    # ------------------------------------------------------------------ #
    def is_available(self) -> bool:
        """True when the audio + model dependencies are importable."""
        self.missing = [m for m in ("faster_whisper", "sounddevice", "numpy") if not module_available(m)]
        return not self.missing

    def load_model(self) -> bool:
        """Load the Whisper model (blocking, ~1-10 s depending on size)."""
        if self._model is not None:
            return True
        faster_whisper = try_import("faster_whisper")
        if faster_whisper is None:
            logger.warning("faster-whisper not installed - voice control disabled")
            return False
        try:
            self._model = faster_whisper.WhisperModel(
                str(self.config.get("model_size", "tiny.en")),
                device=str(self.config.get("device", "cpu")),
                compute_type=str(self.config.get("compute_type", "int8")),
            )
            logger.info("Whisper model '%s' loaded", self.config.get("model_size"))
            return True
        except Exception as exc:
            logger.error("Could not load Whisper model: %s", exc)
            self._model = None
            return False

    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        if self._running:
            return True
        if not self.is_available():
            logger.warning("Voice assistant unavailable (missing: %s)", ", ".join(self.missing))
            return False
        if not self.load_model():
            return False

        sounddevice = try_import("sounddevice")
        if sounddevice is None:
            return False
        try:
            self._stream = sounddevice.RawInputStream(
                samplerate=int(self.config.get("sample_rate", 16000)),
                blocksize=int(int(self.config.get("sample_rate", 16000)) * float(self.config.get("block_seconds", 0.5))),
                device=self.config.get("input_device"),
                channels=1,
                dtype="float32",
            )
            self._stream.start()
        except Exception as exc:
            logger.error("Could not open microphone: %s", exc)
            self._stream = None
            return False

        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, name="GestureOS-Voice", daemon=True)
        self._thread.start()
        logger.info("Voice assistant listening (wake word: %s)", self.config.get("wake_word") or "none")
        return True

    def stop(self) -> None:
        self._running = False
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("Error closing audio stream: %s", exc)
            self._stream = None
        logger.info("Voice assistant stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ #
    def _listen_loop(self) -> None:
        import numpy as np

        sample_rate = int(self.config.get("sample_rate", 16000))
        threshold = float(self.config.get("energy_threshold", 0.012))
        silence_limit = float(self.config.get("silence_seconds", 0.7))
        max_seconds = float(self.config.get("max_chunk_seconds", 6.0))

        chunks: List[Any] = []
        voiced_seconds = 0.0
        silent_seconds = 0.0

        while self._running:
            try:
                data, overflowed = self._stream.read(
                    int(sample_rate * float(self.config.get("block_seconds", 0.5)))
                )
            except Exception as exc:
                logger.error("Microphone read failed: %s", exc)
                break
            if overflowed:
                logger.debug("Audio input overflow (%d frames dropped)", overflowed)

            block = np.frombuffer(data, dtype=np.float32)
            if block.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(block))))
            block_seconds = float(self.config.get("block_seconds", 0.5))

            if rms >= threshold:
                chunks.append(block)
                voiced_seconds += block_seconds
                silent_seconds = 0.0
            else:
                if chunks:
                    silent_seconds += block_seconds
                    chunks.append(block)

            should_transcribe = chunks and (
                silent_seconds >= silence_limit or voiced_seconds >= max_seconds
            )
            if should_transcribe:
                audio = np.concatenate(chunks)
                chunks, voiced_seconds, silent_seconds = [], 0.0, 0.0
                self._transcribe(audio, sample_rate)

        if chunks:
            try:
                self._transcribe(np.concatenate(chunks), sample_rate)
            except Exception:  # pragma: no cover
                pass

    def _transcribe(self, audio, sample_rate: int) -> None:
        try:
            segments, _info = self._model.transcribe(
                audio,
                language=self.config.get("language") or None,
                beam_size=1,
                vad_filter=False,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            return

        if not text:
            return
        command = parse_command(text, str(self.config.get("wake_word", "")))
        entry = {"text": text, "t": time.time(), "command": command}
        self.transcripts.append(entry)
        if len(self.transcripts) > 200:
            del self.transcripts[: len(self.transcripts) - 200]

        if command is None:
            logger.debug("Voice: no command in '%s'", text)
            return

        cooldown = float(self.config.get("command_cooldown", 1.0))
        now = time.time()
        with self._lock:
            if now - self._last_command_at < cooldown:
                logger.debug("Voice command ignored (cooldown): %s", text)
                return
            self._last_command_at = now

        logger.info("Voice command: %s", command)
        try:
            self._events.put_nowait({"text": text, "action": command, "timestamp": now})
        except queue.Full:
            logger.warning("Voice event queue full - dropping command")

    # ------------------------------------------------------------------ #
    def get_event(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """Pop the next voice event, or ``None`` after ``timeout`` seconds."""
        try:
            return self._events.get(timeout=max(0.0, float(timeout)))
        except queue.Empty:
            return None

    def pending(self) -> int:
        return self._events.qsize()

    def last_transcript(self) -> Optional[str]:
        return self.transcripts[-1]["text"] if self.transcripts else None


__all__ = ["VoiceAssistant", "parse_command", "DEFAULT_VOICE_CONFIG"]
