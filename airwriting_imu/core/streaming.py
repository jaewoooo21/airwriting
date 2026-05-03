"""
streaming.py — 연속 문장 스트리밍 인식 로직

프론트엔드 의존 없이, 백엔드에서 센서 데이터를 실시간으로 모아
글자를 판별하고 문장을 이어나가는 StreamingInference 모듈입니다.
"""

import time
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class StreamingInference:
    def __init__(
        self,
        ai_engine,
        debounce_time: float = 0.8,
        char_timeout: float = 0.6,
        space_timeout: float = 2.0,
    ):
        """
        Args:
            ai_engine: AirWritingAI 인스턴스 (predict 메서드 보유)
            debounce_time: 동일 문자의 중복 출력을 무시하는 최소 시간 (초)
            char_timeout: is_writing=False가 유지되면 한 글자로 판정할 시간 (초)
            space_timeout: is_writing=False가 유지되면 띄어쓰기로 판정할 시간 (초)
        """
        self.engine = ai_engine
        self.debounce_time = debounce_time
        self.char_timeout = char_timeout
        self.space_timeout = space_timeout

        self._buffer = []
        self._frame_count = 0

        # State tracking
        self._last_emitted_char: Optional[str] = None
        self._last_emit_time: float = 0.0
        self._last_writing_time: float = time.time()
        self._space_emitted = False

        self.on_text_updated: Optional[Callable[[str, str], None]] = None
        self._current_sentence = ""

    def process_frame(self, frame_data: dict, is_writing: bool):
        """매 프레임 호출되어 텍스트 스트리밍을 제어합니다."""
        now = time.time()

        if is_writing:
            self._last_writing_time = now
            self._space_emitted = False
            self._frame_count += 1

            # 기존 8채널 (ax, ay, az, gx, gy, gz, x, y) 데이터 버퍼링
            self._buffer.append(frame_data)

            if len(self._buffer) > 1000:
                self._buffer.pop(0)

        else:    # 글씨를 안 쓰고 있을 때 타임아웃 판정
            duration_since_writing = now - self._last_writing_time
            
            # 1. 단일 글자 완성 판정 (char_timeout)
            if duration_since_writing > self.char_timeout and len(self._buffer) > 0:
                if len(self._buffer) > 10:  # 너무 짧은 노이즈(더블클릭 연타 등) 무시
                    self._run_inference()
                self._buffer.clear() # 추론 완료 또는 노이즈면 버퍼 비움

            # 2. 띄어쓰기 판정 (space_timeout)
            if duration_since_writing > self.space_timeout and not self._space_emitted:
                if len(self._current_sentence) > 0 and not self._current_sentence.endswith(" "):
                    self._emit_char(" ")
                self._space_emitted = True
                self._last_emitted_char = None # 새 단어이므로 중복방지 해제

    def _run_inference(self):
        """엔진을 돌려 문자를 가져오고 디바운싱합니다. (Asyncio 이벤트 루프 차단 방지)"""
        if not self.engine or len(self._buffer) == 0:
            return

        # ai_model의 predict는 [strokes] 형태의 입력을 기대함
        # 리스트 깊은 복사를 통해 비동기 처리 중 데이터 변조 방지
        session_strokes = [list(self._buffer)]
        
        # AI 연산이 길어지면 수신 루프가 멈춰 통신이 끊어지므로 별도 스레드로 격리
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, self._do_predict_and_emit, session_strokes)
        except RuntimeError:
            self._do_predict_and_emit(session_strokes)

    def _do_predict_and_emit(self, session_strokes):
        # 모델 추론 (Heavy computation)
        predicted_word = self.engine.predict(session_strokes)

        if predicted_word:
            char = predicted_word
            now = time.time()

            # Debouncing: 동일 글자가 짧은 시간 내 다시 인식되면 무시
            is_duplicate = (char == self._last_emitted_char) and ((now - self._last_emit_time) < self.debounce_time)
            
            if not is_duplicate:
                self._emit_char(char)
                self._last_emitted_char = char
                self._last_emit_time = now

    def _emit_char(self, char: str):
        if char == "<ERASE>":
            if len(self._current_sentence) > 0:
                self._current_sentence = self._current_sentence[:-1]
            logger.info(f"🔙 Stream Erased -> Sentence: '{self._current_sentence}'")
        else:
            self._current_sentence += char
            logger.info(f"📝 Stream Emitted: '{char}' -> Sentence: '{self._current_sentence}'")
            
        if self.on_text_updated:
            self.on_text_updated(self._current_sentence, char)

    def reset(self):
        self._buffer.clear()
        self._current_sentence = ""
        self._last_emitted_char = None
        self._space_emitted = False
        if self.on_text_updated:
            self.on_text_updated("", "")
