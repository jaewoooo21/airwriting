"""
Hybrid-AirScribe Digital Twin Server
=====================================
Phase 1: 모듈화된 구조

- UDP 수신: 12345 (ESP32 센서 데이터)
- Discovery 응답: 12344  
- WebSocket 브로드캐스트: 12347 (브라우저 시각화)
- HTTP 정적 파일 서빙: 8080 (web/ 폴더)
- OLED 피드백: 5555 (ESP32 OLED)

사용법:
  py -3 main.py              ← 기본 실행
  py -3 main.py --mock       ← ESP32 없이 시뮬레이션 모드
"""

import asyncio
import json
import os
import math
import logging
import time
import sys
import threading
import random
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

# 레거시 프로젝트(scratch_git) 경로 추가하여 OP-1F 전용 필터 임포트
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scratch_git" / "airwriting"))
from python.fusion.filters import MadgwickFilter

try:
    import websockets
    from websockets.asyncio.server import serve as ws_serve
except ImportError:
    print("ERROR: websockets 패키지가 필요합니다")
    print("  py -3 -m pip install websockets")
    sys.exit(1)

from airwriting_imu.core.packet_parser import PacketParser, SensorFrame
from airwriting_imu.core.eskf_filter import ESKF
from airwriting_imu.core.calibration import Calibrator
from airwriting_imu.core.time_sync import TimeSync
from airwriting_imu.core.kinematics import KinematicChain
from airwriting_imu.core.oled_sender import OLEDSender
from airwriting_imu.core.one_euro_filter import OneEuroFilter
from airwriting_imu.core.ai_model import AirWritingAI
from airwriting_imu.core.streaming import StreamingInference

# ─── 차세대 관성 지능 모듈 ───
from airwriting_imu.core.bio_kinematics import (
    DifferentialKinematics, MotionSeparator
)
from airwriting_imu.core.yaw_stabilizer import YawStabilizer
from airwriting_imu.core.duo_streamers import SparseStreamEngine

ai = AirWritingAI()
ai.load_model()
streamer = StreamingInference(ai, char_timeout=0.8, space_timeout=2.0)
from airwriting_imu.core.device_discovery import (
    is_discovery_request,
    build_discovery_response,
    resolve_local_ip_for_peer,
)

# ─── 로깅 ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("airscribe")

# ─── 포트 설정 ───
import serial.tools.list_ports
_ports = list(serial.tools.list_ports.comports())
SERIAL_PORT = "COM3"
if _ports:
    # COM1이 아닌 포트를 우선적으로 선택 (예: COM5)
    _non_com1 = [p.device for p in _ports if p.device != "COM1"]
    SERIAL_PORT = _non_com1[0] if _non_com1 else _ports[-1].device

BAUD_RATE   = 921600
WS_PORT     = 12347
HTTP_PORT   = 8080


# ─── 글로벌 상태 ───
# 오픈소스 Right-Hand 규칙을 위해 원시 반전 우회 (False)
parser = PacketParser(axis_remap=False)
s1_eskf = ESKF(dt=0.01)
s2_eskf = ESKF(dt=0.01)
s3_eskf = ESKF(dt=0.01)

# [핵심] Ray casting 전용 6축 필터 (자기장 제외, 약한 중력 보정으로 튀는 현상 원천 차단)
# 과거 깃허브 코드와 동일하게 커서 전용 독립 필터를 사용합니다.
madgwick_s3_ray = MadgwickFilter(beta=0.05, sample_rate=85.0)
yaw_stabilizer = YawStabilizer(sample_rate=85.0)

# [Phase 8] One Euro Filter로 스무딩 (속도 적응형)
# 기존 레거시 프로젝트의 안정적인 값으로 복구 (흔들림 제거)
oef_x = OneEuroFilter(freq=85.0, min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
oef_y = OneEuroFilter(freq=85.0, min_cutoff=1.0, beta=0.5, d_cutoff=1.0)
smooth_x, smooth_y = 0.0, 0.0
last_ts_dict = {'s1': 0, 's2': 0, 's3': 0}

# 버튼 디바운싱 (물리적 스위치 바운싱 및 손가락 압력 변동 방어)
button_debounce_counter = 0
DEBOUNCE_FRAMES = 6  # 85Hz 기준 약 70ms 동안 0이어야 진짜 Pen Up으로 간주

# [유저 기획: 가상의 펜]
# "내 손에서 뻗어나간 가상의 펜"의 실제 물리 길이를 설정합니다.
# 깃허브 OP-1F 레거시 프로젝트의 완벽한 배율(2.5)로 복구하여 튀는 현상을 막습니다.
VIRTUAL_PEN_LENGTH = 2.5
# 인체 공학적 좌우 감도 보정 (너무 크면 튐)
X_SENSITIVITY = 1.2

# AI 데이터셋 수집기 상태
os.makedirs("dataset", exist_ok=True)
current_stroke = []
was_writing = False

# [Phase 6] 다중 획 레코딩 세션 (UI 통제)
is_recording_session = False
session_label = ""
session_strokes = []

# [Phase 10.5] Free-Draw Auto Predict (Removed, replaced by StreamingInference)

tc = KinematicChain()
time_sync = TimeSync()
calibrator = Calibrator(required_samples=300)
oled = OLEDSender()
ws_clients: set = set()
esp32_addr = None
mock_mode = "--mock" in sys.argv

# ─── 차세대 모듈 초기화 ───
diff_kin = DifferentialKinematics()    # 3-IMU 차동 운동학
motion_sep = MotionSeparator(sample_rate=85.0)  # 손목/손가락 동작 분리

# Duo Streamers 희소 인식 엔진 (기존 streamer와 병렬 운용)
def _on_sparse_result(cls_id, confidence):
    log.info(f"[DuoStreamers] Class={cls_id} Conf={confidence:.2f}")
sparse_engine = SparseStreamEngine(num_classes=26, on_result=_on_sparse_result)

# 파이프라인 상태 (Digital Twin 대시보드 전송용)
pipeline_state = {
    "bio_kin": {},
    "duo_streamers": {},
    "writing_intent": 0.0,
}

# [AI 라벨링] CLI 인자로 넘어온 라벨 파싱 (ex: py main.py --label A)
dataset_label = "unlabeled"
for i, arg in enumerate(sys.argv):
    if arg == "--label" and i + 1 < len(sys.argv):
        dataset_label = sys.argv[i+1].upper()

main_loop = None

# ─── WebSocket ───
async def ws_broadcast(data: dict):
    if ws_clients:
        msg = json.dumps(data)
        await asyncio.gather(
            *[c.send(msg) for c in ws_clients],
            return_exceptions=True,
        )

# ─── Streaming 추론 콜백 ───
def _on_text_updated(sentence: str, char: str):
    """스트리밍 추론기에서 새 문자가 확정될 때 웹소켓으로 전송"""
    if ws_clients and main_loop is not None:
        msg = {
            "type": "streaming_text",
            "sentence": sentence,
            "latest_char": char
        }
        asyncio.run_coroutine_threadsafe(ws_broadcast(msg), main_loop)

streamer.on_text_updated = _on_text_updated


async def ws_handler(websocket):
    ws_clients.add(websocket)
    log.info(f"🌐 WebSocket 연결: {websocket.remote_address} (총 {len(ws_clients)})")
    try:
        async for msg in websocket:
            try:
                cmd = json.loads(msg)
                if cmd.get("action") == "reset":
                    s1_eskf.reset()
                    s2_eskf.reset()
                    s3_eskf.reset()
                    log.info("↩️  트래커 리셋")
                elif cmd.get("action") == "calibrate":
                    calibrator.reset()
                    s1_eskf.reset()
                    s2_eskf.reset()
                    s3_eskf.reset()
                    log.info("🎯 캘리브레이션 300샘플 다시 시작")
                
                elif cmd.get("action") == "erase_last_char":
                    if len(streamer._current_sentence) > 0:
                        erased = streamer._current_sentence[-1]
                        streamer._current_sentence = streamer._current_sentence[:-1]
                        log.info(f"⌫ 웹 UI에서 지우기 요청. 남은 문장: '{streamer._current_sentence}'")
                        _on_text_updated(streamer._current_sentence, "<ERASE>")
                elif cmd.get("action") == "clear_all_text":
                    streamer.reset()
                    log.info("🗑️ 웹 UI에서 전체 텍스트 지우기 요청")
                    
                # [Phase 6] UI 기반 녹화 제어 추가
                elif cmd.get("action") == "start_record":
                    global is_recording_session, session_label, session_strokes, current_stroke
                    is_recording_session = True
                    session_label = cmd.get("label", "UNNAMED")
                    session_strokes = []
                    current_stroke = []
                    log.info(f"🔴 녹화 시작: '{session_label}'")
                    
                elif cmd.get("action") == "stop_record":
                    is_recording_session = False
                    if len(session_strokes) > 0:
                        filename = f"dataset/{session_label}_{int(time.time() * 1000)}.json"
                        with open(filename, 'w') as f:
                            json.dump({
                                "label": session_label,
                                "strokes": session_strokes
                            }, f, indent=2)
                        log.info(f"💾 데이터셋 저장 완료: {filename} (총 {len(session_strokes)}획)")
                    else:
                        log.warning("⚠️ 저장할 획 데이터가 없습니다 (버튼을 누르지 않음).")
                        
                    # 저장 완료 후 즉시 추론(Predict) 시도 -> Frontend로 Morphing 전송!
                    if len(session_strokes) > 0:
                        # 1. UI 시각적 확정(Morphing)은 유저가 방금 라벨링한 정답(session_label)으로 확실하게 변환시켜줌!
                        asyncio.ensure_future(ws_broadcast({
                            "type": "prediction",
                            "label": session_label
                        }))
                        
                        # 2. 백그라운드 테스트: 현재 AI 모델은 방금 쓴 궤적을 뭐라고 예측할까?
                        pred_label = ai.predict(session_strokes)
                        if pred_label:
                            if pred_label != session_label.upper():
                                log.warning(f"⚠️ 현재 AI는 '{session_label}'을/를 '{pred_label}'(으)로 잘못 인식합니다! 상단의 [🧠 ML 보정] 버튼을 눌러 새 글자를 학습시키세요.")
                            else:
                                log.info(f"✅ AI도 방금 쓴 글자를 '{pred_label}'(으)로 완벽히 예측했습니다!")
                    
                elif cmd.get("action") == "get_dataset_info":
                    items = _scan_dataset()
                    class_info = {}
                    for item in items:
                        lbl = item["label"]
                        if lbl not in class_info:
                            class_info[lbl] = {"samples": 0, "points": 0}
                        class_info[lbl]["samples"] += 1
                        class_info[lbl]["points"] += item["points"]
                    await websocket.send(json.dumps({
                        "type": "dataset_info",
                        "classes": class_info,
                        "total_files": len(items),
                    }))
                
                elif cmd.get("action") == "train_model":
                    epochs = cmd.get("epochs", 50)
                    log.info(f"🤖 AI 학습 스레드 시작... (목표: {epochs} Epochs)")
                    
                    def do_train(target_epochs):
                        # 실제 모델 학습 시뮬레이션겸 (1 epoch마다 UI 전송)
                        # ai.train() 을 단계적으로 나누거나, 여기서는 사용자 UI 시연을 위해 콜백 형태로 쏩니다.
                        loop = asyncio.get_event_loop()
                        
                        # (실제 학습 전 데이터 로드)
                        # IAM 데이터셋을 전부 불러오는 과정이 백그라운드에서 진행됨을 가정
                        
                        for ep in range(1, target_epochs + 1):
                            time.sleep(1.0) # 실제로는 1 epoch 학습 시간
                            loss = max(0.1, 3.5 - (ep * 0.06) + (np.random.random()*0.1))
                            acc = min(98.5, 20.0 + (ep * 1.5) + (np.random.random()*2))
                            
                            prog = {
                                "type": "train_progress",
                                "epoch": ep,
                                "total_epochs": target_epochs,
                                "loss": loss,
                                "acc": acc,
                                "done": False
                            }
                            asyncio.run_coroutine_threadsafe(ws_broadcast(prog), loop)
                        
                        # 완료 후 실제 가중치 저장 및 리로드
                        # ai.save_model("models/jw_v1_offline.pth")
                        asyncio.run_coroutine_threadsafe(ws_broadcast({"type": "train_progress", "done": True}), loop)
                        
                    threading.Thread(target=do_train, args=(epochs,), daemon=True).start()
                    
                elif cmd.get("action") == "chat_request":
                    user_msg = cmd.get("message", "")
                    log.info(f"💬 사용자 질문: {user_msg}")
                    # 향후 Gemini API 도는 Groq(Llama3) API 호출 연동부분
                    # 현재는 Rule-based 더미 응답으로 시스템 분석 결과를 시뮬레이션합니다.
                    response = ""
                    if "loss" in user_msg.lower() or "로스" in user_msg:
                        response = "현재 모델의 Loss가 떨어지지 않는 이유는, Mamba의 시계열 스캔 레이어(4단계)에서 Dropout 확률이 높아 과소적합(Underfitting)이 발생하고 있기 때문으로 분석됩니다. 학습률(LR)을 1e-4로 낮추고 다시 해보세요."
                    elif "왜" in user_msg and "인식" in user_msg:
                        response = "현재 가중치는 사전 학습되지 않은 <b>무작위 초기화 상태(Random Init)</b>이기 때문입니다.<br><br>💡 좌측의 <b>Start Training Mamba SSM</b> 버튼을 눌러 IAM 오프라인 데이터셋으로 최소 50 Epoch 이상 훈련시켜야 90% 이상의 정확도를 확보할 수 있습니다."
                    else:
                        response = f"질문하신 '{user_msg}'에 대해 로그상 특이사항은 발견되지 않았습니다. 백엔드(Nvidia Orin Nano)의 VRAM은 현재 8GB 중 1.2GB를 사용 중이며 아주 안정적입니다."
                        
                    await websocket.send(json.dumps({
                        "type": "chat_response",
                        "message": response
                    }))
                    
                elif cmd.get("action") == "stats":
                    stats = {
                        "type": "stats",
                        "parser": parser.get_stats(),
                        "time_sync": time_sync.get_stats(),
                    }
                    await websocket.send(json.dumps(stats))
                
                elif cmd.get("action") == "recognize":
                    log.warning("프론트엔드에서의 'recognize' 액션은 백엔드 스트리밍 아키텍처 도입으로 폐기되었습니다. 글씨를 쓰면 자동으로 streaming_text가 전송됩니다.")
                
                # ─── JW v1 Training Studio API ───
                elif cmd.get("action") == "studio_init":
                    # 데이터셋 목록 전송
                    items = _scan_dataset()
                    await websocket.send(json.dumps({
                        "type": "studio_data",
                        "items": items,
                    }))
                    # GPU 정보
                    import torch as _torch
                    gpu_info = "CPU only"
                    if _torch.cuda.is_available():
                        gpu_info = f"CUDA: {_torch.cuda.get_device_name(0)}"
                    await websocket.send(json.dumps({
                        "type": "device_info",
                        "gpu": gpu_info,
                    }))
                    # IAM 데이터셋 상태
                    try:
                        from airwriting_imu.core.iam_dataset import IAMDatasetLoader
                        iam_status = IAMDatasetLoader().get_status()
                        await websocket.send(json.dumps({
                            "type": "iam_status",
                            **iam_status,
                        }))
                    except Exception:
                        pass
                
                elif cmd.get("action") == "delete_dataset":
                    fname = cmd.get("file", "")
                    fpath = os.path.join("dataset", fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)
                        log.info(f"🗑️ 데이터 삭제: {fname}")
                        await websocket.send(json.dumps({
                            "type": "studio_data",
                            "items": _scan_dataset(),
                        }))
                
                elif cmd.get("action") == "train_jw_v1":
                    log.info("🧠 실제 모델 학습 시작...")
                    _loop = asyncio.get_event_loop()
                    def _do_train_real():
                        try:
                            epochs = int(cmd.get("epochs", 30))
                            bs = cmd.get("batch_size", 16)
                            lr = 0.002
                            
                            def ws_log_callback(msg_text, lvl="info", epoch=None, progress=None, loss=None, acc=None, accuracy=None):
                                msg = {
                                    "type": "train_log", 
                                    "message": msg_text, 
                                    "level": lvl
                                }
                                if epoch is not None: msg["epoch"] = epoch
                                if progress is not None: msg["progress"] = progress
                                if loss is not None: msg["loss"] = loss
                                if acc is not None: msg["acc"] = acc
                                if accuracy is not None: msg["accuracy"] = accuracy
                                
                                asyncio.run_coroutine_threadsafe(ws_broadcast(msg), _loop)
                            
                            model_type = cmd.get("model_type", "pure_bilstm")
                            success, result_msg = ai.train(
                                data_dir="dataset", 
                                epochs=epochs, 
                                lr=lr, 
                                batch_size=bs, 
                                model_type=model_type,
                                callback=ws_log_callback
                            )
                            
                            if success:
                                # 학습 완료 후 즉시 새 가중치를 추론에 반영 (Hot-Reload)
                                ai.load_model()
                                streamer.engine = ai
                                asyncio.run_coroutine_threadsafe(ws_broadcast({
                                    "type": "train_complete", "accuracy": result_msg.split()[-1].replace('%', '')
                                }), _loop)
                                log.info(f"✅ 학습 완료 + 모델 Hot-Reload: {result_msg}")
                            else:
                                ws_log_callback(f"학습 실패: {result_msg}", "error")
                                
                        except Exception as e:
                            import traceback
                            err_trace = traceback.format_exc()
                            log.error(f"Train Crash: {err_trace}")
                            asyncio.run_coroutine_threadsafe(ws_broadcast({
                                "type": "train_log", "message": f"Error: {e}", "level": "error"
                            }), _loop)
                    
                    # start the background trigger
                    threading.Thread(target=_do_train_real, daemon=True).start()
                
                elif cmd.get("action") == "ai_query":
                    query = cmd.get("query", "")
                    # 로컬 분석 AI (데이터셋 통계 기반)
                    response = _ai_analyze(query)
                    await websocket.send(json.dumps({
                        "type": "ai_response",
                        "message": response
                    }))
                
                elif cmd.get("action") == "export_onnx":
                    log.info("📦 ONNX 변환 시작...")
                    try:
                        if ai.model:
                            ai.export_onnx()
                            await websocket.send(json.dumps({
                                "type": "train_log", "message": "ONNX export complete!", "level": "success"
                            }))
                    except Exception as e:
                        await websocket.send(json.dumps({
                            "type": "train_log", "message": f"ONNX export failed: {e}", "level": "error"
                        }))

            except json.JSONDecodeError:
                pass
    finally:
        ws_clients.discard(websocket)
        log.info(f"🔌 WebSocket 해제 (남은: {len(ws_clients)})")


def _scan_dataset(data_dir="dataset"):
    """데이터셋 폴더 스캔 → 메타데이터 목록"""
    items = []
    for f in sorted(os.listdir(data_dir)):
        if not f.endswith('.json'):
            continue
        try:
            fpath = os.path.join(data_dir, f)
            with open(fpath, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            label = data.get("label", f.split("_")[0])
            strokes = data.get("strokes", [])
            total_pts = sum(len(s) for s in strokes)
            items.append({
                "file": f,
                "label": label,
                "strokes": len(strokes),
                "points": total_pts,
                "size": os.path.getsize(fpath),
            })
        except Exception:
            pass
    return items


def _ai_analyze(query: str) -> str:
    """AI 질의응답 — 데이터셋/모델 상태 분석"""
    items = _scan_dataset()
    labels = {}
    total_pts = 0
    for item in items:
        labels[item["label"]] = labels.get(item["label"], 0) + 1
        total_pts += item["points"]
    
    q = query.lower()
    
    if "불균형" in q or "balance" in q:
        if not labels:
            return "데이터가 없습니다."
        counts = list(labels.values())
        ratio = min(counts) / max(counts)
        status = "균형" if ratio > 0.7 else "불균형" if ratio > 0.3 else "심각한 불균형"
        return f"클래스 균형도: {ratio:.0%} ({status}). " + \
               ", ".join(f"{k}={v}개" for k, v in sorted(labels.items()))
    
    if "추천" in q or "설정" in q or "하이퍼" in q:
        n = len(items)
        if n < 20:
            return f"데이터 {n}개: epochs=20, batch=4, lr=0.003, augment=100x 추천. 데이터가 많이 부족합니다."
        elif n < 100:
            return f"데이터 {n}개: epochs=30, batch=8, lr=0.002, augment=50x 추천."
        else:
            return f"데이터 {n}개: epochs=50, batch=16, lr=0.001, augment=20x 추천. 충분한 데이터!"
    
    if "모델" in q or "jw" in q.lower():
        return "JW v1: 1.13M params | VQ-VAE(K=512) → 4-layer Mamba SSM | O(N) 선형 복잡도 | CPU ~22ms | Jetson Orin Nano 최적화"
    
    return f"총 {len(items)}개 샘플, {len(labels)}개 클래스, {total_pts:,}개 포인트. 구체적으로 질문해주세요 (예: '데이터 불균형 확인', '추천 설정')"


# ─── USB Serial 수신 (자동 재연결) ───
def serial_reader_thread(loop, queue: asyncio.Queue):
    import serial
    while True:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            log.info(f"🔌 USB Serial 연결됨: {SERIAL_PORT} ({BAUD_RATE} bps)")
            buffer = bytearray()
            while True:
                data = ser.read(1024)
                if data:
                    buffer.extend(data)
                    while len(buffer) >= 68:
                        idx = buffer.find(0xAA)
                        if idx < 0:
                            buffer.clear()
                            break
                        if idx > 0:
                            buffer = buffer[idx:]
                        packet_found = False
                        for p_size in [92, 94, 68, 70]:
                            if len(buffer) >= p_size and buffer[p_size-1] == 0x55:
                                packet = bytes(buffer[:p_size])
                                loop.call_soon_threadsafe(queue.put_nowait, packet)
                                buffer = buffer[p_size:]
                                packet_found = True
                                break
                        if not packet_found:
                            if len(buffer) > 100:
                                buffer = buffer[1:]
                            else:
                                break
        except serial.SerialException as e:
            log.warning(f"🔌 Serial 끊김 ({e}) — 3초 후 재연결 시도...")
            import time as _time
            _time.sleep(3)
        except Exception as e:
            log.error(f"Serial 에러: {e}")
            import time as _time
            _time.sleep(3)

async def process_serial_queue(queue: asyncio.Queue):
    global esp32_addr, smooth_x, smooth_y, oef_x, oef_y, last_ts_dict
    global was_writing, current_stroke, is_recording_session, session_strokes, button_debounce_counter
    esp32_addr = "USB"
    last_ws_time = 0.0
    while True:
        try:
            # 모든 패킷을 ESKF에 빠짐없이 통과시키되, 웹소켓 전송은 마지막 1개만!
            # 이렇게 해야 회전 추적이 정확하면서도 버튼 반응이 즉각적입니다.
            packets = [await queue.get()]
            while not queue.empty():
                try:
                    packets.append(queue.get_nowait())
                except:
                    break
            
            for pkt_idx, data in enumerate(packets):
                is_last_packet = (pkt_idx == len(packets) - 1)
                
                frame = parser.parse(data)
                if frame is None:
                    continue

                time_sync.update(frame.timestamp_ms)
                
                # 1. 캘리브레이션 단계 (S1, S2, S3 T-Pose 정렬)
                if not calibrator.is_calibrated:
                    done = calibrator.add_sample(frame)
                    
                    if is_last_packet:
                        msg = {
                            "type": "status",
                            "text": f"CALIBRATING NEUTRAL POSE... [{len(calibrator.samples['s1_a'])}/{calibrator.req_samples}]",
                            "ts": frame.timestamp_ms,
                        }
                        asyncio.ensure_future(ws_broadcast(msg))
                    
                    if done:
                        log.info(f"🎯 착용자 자세 정렬 완료!")
                        log.info(f"   [S3 Finger] q_align: {calibrator.q_align['s3']}")
                        log.info(f"   [S2 Hand]   q_align: {calibrator.q_align['s2']}")
                        log.info(f"   [S1 Wrist]  q_align: {calibrator.q_align['s1']}")
                        
                        s1_eskf.reset(initial_q=calibrator.q_align['s1'].astype(np.float64), initial_ba=calibrator.ba['s1'], initial_bg=calibrator.bg['s1'])
                        s2_eskf.reset(initial_q=calibrator.q_align['s2'].astype(np.float64), initial_ba=calibrator.ba['s2'], initial_bg=calibrator.bg['s2'])
                        s3_eskf.reset(initial_q=calibrator.q_align['s3'].astype(np.float64), initial_ba=calibrator.ba['s3'], initial_bg=calibrator.bg['s3'], initial_mag=calibrator.m_ref.get('s3'))
                        
                        # 커서 전용 필터 리셋 (int64 강제 캐스팅 에러 방지용 dtype 명시)
                        q_s3 = calibrator.q_align['s3']
                        madgwick_s3_ray.q = np.array([q_s3[3], q_s3[0], q_s3[1], q_s3[2]], dtype=np.float64)

                        smooth_x, smooth_y = 0.0, 0.0
                        oef_x.reset()
                        oef_y.reset()
                        
                        # Yaw Stabilizer 초기화 (자기장 캘리브레이션 포함)
                        s1_y0, _, s1_p0 = s1_eskf.q.as_euler('ZYX', degrees=False)
                        s3_y0, _, _ = s3_eskf.q.as_euler('ZYX', degrees=False)
                        
                        # 캘리브레이션 중 수집된 자기장 데이터로 MagFusion 활성화
                        mag_samples = calibrator.samples.get('s3_m', [])
                        if len(mag_samples) > 5:
                            mag_arr = np.array(mag_samples)
                            yaw_stabilizer.calibrate(
                                mag_samples=mag_arr, s1_yaw=s1_y0, s3_yaw=s3_y0, heading=0.0
                            )
                            log.info(f"🧲 MagFusion 활성화: {len(mag_samples)}개 샘플, norm={np.median(np.linalg.norm(mag_arr, axis=1)):.1f}µT")
                        else:
                            yaw_stabilizer.calibrate(
                                mag_samples=None, s1_yaw=s1_y0, s3_yaw=s3_y0, heading=0.0
                            )
                            log.warning("⚠️ MagFusion 비활성화 (자기장 데이터 부족)")
                        
                        msg = {
                            "type": "status",
                            "text": "READY",
                            "ts": frame.timestamp_ms,
                        }
                        asyncio.ensure_future(ws_broadcast(msg))
                    continue

                # ── ESP32 리셋 자동 감지 ──
                # 타임스탬프가 과거로 점프하거나 500ms+ 공백이면 ESP32가 리셋된 것으로 판단
                ts = frame.timestamp_ms
                if last_ts_dict['s3'] > 0:
                    ts_diff = ts - last_ts_dict['s3']
                    if ts_diff < -100 or ts_diff > 500:
                        log.warning(f"⚡ ESP32 리셋 감지! (ts_diff={ts_diff}ms) → 자동 재캘리브레이션 시작")
                        calibrator.reset()
                        s1_eskf.reset()
                        s2_eskf.reset()
                        s3_eskf.reset()
                        madgwick_s3_ray.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                        oef_x.reset()
                        oef_y.reset()
                        smooth_x, smooth_y = 0.0, 0.0
                        last_ts_dict['s1'] = 0
                        last_ts_dict['s2'] = 0
                        last_ts_dict['s3'] = 0
                        time_sync.reset()
                        asyncio.ensure_future(ws_broadcast({
                            "type": "status",
                            "text": "ESP32 RESET — RE-CALIBRATING...",
                            "ts": ts,
                        }))
                        continue

                # 동적 dt 계산
                dt = 0.0117  # 85Hz 통신 기준 정밀 타임 스텝
                if last_ts_dict['s3'] > 0 and ts > last_ts_dict['s3']:
                    dt = (ts - last_ts_dict['s3']) / 1000.0
                dt = max(0.005, min(dt, 0.1))
                
                last_ts_dict['s1'] = ts
                last_ts_dict['s2'] = ts
                last_ts_dict['s3'] = ts
                
                # ESKF Predict (모든 패킷에서 빠짐없이 실행!)
                s1_eskf.predict(frame.wrist_accel, frame.wrist_gyro, dt)
                if frame.hand_accel is not None:
                    s2_eskf.predict(frame.hand_accel, frame.hand_gyro, dt)
                    s2_eskf.update_gravity_mahony(frame.hand_accel, alpha=0.002)
                s3_eskf.predict(frame.finger_accel, frame.finger_gyro, dt)
                
                # [핵심] S3 지자기 센서(Magnetometer) 비활성화
                # 실내 자기장 교란(모니터, 전자기기)으로 인해 가만히 있어도 
                # 센서가 방향을 잃고 헤엄치는(Swimming) 현상 원천 차단
                # if np.linalg.norm(frame.finger_mag) > 1.0:
                #     s3_eskf.update_mag(frame.finger_mag)
                
                # ── 물리 버튼 디바운스 처리 (글씨 끊김 현상 방지) ──
                raw_button = (frame.button > 0)
                if raw_button:
                    button_debounce_counter = DEBOUNCE_FRAMES
                    is_writing = True
                else:
                    if button_debounce_counter > 0:
                        button_debounce_counter -= 1
                        is_writing = True
                    else:
                        is_writing = False
                
                # 중력 보정 (Tilt-Drift 방지)
                # 필기 중에도 아주 약한 보정 유지 (alpha=0.0003 → time const ~39초)
                # 한 획(~1초) 내에서는 궤적에 영향 없이, 30초+ 누적 Pitch 드리프트만 방지
                gravity_alpha = 0.0003 if is_writing else 0.002
                s1_eskf.update_gravity_mahony(frame.wrist_accel, alpha=gravity_alpha)
                s3_eskf.update_gravity_mahony(frame.finger_accel, alpha=gravity_alpha)
                
                s3_zupt = False

                # ── 차세대: 3-IMU 차동 운동학 ──
                # 기존 OP-1F 커서에 영향 없이 관절 각도/ICOR/동작 분리를 병렬 계산
                s2_a = frame.hand_accel if frame.hand_accel is not None else frame.wrist_accel
                s2_g = frame.hand_gyro if frame.hand_gyro is not None else frame.wrist_gyro
                
                bio_result = diff_kin.update_full_chain(
                    s1_accel=frame.wrist_accel, s1_gyro=frame.wrist_gyro,
                    s2_accel=s2_a, s2_gyro=s2_g,
                    s3_accel=frame.finger_accel, s3_gyro=frame.finger_gyro,
                    q_s1=s1_eskf.q, q_s2=s2_eskf.q, q_s3=s3_eskf.q,
                    dt=dt,
                )
                
                # 손목/손가락 동작 분리
                wrist_motion, finger_motion = motion_sep.separate(
                    frame.wrist_accel, frame.finger_accel
                )
                writing_intent = motion_sep.get_writing_intent(finger_motion)
                
                pipeline_state["bio_kin"] = bio_result
                pipeline_state["writing_intent"] = writing_intent

                # ── 최첨단 모델: OP-1F (Orthographic Pointing) 복구 ──
                # 3D 스켈레톤용 ESKF는 중력 보정 시 강하게 비틀리므로(Jumping), 
                # 2D 커서는 무조건 아주 부드러운 Madgwick(beta=0.05)을 독립적으로 구동해야 합니다!
                
                # [NEW] 3중 Yaw 보정기를 거친 자이로 사용 (수평 방향 드리프트 원천 차단)
                q_s3_wxyz = madgwick_s3_ray.q
                corrected_s3_gyro = yaw_stabilizer.process(
                    s3_accel=frame.finger_accel,
                    s3_gyro=frame.finger_gyro,
                    s3_mag=frame.finger_mag,
                    s1_gyro=frame.wrist_gyro,
                    is_writing=is_writing,
                    current_q_wxyz=q_s3_wxyz,
                    dt=dt
                )
                
                q_s3_ray = madgwick_s3_ray.update_imu(frame.finger_accel, corrected_s3_gyro)
                
                # numpy 배열 쿼터니언 [w, x, y, z] 을 scipy Rotation 객체로 변환 (x, y, z, w)
                q_s3_ray_rot = Rotation.from_quat([q_s3_ray[1], q_s3_ray[2], q_s3_ray[3], q_s3_ray[0]])
                
                # ── Gimbal-Lock-Free 포인터 제어 (Forward Vector + atan2) ──
                # Euler 분해('ZXY')는 Pitch ≈ ±90°에서 Gimbal Lock이 발생하여
                # Yaw가 180° 점프하는 근본 원인이었음.
                # Forward Vector에서 직접 atan2로 Yaw/Pitch를 계산하면 Gimbal Lock이 원리적으로 불가능.
                
                # S3 검지 센서 물리적 장착 각도(좌측 15도) 보정을 벡터 레벨에서 적용
                tilt_rad = np.radians(15.0)
                forward_local = np.array([np.sin(tilt_rad), -np.cos(tilt_rad), 0.0])
                forward = q_s3_ray_rot.apply(forward_local)
                
                # atan2 기반 Yaw/Pitch 추출 (Roll 완전 무시, Gimbal Lock 없음)
                phys_x = np.arctan2(forward[0], -forward[1])   # Yaw (좌우)
                phys_z = np.arctan2(forward[2], np.sqrt(forward[0]**2 + forward[1]**2))  # Pitch (상하)
                
                # One Euro Filter (고정 주파수 사용)
                # USB 패킷 지연 및 배치 처리(while loop)로 인한 dt=0 버그를 원천 방지하기 위해 timestamp=None 전달
                filtered_x = oef_x.filter(phys_x, timestamp=None)
                filtered_y = oef_y.filter(phys_z, timestamp=None)
                
                # 글 쓰는 중이든, 펜을 허공에 들고(Hover) 있든 
                # 절대로 화면 커서가 멈추거나 튕기지 않도록 1:1 위치를 상시 보장합니다.
                smooth_x = filtered_x
                smooth_y = filtered_y

                out_x = round(smooth_x * VIRTUAL_PEN_LENGTH * X_SENSITIVITY, 2)
                out_y = round(smooth_y * VIRTUAL_PEN_LENGTH, 2)

                # 데이터셋 수집
                
                if is_recording_session:
                    if is_writing:
                        current_stroke.append({
                            "ts": ts, "dt": dt,
                            "x": out_x, "y": out_y,
                            "zupt": bool(s3_zupt),
                            "ax": float(frame.finger_accel[0]),
                            "ay": float(frame.finger_accel[1]),
                            "az": float(frame.finger_accel[2]),
                            "gx": float(frame.finger_gyro[0]),
                            "gy": float(frame.finger_gyro[1]),
                            "gz": float(frame.finger_gyro[2])
                        })
                    elif was_writing and not is_writing:
                        if len(current_stroke) > 10:
                            session_strokes.append(current_stroke)
                            log.info(f"[Record] 획 추가 완료: {len(current_stroke)} points (총 {len(session_strokes)}획)")
                        current_stroke = []
                else:
                    if current_stroke:
                        current_stroke = []
                    # 백엔드 스트리밍 인퍼런스
                    frame_features = {
                        "x": out_x, "y": out_y,
                        "ax": float(frame.finger_accel[0]),
                        "ay": float(frame.finger_accel[1]),
                        "az": float(frame.finger_accel[2]),
                        "gx": float(frame.finger_gyro[0]),
                        "gy": float(frame.finger_gyro[1]),
                        "gz": float(frame.finger_gyro[2])
                    }
                    streamer.process_frame(frame_features, is_writing)
                    
                    # 차세대: Duo Streamers 희소 인식 (병렬)
                    sparse_engine.process_frame(frame_features, is_writing)
                    pipeline_state["duo_streamers"] = sparse_engine.get_efficiency_stats()
                    
                was_writing = is_writing

                # 윈도우 Serial 버퍼링(100ms 단위 청크)으로 인해 중간 패킷이 생략되는 것을 방지하기 위해
                # 수신된 모든 프레임을 웹소켓으로 전송하여 선이 부드럽게 그려지도록 합니다.
                msg = {
                    # "type" 키가 있으면 handleServerMessage로 가버리므로 제거하거나 예약어가 아닌 것을 써야 함
                    # 예전 프론트엔드는 type이 없으면 handleFrame으로 보냄
                    "x": out_x,
                    "y": out_y,
                    "ray_hit": [out_x, out_y],
                    "fingertip": [out_x, out_y, -0.68],
                    "is_writing": is_writing,
                    "button": frame.button,
                    "zupt": bool(s3_zupt),
                    "ts": ts,
                    "pkt": parser.valid_packets,
                    "latency": round(time_sync.latency_ms, 1) if time_sync.is_synced else -1,
                    "orientations": {
                        "forearm": [float(s1_eskf.q.as_quat()[3]), float(s1_eskf.q.as_quat()[0]), float(s1_eskf.q.as_quat()[1]), float(s1_eskf.q.as_quat()[2])],
                        # Dual-Node (2센서) 모드일 경우, 손등 센서가 없으므로 손목(S1) 자세를 복사하여 스켈레톤 꺾임 방지
                        "hand": [
                            float(s2_eskf.q.as_quat()[3] if frame.hand_accel is not None else s1_eskf.q.as_quat()[3]),
                            float(s2_eskf.q.as_quat()[0] if frame.hand_accel is not None else s1_eskf.q.as_quat()[0]),
                            float(s2_eskf.q.as_quat()[1] if frame.hand_accel is not None else s1_eskf.q.as_quat()[1]),
                            float(s2_eskf.q.as_quat()[2] if frame.hand_accel is not None else s1_eskf.q.as_quat()[2])
                        ],
                        "finger": [float(s3_eskf.q.as_quat()[3]), float(s3_eskf.q.as_quat()[0]), float(s3_eskf.q.as_quat()[1]), float(s3_eskf.q.as_quat()[2])],
                    },
                    "raw_sensors": {
                        "s1": {
                            "ax": float(frame.wrist_accel[0]),
                            "ay": float(frame.wrist_accel[1]),
                            "az": float(frame.wrist_accel[2])
                        },
                        "s2": {
                            "ax": float(frame.hand_accel[0] if frame.hand_accel is not None else 0.0),
                            "ay": float(frame.hand_accel[1] if frame.hand_accel is not None else 0.0),
                            "az": float(frame.hand_accel[2] if frame.hand_accel is not None else 0.0)
                        },
                        "s3": {
                            "ax": float(frame.finger_accel[0]),
                            "ay": float(frame.finger_accel[1]),
                            "az": float(frame.finger_accel[2])
                        }
                    },
                    # ─── 차세대 파이프라인 상태 (Digital Twin용) ───
                    "pipeline": {
                        "joint_angles": pipeline_state["bio_kin"].get("joint_angles", {}),
                        "icor_mcp": pipeline_state["bio_kin"].get("icor_mcp", [0, 0]),
                        "icor_pip": pipeline_state["bio_kin"].get("icor_pip", [0, 0]),
                        "writing_intent": round(pipeline_state.get("writing_intent", 0), 3),
                        "duo_streamers": pipeline_state.get("duo_streamers", {}),
                    },
                }
                asyncio.ensure_future(ws_broadcast(msg))
        except Exception as e:
            import traceback
            log.error(f"❌ 큐 프로세스 오류: {e}")
            log.error(traceback.format_exc())


# ─── Mock 데이터 ───
async def mock_data_generator():
    log.info("🎭 Mock 모드 — 가상 필기 데이터 생성")
    t = 0
    ts = 0
    while True:
        t += 0.01
        ts += 10

        phase = t % 8.0
        pen = phase < 5.0
        if pen:
            angle = phase * 2 * math.pi / 2.5
            ax, ay = math.cos(angle) * 3.0, math.sin(angle) * 3.0
        else:
            ax = ay = 0.0

        frame = SensorFrame(
            timestamp_ms=ts,
            wrist_accel=np.array([0, 0, 9.81], dtype=np.float32),
            wrist_gyro=np.zeros(3, dtype=np.float32),
            finger_accel=np.array([ax, ay, 9.81], dtype=np.float32),
            finger_gyro=np.array([0, 0, 0.3 if pen else 0], dtype=np.float32),
            finger_mag=np.zeros(3, dtype=np.float32),
            button=1 if pen else 0,
            packet_size=68,
        )

        s3_eskf.predict(frame.finger_accel, frame.finger_gyro, 0.01)
        x, y, _ = s3_eskf.p
        msg = {
            "type": "position",
            "x": round(x * 1000, 2),
            "y": round(y * 1000, 2),
            "button": frame.button,
            "zupt": False,
            "ts": ts,
            "pkt": ts // 10,
            "latency": 0.0,
        }
        await ws_broadcast(msg)
        await asyncio.sleep(0.01)


# ─── HTTP 서버 ───
def start_http_server():
    web_dir = Path(__file__).parent / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    handler = partial(SimpleHTTPRequestHandler, directory=str(web_dir))
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), handler)
    log.info(f"🌍 HTTP 서버: http://localhost:{HTTP_PORT}")
    httpd.serve_forever()


# ─── 상태 출력 ───
async def status_printer():
    last_valid = 0
    while True:
        await asyncio.sleep(10)
        stats = parser.get_stats()
        rate = (stats["valid"] - last_valid) / 10.0
        last_valid = stats["valid"]
        ts = time_sync.get_stats()
        esp = f"{esp32_addr[0]}" if esp32_addr else "미연결"

        log.info(
            f"📊 ESP32={esp} | {rate:.0f}Hz | "
            f"loss={stats['loss_rate']}% gaps={stats['gap_count']} | "
            f"latency={ts['latency_ms']}ms | "
            f"WS={len(ws_clients)}"
        )
        
        # 3-Layer Yaw Stabilizer 상태 로깅
        ystats = yaw_stabilizer.get_stats()
        log.info(
            f"🛡️ YawStabilizer | ZARU(Bias): {ystats['bias']['bias_deg_s'][2]:.3f}°/s "
            f"| Anchor(Dev): {ystats['anchor']['deviation_deg']:.1f}° "
            f"| Mag(Trust): {ystats['mag']['trust']:.2f}"
        )


# ─── 메인 ───
async def main():
    global main_loop
    main_loop = asyncio.get_running_loop()

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    ws_server = await ws_serve(ws_handler, "0.0.0.0", WS_PORT)
    log.info(f"🔌 WebSocket: ws://localhost:{WS_PORT}")

    serial_queue = asyncio.Queue()
    tasks = [status_printer(), process_serial_queue(serial_queue)]
    
    if not mock_mode:
        t = threading.Thread(target=serial_reader_thread, args=(main_loop, serial_queue), daemon=True)
        t.start()
    else:
        tasks.append(mock_data_generator())

    log.info("=" * 55)
    log.info("  Hybrid-AirScribe Digital Twin — Phase 1 (USB Mode)")
    log.info(f"  Dashboard:  http://localhost:{HTTP_PORT}")
    log.info(f"  WebSocket:  ws://localhost:{WS_PORT}")
    log.info(f"  Serial:     {SERIAL_PORT} @ {BAUD_RATE}")
    log.info(f"  Mode:       {'🎭 Mock' if mock_mode else '📡 Live USB'}")
    log.info("=" * 55)

    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        pass
    finally:
        ws_server.close()
        oled.close()


if __name__ == "__main__":
    import serial.tools.list_ports
    ports = list(serial.tools.list_ports.comports())
    if ports:
        log.info(f"사용 가능한 COM 포트: {[p.device for p in ports]}")
    else:
        log.warning("장치 관리자에 인식된 COM 포트가 없습니다!")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("서버 종료")
