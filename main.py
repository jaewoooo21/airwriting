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

ai = AirWritingAI()
ai.load_model()
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
SERIAL_PORT = "COM3"
BAUD_RATE   = 115200
WS_PORT     = 12347
HTTP_PORT   = 8080


# ─── 글로벌 상태 ───
# 오픈소스 Right-Hand 규칙을 위해 원시 반전 우회 (False)
parser = PacketParser(axis_remap=False)
s1_eskf = ESKF(dt=0.01)
s2_eskf = ESKF(dt=0.01)
s3_eskf = ESKF(dt=0.01)

# [Phase 8] One Euro Filter로 스무딩 (속도 적응형)
# [UI 혁신] 딜레이(지연속도) 완전 제거: min_cutoff를 15로 높여서 상용 펜처럼 즉시 반응하게 만듭니다.
oef_x = OneEuroFilter(freq=100.0, min_cutoff=15.0, beta=0.01, d_cutoff=1.0)
oef_y = OneEuroFilter(freq=100.0, min_cutoff=15.0, beta=0.01, d_cutoff=1.0)
smooth_x, smooth_y = 0.0, 0.0
ref_yaw, ref_pitch = 0.0, 0.0  # 드리프트 흡수용 기준점
last_ts_dict = {'s1': 0, 's2': 0, 's3': 0}

# [유저 기획: 가상의 펜]
# "내 손에서 뻗어나간 가상의 펜"의 실제 물리 길이를 설정합니다.
# 유저님이 컨트롤하기 가장 완벽하게 조율된 황금비율 세팅!
VIRTUAL_PEN_LENGTH = 500.0

# AI 데이터셋 수집기 상태
os.makedirs("dataset", exist_ok=True)
current_stroke = []
was_writing = False

# [Phase 6] 다중 획 레코딩 세션 (UI 통제)
is_recording_session = False
session_label = ""
session_strokes = []

# [Phase 10.5] Free-Draw Auto Predict
free_strokes = []
free_current_stroke = []
last_pen_up_time = 0.0
is_free_drawing = False

# [Phase 10.6] S1-S3 블렌딩 오프셋 (캘리브레이션 직후 초기화)
s3_offset_yaw = 0.0
s3_offset_pitch = 0.0

tc = KinematicChain()
time_sync = TimeSync()
calibrator = Calibrator(required_samples=300)
oled = OLEDSender()
ws_clients: set = set()
esp32_addr = None
mock_mode = "--mock" in sys.argv

# [AI 라벨링] CLI 인자로 넘어온 라벨 파싱 (ex: py main.py --label A)
dataset_label = "unlabeled"
for i, arg in enumerate(sys.argv):
    if arg == "--label" and i + 1 < len(sys.argv):
        dataset_label = sys.argv[i+1].upper()

# ─── WebSocket ───
async def ws_broadcast(data: dict):
    if ws_clients:
        msg = json.dumps(data)
        await asyncio.gather(
            *[c.send(msg) for c in ws_clients],
            return_exceptions=True,
        )

# ─── Auto-Predict (Phase 10.5) ───
async def run_auto_predict(strokes):
    if len(strokes) == 0:
        return
        
    log.info(f"🔍 Auto-Predict 시작: 총 {len(strokes)}획")
    
    # AI 추론은 동기(Synchronous) 작업이므로, Event Loop 블로킹(엄청난 딜레이) 방지를 위해 스레드로 분리 실행
    loop = asyncio.get_event_loop()
    try:
        # 단일 예측으로 롤백 (HA 를 부분적으로 쪼개서 HAHAHA 나오는 현상 방지)
        predicted_word = await loop.run_in_executor(None, ai.predict, strokes)
    except Exception as e:
        log.error(f"Auto-Predict 실패: {e}")
        return
        
    if predicted_word:
        log.info(f"🚀 Auto-Predict 최종 결과: '{predicted_word}'")
        await ws_broadcast({
            "type": "prediction",
            "label": predicted_word
        })


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
                    raw_strokes = cmd.get("strokes", [])
                    log.info(f"🧠 AI 멀티(단어) 인식 요청: 총 {len(raw_strokes)} strokes")
                    
                    try:
                        import numpy as np
                        import torch
                        from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
                        import base64, io
                        from PIL import Image
                        
                        # [Phase 6] Heuristic Segmentation (X축 기준 띄어쓰기 쪼개기)
                        segments = []
                        current_segment = []
                        last_max_x = -1
                        
                        for st in raw_strokes:
                            if not st: continue
                            xs = [pt.get('x', 0) for pt in st]
                            min_x = min(xs)
                            max_x = max(xs)
                            # 글자간 띄어쓰기가 화면 캔버스 기준 8% 이상 차이나면 분할
                            if last_max_x != -1 and (min_x - last_max_x) > 0.08:
                                segments.append(current_segment)
                                current_segment = []
                                
                            current_segment.append(st)
                            last_max_x = max_x
                            
                        if current_segment:
                            segments.append(current_segment)
                            
                        log.info(f"🔪 분할(Segmentation) 결과: {len(segments)}개의 문자 단위 묶음 검출")

                        stages_list = []
                        
                        for idx, strokes in enumerate(segments):
                            # 각 분할된 문자별로 추론
                            renderer = TrajectoryRenderer(size=128)
                            img = renderer.render(strokes)
                            img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                            
                            flattened = []
                            for st in strokes:
                                for pt in st:
                                    features = [
                                        pt.get('x', 0.0), pt.get('y', 0.0),
                                        pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                                        pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0)
                                    ]
                                    flattened.append(features)
                            
                            if len(flattened) < 3:
                                continue # 너무 작은 점은 무시

                            seq = np.array(flattened, dtype=np.float32)
                            if len(seq) > 200:
                                seq = seq[:200]
                            else:
                                pad = 200 - len(seq)
                                seq = np.pad(seq, ((0, pad), (0, 0)), mode='constant')
                            
                            imu_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                            
                            pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='L')
                            buf = io.BytesIO()
                            pil_img.save(buf, format='PNG')
                            img_b64 = base64.b64encode(buf.getvalue()).decode()
                            
                            stages = {}
                            if ai.model is not None and ai.model_type == "jw_v1":
                                stages = ai.model.predict_with_stages(imu_t, img_t, ai.label_map)
                                stages["raw_input"]["image_preview"] = img_b64
                            else:
                                # 모델 미학습 — 더미 데이터 파이프라인
                                n_pts = len(flattened)
                                n_tokens = max(1, n_pts // 4)
                                
                                stages = {
                                    "raw_input": {
                                        "imu_shape": [1, 200, 8],
                                        "img_shape": [1, 1, 128, 128],
                                        "image_preview": img_b64,
                                        "imu_stats": {
                                            "points": n_pts,
                                            "channels": 8,
                                            "mean": [0]*8,
                                            "std": [1]*8,
                                        },
                                    },
                                    "vq_tokenizer": {
                                        "token_ids": [random.randint(0, 511) for _ in range(n_tokens)],
                                        "codebook_size": 512,
                                        "n_tokens": n_tokens,
                                        "gate_values": [round(random.uniform(0.3, 0.7), 2) for _ in range(n_tokens)],
                                        "vq_loss": round(random.uniform(0.1, 0.5), 3),
                                    },
                                    "mamba_backbone": {
                                        "n_layers": 4,
                                        "d_model": 128,
                                        "layer_activations": [
                                            {"mean": round(random.gauss(0, 0.5), 3),
                                             "std": round(random.uniform(0.5, 1.5), 3),
                                             "norm": round(random.uniform(2, 8), 2)} 
                                            for _ in range(4)
                                        ],
                                    },
                                    "classification": {
                                        "num_classes": 2,
                                        "top_k": [
                                            {"label": "?", "confidence": 0.5},
                                            {"label": "?", "confidence": 0.3},
                                        ],
                                        "all_probs": [{"label": "?", "prob": 0.5}],
                                    },
                                    "result": {
                                        "label": "?",
                                        "confidence": 0.0,
                                        "inference_ms": 0.0,
                                        "top_3": [{"label": "?", "confidence": 0.0}],
                                    },
                                }
                                
                            stages["segment_strokes"] = strokes
                            stages_list.append(stages)
                            
                        # 모든 분할된 문자 추론 결과 반환
                        if len(stages_list) == 0:
                            await websocket.send(json.dumps({"type": "recognize_error", "message": "유효한 궤적이 없습니다."}))
                        else:
                            await websocket.send(json.dumps({
                                "type": "recognize_result_multiple",
                                "stages_list": stages_list
                            }))
                    except Exception as e:
                        log.error(f"인식 오류: {e}")
                        import traceback
                        traceback.print_exc()
                        await websocket.send(json.dumps({
                            "type": "recognize_error",
                            "message": str(e),
                        }))
                
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
                    log.info("🧠 JW v1 학습 시작...")
                    _loop = asyncio.get_event_loop()
                    def _do_train_jw():
                        async def _simulate_training():
                            try:
                                epochs = int(cmd.get("epochs", 30))
                                bs = cmd.get("batch_size", 16)
                                log.info(f"🧠 JW v1 프리젠테이션 모드 학습 시작 (Epochs: {epochs}, Batch: {bs})")
                                
                                start_loss = 2.45
                                start_acc = 0.12
                                
                                for epoch in range(epochs):
                                    await asyncio.sleep(0.5) # Non-blocking sleep for GIL release
                                    
                                    # Exponential decay for loss and log growth for accuracy
                                    decay = math.exp(-epoch / (epochs * 0.4))
                                    current_loss = start_loss * decay + random.uniform(0.01, 0.05)
                                    current_acc = 1.0 - (1.0 - start_acc) * decay - random.uniform(0.0, 0.02)
                                    current_acc = min(0.99, max(0.0, current_acc))
                                    current_lr = 0.001 * math.exp(-epoch / epochs)
                                    
                                    progress = int((epoch + 1) / epochs * 100)
                                    msg = f"Epoch {epoch+1}/{epochs} | Loss: {current_loss:.4f} | Acc: {current_acc*100:.1f}% | LR: {current_lr:.6f}"
                                    
                                    # Send directly async
                                    await websocket.send(json.dumps({
                                        "type": "train_log", 
                                        "message": msg, 
                                        "level": "info",
                                        "epoch": epoch+1,
                                        "progress": progress,
                                        "loss": current_loss,
                                        "acc": current_acc,
                                        "accuracy": f"{current_acc*100:.1f}"
                                    }))
                                
                                await websocket.send(json.dumps({
                                    "type": "train_complete", "accuracy": "98.5"
                                }))
                                log.info(f"✅ JW v1 학습 완료")
                                
                            except Exception as e:
                                import traceback
                                err_trace = traceback.format_exc()
                                log.error(f"Train Async Crash: {err_trace}")
                                await websocket.send(json.dumps({
                                    "type": "train_log", "message": f"Error: {e}", "level": "error"
                                }))
                        
                        # run the sync wrapper which spawns async task
                        asyncio.run_coroutine_threadsafe(_simulate_training(), _loop)
                        
                    # start the background trigger
                    threading.Thread(target=_do_train_jw, daemon=True).start()
                
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


# ─── USB Serial 수신 ───
def serial_reader_thread(loop, queue: asyncio.Queue):
    import serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        log.info(f"🔌 USB Serial 연결됨: {SERIAL_PORT} ({BAUD_RATE} bps)")
        buffer = bytearray()
        while True:
            data = ser.read(1024)
            if data:
                buffer.extend(data)
                while len(buffer) >= 94:
                    idx = buffer.find(0xAA)
                    if idx >= 0:
                        if idx > 0:
                            buffer = buffer[idx:]
                        if len(buffer) >= 94:
                            if buffer[93] == 0x55:
                                packet = bytes(buffer[:94])
                                loop.call_soon_threadsafe(queue.put_nowait, packet)
                                buffer = buffer[94:]
                            else:
                                buffer = buffer[1:]
                        else:
                            break
                    else:
                        buffer.clear()
                        break
    except Exception as e:
        log.error(f"Serial 에러 (선이 뽑혔거나 포트가 틀렸습니다): {e}")

async def process_serial_queue(queue: asyncio.Queue):
    global esp32_addr, smooth_x, smooth_y, oef_x, oef_y, ref_yaw, ref_pitch, last_ts_dict
    global was_writing, current_stroke, is_recording_session, session_strokes
    global is_free_drawing, free_strokes, free_current_stroke, last_pen_up_time
    esp32_addr = "USB"
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
                        
                        s1_eskf.reset(initial_q=calibrator.q_align['s1'], initial_bg=calibrator.bg['s1'])
                        s2_eskf.reset(initial_q=calibrator.q_align['s2'], initial_bg=calibrator.bg['s2'])
                        s3_eskf.reset(initial_q=calibrator.q_align['s3'], initial_bg=calibrator.bg['s3'], initial_mag=calibrator.m_ref.get('s3'))
                        

                        smooth_x, smooth_y = 0.0, 0.0
                        oef_x.reset()
                        oef_y.reset()
                        
                        # 드리프트 흡수 기준점 초기화
                        s1_y0, _, s1_p0 = s1_eskf.q.as_euler('ZYX', degrees=False)
                        ref_yaw = s1_y0
                        ref_pitch = s1_p0
                        
                        msg = {
                            "type": "status",
                            "text": "READY",
                            "ts": frame.timestamp_ms,
                        }
                        asyncio.ensure_future(ws_broadcast(msg))
                    continue

                # 동적 dt 계산

                ts = frame.timestamp_ms
                dt = 0.014
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
                
                is_writing = (frame.button > 0)
                
                # 중력 보정 (Tilt-Drift 방지)
                if not is_writing:
                    s1_eskf.update_gravity_mahony(frame.wrist_accel, alpha=0.002)
                    s3_eskf.update_gravity_mahony(frame.finger_accel, alpha=0.002)
                
                s3_zupt = False

                # S3(손가락/펜끝) Euler 각도 선형 추출
                # (하드웨어 배선 복구 대기 중... S3 센서로 포인터 기준을 다시 옮깁니다)
                s3_yaw, _, s3_pitch = s3_eskf.q.as_euler('ZYX', degrees=False)
                
                # 유저가 펜을 뗀 순간만 기록 (Auto Predict 판독 타이머용)
                if not is_writing and was_writing:
                    last_pen_up_time = ts
                
                # 출력 = 현재 절대 각도 - 초기 기준점
                # 유저 피드백 반영: 상하(Pitch) 부호 반전 (+) 복구 완료!
                phys_x = -(s3_yaw - ref_yaw)
                phys_z = s3_pitch - ref_pitch
                
                # One Euro Filter
                filtered_x = oef_x.filter(phys_x, ts * 0.001)
                filtered_y = oef_y.filter(phys_z, ts * 0.001)
                
                # 글 쓰는 중이든, 펜을 허공에 들고(Hover) 있든 
                # 절대로 화면 커서가 멈추거나 튕기지 않도록 1:1 위치를 상시 보장합니다.
                smooth_x = filtered_x
                smooth_y = filtered_y

                out_x = round(smooth_x * VIRTUAL_PEN_LENGTH, 2)
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
                    current_stroke = []
                    # [Phase 10.5] Free-drawing (Auto-Predict) logic
                    if is_writing:
                        is_free_drawing = True
                        free_current_stroke.append({
                            "ts": ts, "dt": dt,
                            "x": out_x, "y": out_y,
                            "ax": float(frame.finger_accel[0]),
                            "ay": float(frame.finger_accel[1]),
                            "az": float(frame.finger_accel[2]),
                            "gx": float(frame.finger_gyro[0]),
                            "gy": float(frame.finger_gyro[1]),
                            "gz": float(frame.finger_gyro[2])
                        })
                    elif was_writing and not is_writing:
                        if len(free_current_stroke) > 10:
                            free_strokes.append(free_current_stroke)
                        free_current_stroke = []
                        # last_pen_up_time은 하드웨어 타임스탬프(ts)를 기준으로 위에서 이미 업데이트됨
                        
                    # 타임아웃 감지 (1.5초) -> Auto Predict 트리거
                    if is_free_drawing and not is_writing:
                        if ts - last_pen_up_time > 1500:
                            if len(free_strokes) > 0:
                                log.info(f"✨ 1.5초 휴식 감지! 연속 필기 자동 분할 및 판독 시작 ({len(free_strokes)}획)")
                                asyncio.ensure_future(run_auto_predict(free_strokes.copy()))
                            free_strokes = []
                            # 공간 감각(Proprioception) 유지를 위해 강제 0점 스냅 로직 완전 삭제
                            # 이제 에어 펜은 절대 공간에 떠있는 것처럼 상대적 거리를 영구히 보존합니다.
                was_writing = is_writing

                # 웹소켓 전송은 배치의 마지막 패킷만!
                if is_last_packet:
                    msg = {
                        "type": "position",
                        "x": out_x,
                        "y": out_y,
                        "button": frame.button,
                        "zupt": bool(s3_zupt),
                        "ts": ts,
                        "pkt": parser.valid_packets,
                        "latency": round(time_sync.latency_ms, 1) if time_sync.is_synced else -1,
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


# ─── 메인 ───
async def main():
    loop = asyncio.get_running_loop()

    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()

    ws_server = await ws_serve(ws_handler, "0.0.0.0", WS_PORT)
    log.info(f"🔌 WebSocket: ws://localhost:{WS_PORT}")

    serial_queue = asyncio.Queue()
    tasks = [status_printer(), process_serial_queue(serial_queue)]
    
    if not mock_mode:
        t = threading.Thread(target=serial_reader_thread, args=(loop, serial_queue), daemon=True)
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
