"""
AirScribe AI Engine — CNN-BiLSTM 하이브리드 + Transformer (Phase 9)

아키텍처 근거:
  학계 SOTA 벤치마크에서 IMU 기반 손글씨 인식은 CNN(로컬 패턴 추출) +
  BiLSTM(양방향 시퀀스 컨텍스트)이 순수 Transformer 대비 일관된 성능 우위.
  특히 노이즈가 심한 IMU 데이터에서 CNN의 1D-Conv 특징 추출이 압도적.

지원 타겟: 알파벳 A-Z + 영어 문장 (Phase 9 목표)
배포 타겟: Jetson Orin Nano (ONNX → TensorRT 변환 지원)
"""

import os
import json
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


class GestureDataset(Dataset):
    def __init__(self, data_dir="dataset", max_seq_len=200):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.scaler = StandardScaler()
        
        self._load_data()
        
    def _load_data(self):
        files = glob.glob(os.path.join(self.data_dir, "*.json"))
        raw_data = []
        
        label_idx = 0
        for f in files:
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    
                    strokes_list = []
                    label = ""
                    
                    if isinstance(data, dict) and "strokes" in data:
                        strokes_list = data["strokes"]
                        label = data.get("label", "Unknown")
                    else:
                        strokes_list = [data]
                        label = os.path.basename(f).split('_')[0]
                        
                    if label not in self.label_map:
                        self.label_map[label] = label_idx
                        label_idx += 1
                        
                    # Flatten strokes (획 간 구분 마커 포함)
                    flattened = []
                    for si, st in enumerate(strokes_list):
                        for pt in st:
                            features = [
                                pt.get('x', 0.0), pt.get('y', 0.0),
                                pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                                pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0)
                            ]
                            flattened.append(features)
                            
                    if len(flattened) > 5:
                        raw_data.append(np.array(flattened))
                        self.labels.append(self.label_map[label])
            except Exception as e:
                print(f"Failed to load {f}: {e}")
                
        if len(raw_data) == 0:
            return
            
        all_pts = np.vstack(raw_data)
        self.scaler.fit(all_pts)
        
        for seq in raw_data:
            seq_norm = self.scaler.transform(seq)
            if len(seq_norm) > self.max_seq_len:
                seq_norm = seq_norm[:self.max_seq_len]
            else:
                pad_len = self.max_seq_len - len(seq_norm)
                seq_norm = np.pad(seq_norm, ((0, pad_len), (0, 0)), mode='constant')
            self.samples.append(seq_norm)
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ─── Phase 7: 기존 Transformer (하위 호환) ───
class GestureTransformer(nn.Module):
    def __init__(self, input_dim=8, d_model=64, nhead=4, num_layers=3, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        features = x.mean(dim=1)
        out = self.classifier(features)
        return out


# ─── Phase 9: CNN-BiLSTM 하이브리드 (SOTA) ───
class GestureCNNBiLSTM(nn.Module):
    """
    CNN(로컬 모션 패턴 추출) + BiLSTM(양방향 시퀀스 이해) + Attention Pooling
    
    IMU 손글씨 인식 SOTA 아키텍처 (arXiv 2024 벤치마크 기준)
    Jetson Orin Nano FP16 추론에 최적화된 경량 설계
    """
    def __init__(self, input_dim=8, num_classes=2, hidden_dim=64):
        super().__init__()
        
        # 1D-CNN: 로컬 시간 패턴 추출 (커널 3→5→7 다중 스케일)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # BiLSTM: 양방향 시퀀스 컨텍스트
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Attention Pooling: 중요한 타임스텝에 가중치 집중
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        
        # CNN expects [batch, channels, seq_len]
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        
        # Back to [batch, seq_len, features]
        x_lstm_in = x_cnn.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x_lstm_in)
        
        # Attention Pooling
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden*2]
        
        # Classification
        out = self.classifier(context)
        return out


class AirWritingAI:
    """
    통합 AI 관리자.
    model_type:
      - "jw_v1"      : JW v1 (VQ-VAE + Mamba SSM) — 최신
      - "cnn_bilstm"  : CNN-BiLSTM 하이브리드 — 하위 호환
      - "transformer" : 기존 Transformer — 하위 호환
    """
    def __init__(self):
        self.model = None
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_map = {}
        self.scaler = None
        self.model_type = "jw_v1"
        self.renderer = None  # TrajectoryRenderer (JW v1용)
        
    def train(self, data_dir="dataset", epochs=30, model_type=None,
              augment_factor=0, image_size=128, lr=0.001,
              batch_size=16, callback=None):
        """
        통합 학습 메서드.
        
        callback: (msg, level, **kw) → 실시간 로그 전송용
        """
        mt = model_type or self.model_type
        
        if mt == "jw_v1":
            return self._train_jw_v1(data_dir, epochs, augment_factor,
                                     image_size, lr, batch_size, callback)
        else:
            return self._train_legacy(data_dir, epochs)
    
    def _train_jw_v1(self, data_dir, epochs, augment_factor,
                     image_size, lr, batch_size, callback):
        """JW v1 학습 파이프라인"""
        log = callback or (lambda msg, lvl="info", **kw: print(f"[{lvl}] {msg}"))
        
        log("JW v1 학습 파이프라인 시작...", "info")
        
        # 1. 궤적 이미지 렌더링
        from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
        self.renderer = TrajectoryRenderer(size=image_size)
        rendered = self.renderer.render_dataset(data_dir)
        log(f"에어라이팅 궤적 이미지 {len(rendered)}개 렌더링 완료 ({image_size}×{image_size})", "success")
        
        # 1.5. IAM 데이터 하이브리드 (옵션)
        try:
            from airwriting_imu.core.iam_dataset import IAMDatasetLoader
            iam_loader = IAMDatasetLoader(image_size=image_size)
            if iam_loader.is_available():
                iam_data = iam_loader.load_images(mode="word", max_samples=2000)
                rendered.extend(iam_data)
                log(f"IAM 하이브리드: +{len(iam_data)}개 추가 (총 {len(rendered)}개)", "success")
            else:
                log("IAM 데이터 없음 — 에어라이팅 데이터만 사용", "dim")
        except Exception as e:
            log(f"IAM 로드 스킵: {e}", "dim")
        
        if len(rendered) < 2:
            return False, "데이터가 2개 미만입니다."
        
        # 2. 라벨 매핑
        label_set = sorted(set(r["label"] for r in rendered))
        self.label_map = {lbl: idx for idx, lbl in enumerate(label_set)}
        num_classes = len(self.label_map)
        log(f"클래스 {num_classes}개: {', '.join(label_set[:20])}{'...' if num_classes > 20 else ''}", "info")
        
        if num_classes < 2:
            return False, "최소 2종류 이상의 글자를 수집하세요."
        
        # 3. 데이터 증강
        if augment_factor > 1:
            from airwriting_imu.core.data_augmentor import DataAugmentor
            augmentor = DataAugmentor()
            original_count = len(rendered)
            
            augmented = []
            for r in rendered:
                fpath = r["path"]
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                strokes = data.get("strokes", [])
                aug_list = augmentor.augment_sample(strokes, n_augments=augment_factor)
                for aug_strokes in aug_list:
                    aug_img = self.renderer.render(aug_strokes)
                    augmented.append({
                        "image": aug_img,
                        "label": r["label"],
                    })
            rendered.extend(augmented)
            log(f"데이터 증강: {original_count} → {len(rendered)}개 (×{augment_factor})", "success")
        
        # 4. 시계열 데이터 준비 (GestureDataset에서 가져옴)
        self.dataset = GestureDataset(data_dir=data_dir)
        self.scaler = self.dataset.scaler
        
        # 5. JW v1 모델 생성
        from airwriting_imu.core.jw_v1 import JWv1
        self.model = JWv1(
            codebook_size=512,
            d_model=128,
            n_layers=4,
            d_state=16,
            num_classes=num_classes,
        ).to(self.device)
        
        report = self.model.count_parameters()
        log(f"JW v1 모델 생성: {report['total']:,} params ({report['size_mb_fp16']:.2f}MB FP16)", "info")
        
        # 6. 학습
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        # 간이 데이터로더 (이미지 + 시계열 쌍)
        images = []
        imu_seqs = []
        labels = []
        
        for r in rendered:
            img = r["image"]  # [H, W] float32
            lbl = self.label_map[r["label"]]
            images.append(torch.tensor(img, dtype=torch.float32).unsqueeze(0))  # [1, H, W]
            labels.append(lbl)
        
        # 시계열은 dataset에서 매핑 (증강분은 원본 재사용)
        for i in range(len(images)):
            idx = min(i, len(self.dataset) - 1) if len(self.dataset) > 0 else 0
            if len(self.dataset) > 0:
                seq, _ = self.dataset[idx % len(self.dataset)]
                imu_seqs.append(seq)
            else:
                imu_seqs.append(torch.zeros(200, 8))
        
        images_t = torch.stack(images).to(self.device)
        imu_t = torch.stack(imu_seqs).to(self.device)
        labels_t = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        n_samples = len(labels_t)
        best_loss = float('inf')
        patience = 0
        final_acc = 0.0
        
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            # 미니배치
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                idx = indices[start:end]
                
                batch_imu = imu_t[idx]
                batch_img = images_t[idx]
                batch_lbl = labels_t[idx]
                
                optimizer.zero_grad()
                logits, vq_loss = self.model(batch_imu, batch_img, mode="classify")
                cls_loss = criterion(logits, batch_lbl)
                loss = cls_loss + 0.1 * vq_loss  # VQ loss 가중치 0.1
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == batch_lbl).sum().item()
                total += len(batch_lbl)
            
            scheduler.step()
            n_batches = max(1, (n_samples + batch_size - 1) // batch_size)
            avg_loss = total_loss / n_batches
            acc = correct / max(total, 1) * 100
            final_acc = acc
            
            progress = int((epoch + 1) / epochs * 100)
            log(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}",
                "info", epoch=epoch+1, accuracy=f"{acc:.1f}", progress=progress, loss=avg_loss, acc=acc/100)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience = 0
            else:
                patience += 1
                if patience >= 7:
                    log(f"Early stopping at epoch {epoch+1}", "warn")
                    break
        
        # 7. 저장
        os.makedirs("weights", exist_ok=True)
        torch.save(self.model.state_dict(), "weights/jw_v1.pt")
        import pickle
        with open("weights/meta.pkl", "wb") as f:
            pickle.dump({
                "label_map": self.label_map,
                "scaler": self.scaler,
                "model_type": "jw_v1",
                "image_size": image_size,
            }, f)
        
        log(f"JW v1 학습 완료! {num_classes}개 클래스, 정확도 {final_acc:.1f}%", "success")
        return True, f"학습 성공! {num_classes}개 클래스, 정확도 {final_acc:.1f}%"
    
    def _train_legacy(self, data_dir, epochs):
        """기존 CNN-BiLSTM/Transformer 학습 (하위 호환)"""
        print(f"AI {self.model_type.upper()} 학습 모듈 초기화...")
        self.dataset = GestureDataset(data_dir=data_dir)
        
        if len(self.dataset) == 0:
            return False, "데이터를 먼저 수집하세요."
            
        self.label_map = self.dataset.label_map
        num_classes = len(self.label_map)
        
        if num_classes < 2:
            return False, "최소 2종류 이상의 글자를 수집하세요."
        
        self.model = GestureCNNBiLSTM(num_classes=num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        acc = 0
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                pred = torch.argmax(out, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
                
            scheduler.step()
            avg_loss = total_loss / len(loader)
            acc = correct / total * 100
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 7:
                    break
            
        os.makedirs("weights", exist_ok=True)
        torch.save(self.model.state_dict(), "weights/cnn_bilstm.pt")
        import pickle
        with open("weights/meta.pkl", "wb") as f:
            pickle.dump({
                "label_map": self.label_map,
                "scaler": self.dataset.scaler,
                "model_type": "cnn_bilstm"
            }, f)
            
        return True, f"학습 성공! {num_classes}개 클래스, 정확도 {acc:.1f}%"
        
    def load_model(self):
        import pickle
        try:
            with open("weights/meta.pkl", "rb") as f:
                meta = pickle.load(f)
                self.label_map = meta["label_map"]
                self.scaler = meta.get("scaler")
                model_type = meta.get("model_type", "transformer")
                
            num_classes = len(self.label_map)
            
            if model_type == "jw_v1":
                from airwriting_imu.core.jw_v1 import JWv1
                self.model = JWv1(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load("weights/jw_v1.pt", map_location=self.device))
                # 이미지 렌더러 초기화
                from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
                img_size = meta.get("image_size", 128)
                self.renderer = TrajectoryRenderer(size=img_size)
            elif model_type == "cnn_bilstm":
                self.model = GestureCNNBiLSTM(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load("weights/cnn_bilstm.pt", map_location=self.device))
            else:
                self.model = GestureTransformer(num_classes=num_classes).to(self.device)
                self.model.load_state_dict(
                    torch.load("weights/transformer.pt", map_location=self.device))
                
            self.model.eval()
            self.model_type = model_type
            print(f"모델 로드 완료: {model_type} ({num_classes}개 클래스)")
            return True
        except Exception as e:
            print(f"모델 로드 실패 (아직 학습 전): {e}")
            return False
            
    def predict(self, session_strokes):
        if self.model is None or not self.label_map:
            return None
        
        if self.model_type == "jw_v1":
            return self._predict_jw_v1(session_strokes)
        else:
            return self._predict_legacy(session_strokes)
    
    def _predict_jw_v1(self, session_strokes):
        """JW v1 추론: 궤적 이미지 + IMU 시계열 동시 입력"""
        if self.renderer is None:
            from airwriting_imu.core.trajectory_renderer import TrajectoryRenderer
            self.renderer = TrajectoryRenderer(size=128)
        
        # 궤적 → 이미지
        img = self.renderer.render(session_strokes)
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 궤적 → IMU 시계열 (flatten)
        flattened = []
        for st in session_strokes:
            for pt in st:
                features = [
                    pt.get('x', 0.0), pt.get('y', 0.0),
                    pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                    pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0)
                ]
                flattened.append(features)
        
        if len(flattened) < 5:
            return None
        
        seq = np.array(flattened)
        if self.scaler:
            seq = self.scaler.transform(seq)
        
        if len(seq) > 200:
            seq = seq[:200]
        else:
            pad_len = 200 - len(seq)
            seq = np.pad(seq, ((0, pad_len), (0, 0)), mode='constant')
        
        imu_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # JW v1 추론
        label, conf, top_k = self.model.predict(imu_t, img_t, self.label_map)
        if label:
            print(f"[JW v1] 예측: {label} (신뢰도: {conf*100:.1f}%)")
        return label
    
    def _predict_legacy(self, session_strokes):
        """기존 CNN-BiLSTM 추론"""
        flattened = []
        for st in session_strokes:
            for pt in st:
                features = [
                    pt.get('x', 0.0), pt.get('y', 0.0),
                    pt.get('ax', 0.0), pt.get('ay', 0.0), pt.get('az', 0.0),
                    pt.get('gx', 0.0), pt.get('gy', 0.0), pt.get('gz', 0.0)
                ]
                flattened.append(features)
                
        if len(flattened) < 5:
            return None
            
        seq = np.array(flattened)
        seq_norm = self.scaler.transform(seq)
        
        if len(seq_norm) > 200:
            seq_norm = seq_norm[:200]
        else:
            pad_len = 200 - len(seq_norm)
            seq_norm = np.pad(seq_norm, ((0, pad_len), (0, 0)), mode='constant')
            
        x = torch.tensor(seq_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x)
            probs = torch.softmax(out, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            conf = confidence.item()
            idx = pred_idx.item()
            
        if conf < 0.5:
            return None
            
        for label, i in self.label_map.items():
            if i == idx:
                print(f"[AI] 예측: {label} (신뢰도: {conf*100:.1f}%)")
                return label
        return None
    
    def export_onnx(self, path="weights/model.onnx"):
        if self.model is None:
            print("모델이 없습니다. 먼저 학습하세요.")
            return False
        
        if self.model_type == "jw_v1":
            self.model.export_onnx("weights/jw_v1.onnx")
        else:
            self.model.eval()
            dummy = torch.randn(1, 200, 8).to(self.device)
            torch.onnx.export(
                self.model, dummy, path,
                opset_version=17,
                input_names=["imu_sequence"],
                output_names=["prediction"],
                dynamic_axes={"imu_sequence": {0: "batch"}}
            )
            print(f"ONNX 변환 완료: {path}")
        return True
