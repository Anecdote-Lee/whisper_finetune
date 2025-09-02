# Whisper LoRA Fine-tuning: Online/Offline Setup

이 프로젝트는 Whisper 모델을 LoRA (Low-Rank Adaptation) 방법으로 파인튜닝하는 것을 온라인과 오프라인 환경으로 분리한 구현입니다.

## 📁 프로젝트 구조

```
whisper_offline_training/
├── config.py              # 설정 파일
├── utils.py               # 유틸리티 함수들
├── download_data.py       # 온라인: 데이터셋 다운로드 및 전처리
├── download_model.py      # 온라인: 베이스 모델 다운로드
├── run_online_setup.py    # 온라인: 전체 설정 실행 스크립트
├── train_offline.py       # 오프라인: 모델 학습
├── evaluate_model.py      # 오프라인: 모델 평가
├── inference.py           # 오프라인: 추론 및 데모
├── requirements.txt       # 의존성 패키지 목록
└── README.md             # 사용 가이드
```

## 🚀 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. CUDA 설정 (선택사항)

`config.py`에서 사용할 GPU 디바이스를 설정할 수 있습니다:

```python
cuda_visible_devices: str = "0"  # 사용할 GPU ID
```

## 📖 사용 방법

### Phase 1: 온라인 설정 (인터넷 연결 필요)

온라인 환경에서 다음 스크립트를 실행하여 모든 데이터와 모델을 다운로드하고 전처리합니다:

```bash
python run_online_setup.py
```

또는 개별적으로 실행:

```bash
# 데이터셋 다운로드 및 전처리
python download_data.py

# 베이스 모델 다운로드
python download_model.py
```

이 단계에서 다음 작업이 수행됩니다:
- Common Voice 한국어 데이터셋 다운로드
- 오디오 데이터 리샘플링 (48kHz → 16kHz)
- 특성 추출 및 토크나이제이션
- Whisper 베이스 모델 캐싱
- 모든 데이터를 로컬 디스크에 저장

### Phase 2: 오프라인 학습 (인터넷 연결 불필요)

오프라인 환경에서 사전 처리된 데이터를 사용하여 학습을 진행합니다:

```bash
# 모델 학습
python train_offline.py

# 모델 평가
python evaluate_model.py

# 추론 및 데모
python inference.py
```

## ⚙️ 설정 옵션

`config.py`에서 다음 설정을 수정할 수 있습니다:

### 모델 설정
- `model_name_or_path`: 베이스 Whisper 모델 ("openai/whisper-small")
- `language`: 대상 언어 ("Korean")
- `language_abbr`: 언어 코드 ("ko")
- `task`: 작업 유형 ("transcribe")

### LoRA 설정
- `lora_r`: LoRA rank (32)
- `lora_alpha`: LoRA alpha (64)
- `lora_dropout`: LoRA dropout (0.05)
- `target_modules`: 적용할 모듈 (["q_proj", "v_proj"])

### 학습 설정
- `per_device_train_batch_size`: 학습 배치 크기 (8)
- `learning_rate`: 학습률 (1e-3)
- `num_train_epochs`: 에폭 수 (3)
- `warmup_steps`: 워밍업 스텝 (50)

## 📊 특징

### LoRA (Low-Rank Adaptation)
- 전체 모델의 약 1.4%만 학습 (Parameter-Efficient Fine-Tuning)
- 메모리 효율적인 학습
- 빠른 학습 속도

### 8-bit 양자화
- `bitsandbytes`를 사용한 INT8 학습
- GPU 메모리 사용량 대폭 감소
- Colab T4 GPU (16GB VRAM)에서도 실행 가능

### 오프라인 지원
- 모든 데이터와 모델을 로컬에 캐싱
- 인터넷 연결 없이 학습 가능
- 재현 가능한 실험 환경

## 🔍 파일 설명

### `config.py`
모든 설정을 관리하는 중앙 설정 파일입니다.

### `utils.py`
데이터 전처리, 메트릭 계산 등 공통 유틸리티 함수들을 포함합니다.

### `download_data.py` (온라인 필요)
Common Voice 데이터셋을 다운로드하고 Whisper 입력 형식으로 전처리합니다.

### `download_model.py` (온라인 필요)
Whisper 베이스 모델을 다운로드하고 로컬에 캐싱합니다.

### `train_offline.py` (오프라인 가능)
LoRA를 사용하여 Whisper 모델을 파인튜닝합니다.

### `evaluate_model.py` (오프라인 가능)
학습된 모델을 평가하고 WER(Word Error Rate)을 계산합니다.

### `inference.py` (오프라인 가능)
학습된 모델로 추론을 실행하거나 Gradio 데모를 실행합니다.

## 🎯 워크플로우

1. **온라인 환경에서**:
   ```bash
   python run_online_setup.py
   ```

2. **오프라인 환경으로 파일 이동**:
   - 전체 프로젝트 폴더를 오프라인 환경으로 복사

3. **오프라인 환경에서**:
   ```bash
   python train_offline.py    # 학습
   python evaluate_model.py   # 평가
   python inference.py        # 추론
   ```

## 📝 주의사항

1. **메모리 요구사항**: 최소 16GB GPU 메모리 권장
2. **디스크 공간**: 데이터셋과 모델 캐싱을 위해 충분한 공간 확보
3. **Python 버전**: Python 3.8 이상 권장
4. **CUDA**: CUDA 지원 GPU 필요

## 🔧 트러블슈팅

### ImportError 발생 시
```bash
pip install --upgrade transformers datasets peft bitsandbytes
```

### CUDA 메모리 부족 시
`config.py`에서 배치 크기를 줄이세요:
```python
per_device_train_batch_size: int = 4  # 기본값: 8
per_device_eval_batch_size: int = 4   # 기본값: 8
```

### 학습 속도 개선
`config.py`에서 gradient accumulation을 조정하세요:
```python
gradient_accumulation_steps: int = 2  # 기본값: 1
```

## 📄 라이선스

이 프로젝트는 원본 Hugging Face 노트북을 기반으로 하며, 교육 및 연구 목적으로 사용됩니다. 