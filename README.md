# 📊 Cepstrum Analysis

## 프로젝트 개요

본 프로젝트는 인간의 발화 특성을 정량적으로 분석하기 위해 **FFT 기반 캡스트럼(Cepstrum) 분석**을 수행한 신호처리 기반 음성 인식 실험입니다. 
음성 데이터를 전처리 및 정규화한 후, 고유 주파수 특성을 추출하고 시각화함으로써 남녀 음성 차이, 모음 간 주파수 패턴 등을 분석하였습니다.

---

## ⚙️ 전처리 및 정규화

### 정규화 이유
- 특성 간 단위/범위 불일치로 인한 모델 학습 왜곡 방지
- 발화 신호의 진폭, 노이즈, 길이 등 표준화 필요

### 주요 처리 흐름

```python
def normalize_amplitude(audio):
    return audio / np.max(np.abs(audio))

def resample_audio(audio, original_sr, target_sr):
    return librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)

def remove_noise(audio, sr):
    b, a = scipy.signal.butter(N=10, Wn=1000 / (sr / 2), btype="highpass")
    return scipy.signal.filtfilt(b, a, audio)

def trim_or_pad_audio(audio, length):
    if len(audio) > length:
        return audio[:length]
    elif len(audio) < length:
        return np.pad(audio, (0, length - len(audio)), "constant")
    return audio
```

---

## 🔊 대역 통과 필터 적용

- 왜곡 방지 및 음질 향상을 위한 주파수 선택 (예: 80Hz ~ 4000Hz)
- 발화에 관련된 유효 주파수 영역 강조


```python
def bandpass_filter(audio, sr, low, high):
    nyquist = sr // 2
    low = low / nyquist
    high = high / nyquist
    b, a = scipy.signal.butter(N=4, Wn=[low, high], btype="band")
    return scipy.signal.lfilter(b, a, audio)
```

---

## 🧠 캡스트럼 분석 과정

```python
def compute_cepstrum(signal):
    spectrum = fft(signal)
    epsilon = 1e-8
    log_amplitude = np.log(np.abs(spectrum + epsilon))
    cepstrum = np.abs(ifft(log_amplitude))
    return cepstrum

def compute_human_cepstrum(xs):
    cepstrum = compute_cepstrum(xs)
    quefrencies = np.array(range(len(xs))) / 16000
    period_lb = 1 / 270
    period_ub = 1 / 70
    cepstrum_filtered = []
    quefrencies_filtered = []
    for i, q in enumerate(quefrencies):
        if q < period_lb or q > period_ub:
            continue
        quefrencies_filtered.append(q)
        cepstrum_filtered.append(cepstrum[i])
    return quefrencies_filtered, cepstrum_filtered
```

---

## 📊 분석 결과 및 인사이트

### 남녀 음성 비교
- 기본 주파수 범위:
  - 남성: 109~120Hz
  - 여성: 209~223Hz
- 남성은 낮은 음역대, 여성은 높은 음역대에 해당하는 경향

### 모음 간 비교
- 남성과 여성 모두 `e`, `i` 모음이 높은 주파수 값을 가짐
- `o`, `u`, `a` 계열은 상대적으로 낮고 유사한 값 분포
- 이로 인해 `e`, `i`, `o`, `u`, `a`의 이동 경로가 음성공간 상 하나의 벡터 방향처럼 나타남


> 위 결과는 사람의 발음 구조적 차이, 성별의 생리학적 특성, 그리고 데이터 구성에 따른 정량적 특징을 모두 반영함

---

## 📈 시각화 예시

<img width="920" alt="{0F1E65B4-ACAB-4D7D-908F-BF1CF63459A3}" src="https://github.com/user-attachments/assets/6e9db592-bfcd-41f6-8aaa-be12c9eb6d14" />

<img width="940" alt="{46873086-B413-4E8A-8CE0-F3E2F373F315}" src="https://github.com/user-attachments/assets/bf558445-f7f7-4c23-a817-c6a687859296" />

---

## 📁 폴더 구성

| 폴더/파일 | 설명 |
|------------|------|
| `raw_data/` | 원본 음성 파일 |
| `normalized_data/` | 정규화 및 필터링된 음성 파일 저장 경로 |
| `cepstrum.py` | cepstrum 계산 관련 함수 모듈 |
| `data_normalization.py` | 전처리 및 정규화 파이프라인 코드 |
| `quantitative_eval.py` | 시각화 및 정량 분석용 스크립트 |
| `main.py` | 전체 실험 실행 메인 파일 |

---

## 🔍 주요 학습 포인트
- FFT, IFFT를 활용한 신호 분석의 이론 및 구현 경험
- 주파수 필터링, 스펙트럼 해석 등 정량적 신호처리 기법 실습
- 음성 기반 데이터의 정규화 및 시각화 흐름 전반 숙지

