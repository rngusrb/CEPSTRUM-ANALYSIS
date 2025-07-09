# 📊 Cepstrum Analysis

## 프로젝트 개요

해당 프로젝트는 음성 신호로부터 인간의 발화 특성을 추출하고자 **FFT 기반 캡스트럼 분석(Cepstrum Analysis)**을 수행한 실험 프로젝트입니다. 
신호처리의 전처리 단계부터 고유한 주파수 영역 필터링까지 구현했으며, 남성과 여성, 그리고 모음 별 cepstrum 값을 정량적으로 비교하는 시각화 도구도 포함합니다.

---

## ⚙️ 전처리 및 정규화

### 정규화가 필요한 이유
- 데이터 간 단위 불일치 혹은 스케일 차이로 인해 모델의 왜곡 가능성
- 스펙트럼 해석 전, 잡음을 제거하고 진폭 및 길이 표준화 필수

### 정규화 과정
![정규화 코드](https://github.com/user-attachments/assets/d1073960-df81-48d9-a5d2-13fbb802b66c)

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

- 왜곡 방지 및 음질 향상을 위해 발음에 관련된 주파수 대역만 통과
- 주파수 범위: 80Hz ~ 4000Hz 사용

![밴드패스 필터](https://github.com/user-attachments/assets/b90f638a-3ac8-473f-921a-38e01c16a394)

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

![캡스트럼 계산](https://github.com/user-attachments/assets/1c5b7b04-f53e-4f3c-a5ac-c51912e01d87)

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

## 📈 시각화 및 비교 분석

![캡스트럼 비교](https://github.com/user-attachments/assets/42dd6fb7-1fec-4fb5-87fc-3c94591958c1)
![종합 그래프](https://github.com/user-attachments/assets/5789a474-7223-4b36-a29b-6d31045a9454)

- 각 모음(a, e, i, o, u)에 대한 남성/여성의 cepstrum 분석값 시각화
- 특정 주파수 패턴이 성별/모음에 따라 다르게 나타나는지 확인
- `compare_all_cepstrum()`, `plot_signal_and_cepstrum()` 등으로 구현

---

## 🧪 정량적 평가 지표

- 평균 cepstrum 값, 분산
- quefrency 영역별 집중도 비교
- 남성/여성 음성의 특이성 판단 가능

---

## 📝 결론

본 프로젝트는 음성 데이터의 특징 추출을 위해 신호처리 기반의 전처리 및 캡스트럼 분석을 수행하였으며, 
정규화 → 필터링 → FFT/IFFT → 사람의 발화 범위 필터링 → 시각화의 일련의 흐름을 코드로 구현했습니다.

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
