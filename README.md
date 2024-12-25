# CEPSTRUM ANALYSIS

## Data Normalization
![image](https://github.com/user-attachments/assets/aab5de2d-5903-499a-ad3c-228d65d93d14)
### 대역 통과 필터 적용
![image](https://github.com/user-attachments/assets/b4364824-3805-42d4-9cb8-938a2bb90d1e)


## CEPSTRUM


```
def compute_cepstrum(signal):
    # 신호에 FFT를 적용해 스펙트럼을 계산
    spectrum = fft(signal)
    # 스펙트럼의 로그 절댓값을 계산
    epsilon = 1e-8
    log_amplitude = np.log(np.absolute(spectrum + epsilon))
    # 로그 스펙트럼에 IFFT를 적용해 켑스트럼을 계산
    cepstrum = np.abs(ifft(log_amplitude))
    return cepstrum
```
```
def compute_human_cepstrum(xs):
    cepstrum = compute_cepstrum(xs)
    quefrencies = np.array(range(len(xs))) / 16000

    # Filter values that are not within human pitch range
    # highest frequency
    period_lb = 1 / 270
    # lowest frequency
    period_ub = 1 / 70

    cepstrum_filtered = []
    quefrencies_filtered = []
    for i, quefrency in enumerate(quefrencies):
        if quefrency < period_lb or quefrency > period_ub:
            continue

        quefrencies_filtered.append(quefrency)
        cepstrum_filtered.append(cepstrum[i])

    return quefrencies_filtered, cepstrum_filtered
```
입력 신호의 고속 푸리에 변환(FT)

스펙트럼의 절대값을 취한 후 자연로그 계산

역 FFT (FFT)를 적용하여 다시 캡스트럼 도메인으로 가져온다.
- 사람에서 나올 수 있는 주파수 범위를 필터링하는 과정

quefrency 배열을 생성 (주파수의 역수)

인간의 음성에서 나오는 일반적인 주파수 범위로 필터링

지정한 주파수 범위보다 높거나 낮으면 쓰지 않고 건너뛴다.

범위 안에 속한 값은 리스트에 저장

## 결과
![image](https://github.com/user-attachments/assets/ea66b518-22d6-44d2-a4ca-f3cd0d2559ee)
![image](https://github.com/user-attachments/assets/72633213-39d9-4735-9075-0b33b5e1def3)
![image](https://github.com/user-attachments/assets/cd1925c2-53b6-4b90-9e61-5d1e1e45ed0f)

