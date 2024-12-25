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
---

## 결과
![image](https://github.com/user-attachments/assets/ea66b518-22d6-44d2-a4ca-f3cd0d2559ee)
