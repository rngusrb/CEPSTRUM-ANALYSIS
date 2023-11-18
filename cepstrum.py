import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
from quantitative_eval import (
    compute_cpp,
    compute_mean_std,
    compute_correlation,
    compute_distances,
    perform_t_test,
)

A = 0
E = 1
I = 2
O = 3
U = 4
MALE = 0
FEMALE = 0
FS = 16000
BY_GENDER = 5


def compute_cepstrum(signal):
    # 신호에 FFT를 적용해 스펙트럼을 계산
    spectrum = fft(signal)
    # 스펙트럼의 로그 절댓값을 계산
    log_amplitude = np.log(np.absolute(spectrum))
    # 로그 스펙트럼에 IFFT를 적용해 켑스트럼을 계산
    cepstrum = np.abs(ifft(log_amplitude))
    return cepstrum


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

    return cepstrum_filtered


def plot_signal_and_cepstrum(signal, cepstrum, fs, title):
    time = np.arange(len(signal)) / fs
    quefrency = np.arange(len(cepstrum)) / fs

    plt.figure(figsize=(12, 6))

    # Plot the time-domain signal
    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title(f"Time-Domain Signal - {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Plot the cepstrum
    plt.subplot(2, 1, 2)
    plt.plot(quefrency, cepstrum)
    plt.title(f"Cepstrum - {title}")
    plt.xlabel("Quefrency (s)")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def compare_two_cepstrum(cepstrum1, cepstrum2, standard):
    FS = 16000  # 샘플링 레이트를 가정합니다.
    quefrency1 = np.arange(len(cepstrum1)) / FS
    quefrency2 = np.arange(len(cepstrum2)) / FS

    if standard == A:
        title1 = "male_a"
        title2 = "female_a"
    elif standard == E:
        title1 = "male_e"
        title2 = "female_e"
    elif standard == I:
        title1 = "male_i"
        title2 = "female_i"
    elif standard == O:
        title1 = "male_o"
        title2 = "female_o"
    elif standard == U:
        title1 = "male_u"
        title2 = "female_u"

    row_labels = ["male", "female"]
    column_labels = [
        "Cepstral Peak Prominence",
        "Mean / Standard",
        "Correlation",
        "Distances",
        "T-Test",
    ]
    cell_text = [
        [
            compute_cpp(cepstrum1),
            compute_mean_std(cepstrum1),
            compute_correlation(cepstrum1, cepstrum2),
            compute_distances(cepstrum1, cepstrum2),
            perform_t_test(cepstrum1, cepstrum2),
        ],
        [
            compute_cpp(cepstrum2),
            compute_mean_std(cepstrum2),
            compute_correlation(cepstrum1, cepstrum2),
            compute_distances(cepstrum1, cepstrum2),
            perform_t_test(cepstrum1, cepstrum2),
        ],
    ]
    # 전체 그림 및 서브플롯 설정
    fig = plt.figure(figsize=(12, 10))

    # 첫 번째 켑스트럼 플롯
    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(quefrency1, cepstrum1)
    ax1.set_title(f"Cepstrum - {title1}")
    ax1.set_xlabel("Quefrency (s)")
    ax1.set_ylabel("Amplitude")

    # 두 번째 켑스트럼 플롯
    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(quefrency2, cepstrum2)
    ax2.set_title(f"Cepstrum - {title2}")
    ax2.set_xlabel("Quefrency (s)")
    ax2.set_ylabel("Amplitude")

    # 테이블 추가
    ax3 = fig.add_subplot(4, 1, 3)
    ax3.axis("off")  # 테이블에는 축이 필요 없으므로 숨깁니다.
    table = ax3.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=column_labels,
        loc="center",
        cellLoc="center",  # 셀 내부 글자 정렬
        fontsize=10,  # 글자 크기 조정
    )
    table.scale(1, 2)  # 테이블 크기 조정 (폭, 높이)

    plt.tight_layout()
    plt.show()


def get_cepstrum_arrays():
    male_cepstrums, female_cepstrums = [], []
    fs1, a = wavfile.read("./normalized_data/male/male_0_a.wav")
    fs2, e = wavfile.read("./normalized_data/male/male_0_e.wav")
    fs3, i = wavfile.read("./normalized_data/male/male_0_i.wav")
    fs4, o = wavfile.read("./normalized_data/male/male_0_o.wav")
    fs5, u = wavfile.read("./normalized_data/male/male_0_u.wav")
    male_cepstrums.append(compute_human_cepstrum(compute_cepstrum(a)))
    male_cepstrums.append(compute_human_cepstrum(compute_cepstrum(e)))
    male_cepstrums.append(compute_human_cepstrum(compute_cepstrum(i)))
    male_cepstrums.append(compute_human_cepstrum(compute_cepstrum(o)))
    male_cepstrums.append(compute_human_cepstrum(compute_cepstrum(u)))
    fs1, a = wavfile.read("./normalized_data/female/female_4_a.wav")
    fs2, e = wavfile.read("./normalized_data/female/female_4_e.wav")
    fs3, i = wavfile.read("./normalized_data/female/female_4_i.wav")
    fs4, o = wavfile.read("./normalized_data/female/female_4_o.wav")
    fs5, u = wavfile.read("./normalized_data/female/female_4_u.wav")
    female_cepstrums.append(compute_human_cepstrum(compute_cepstrum(a)))
    female_cepstrums.append(compute_human_cepstrum(compute_cepstrum(e)))
    female_cepstrums.append(compute_human_cepstrum(compute_cepstrum(i)))
    female_cepstrums.append(compute_human_cepstrum(compute_cepstrum(o)))
    female_cepstrums.append(compute_human_cepstrum(compute_cepstrum(u)))
    a_ceptrums, e_ceptrums, i_ceptrums, o_ceptrums, u_ceptrums = [], [], [], [], []
    a_ceptrums.append(male_cepstrums[A])
    a_ceptrums.append(female_cepstrums[A])
    e_ceptrums.append(male_cepstrums[E])
    e_ceptrums.append(female_cepstrums[E])
    i_ceptrums.append(male_cepstrums[I])
    i_ceptrums.append(female_cepstrums[I])
    o_ceptrums.append(male_cepstrums[O])
    o_ceptrums.append(female_cepstrums[O])
    u_ceptrums.append(male_cepstrums[U])
    u_ceptrums.append(female_cepstrums[U])
    return (
        male_cepstrums,
        female_cepstrums,
        a_ceptrums,
        e_ceptrums,
        i_ceptrums,
        o_ceptrums,
        u_ceptrums,
    )


(
    male_cepstrums,
    female_cepstrums,
    a_ceptrums,
    e_ceptrums,
    i_ceptrums,
    o_ceptrums,
    u_ceptrums,
) = get_cepstrum_arrays()


compare_two_cepstrum(male_cepstrums[A], female_cepstrums[A], A)
compare_two_cepstrum(male_cepstrums[E], female_cepstrums[E], E)
compare_two_cepstrum(male_cepstrums[I], female_cepstrums[I], I)
compare_two_cepstrum(male_cepstrums[O], female_cepstrums[O], O)
compare_two_cepstrum(male_cepstrums[U], female_cepstrums[U], U)
