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

def get_mean():
    mean_male=[]
    mean_female=[]
    for n in ('a', 'e', 'i', 'o', 'u'):
        for gender in ('male','female'):
            fs1, a1 = wavfile.read("./"+gender+"/normalized_data_"+gender+"_"+gender+"_0_"+n+".wav")
            full_data = [0 for i in range(len(a1))]
            for i in ('0', '1', '2', '3', '4'):
                name = "./"+gender+"/normalized_data_"+gender+"_"+gender+"_" + i + "_"+n+".wav"
                fs1, a2 = wavfile.read(name)
                for i in range(len(a1)):
                    full_data[i] += a2[i]
            for i in range(len(a1)):
                full_data[i]=full_data[i] / 5
            if (gender=="male"):
                mean_male.append(compute_human_cepstrum(full_data))
            else:
                mean_female.append(compute_human_cepstrum(full_data))

    return mean_male,mean_female



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


def compare_two_mean_cepstrum(cepstrum1, cepstrum2, standard):
    FS = 16000  # 샘플링 레이트를 가정합니다.
    quefrency1 = np.arange(len(cepstrum1)) / FS
    quefrency2 = np.arange(len(cepstrum2)) / FS

    if standard == A:
        title1 = "mean_male_a"
        title2 = "mean_female_a"
    elif standard == E:
        title1 = "mean_male_e"
        title2 = "mean_female_e"
    elif standard == I:
        title1 = "mean_male_i"
        title2 = "mean_female_i"
    elif standard == O:
        title1 = "mean_male_o"
        title2 = "mean_female_o"
    elif standard == U:
        title1 = "mean_male_u"
        title2 = "mean_female_u"

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

def compare_all_cepstrum(cepstrum1, cepstrum2, cepstrum3,cepstrum4,cepstrum5):
    FS = 16000  # 샘플링 레이트를 가정합니다.
    quefrency1 = np.arange(len(cepstrum1)) / FS
    quefrency2 = np.arange(len(cepstrum2)) / FS
    quefrency3 = np.arange(len(cepstrum3)) / FS
    quefrency4 = np.arange(len(cepstrum4)) / FS
    quefrency5 = np.arange(len(cepstrum5)) / FS

    fig = plt.figure(figsize=(8, 8))

    # 첫 번째 켑스트럼 플롯
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(quefrency1, cepstrum1)
    ax1.set_xlabel("Quefrency (s)")
    ax1.set_ylabel("Amplitude")

    # 두 번째 켑스트럼 플롯
    ax2 = fig.add_subplot(5, 1, 2)
    ax2.plot(quefrency2, cepstrum2)
    ax2.set_xlabel("Quefrency (s)")
    ax2.set_ylabel("Amplitude")

    ax3 = fig.add_subplot(5, 1, 3)
    ax3.plot(quefrency3, cepstrum3)
    ax3.set_xlabel("Quefrency (s)")
    ax3.set_ylabel("Amplitude")

    ax4 = fig.add_subplot(5, 1, 4)
    ax4.plot(quefrency4, cepstrum4)
    ax4.set_xlabel("Quefrency (s)")
    ax4.set_ylabel("Amplitude")

    ax5 = fig.add_subplot(5, 1, 5)
    ax5.plot(quefrency5, cepstrum5)
    ax5.set_xlabel("Quefrency (s)")
    ax5.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def get_cepstrum_arrays():
    male_cepstrums, female_cepstrums = [], []
    for gender in ('male','female'):
        for i in ('a', 'e', 'i', 'o', 'u'):
            name = "./"+gender+"/normalized_data_"+gender+"_"+gender+"_0_" + i + ".wav"
            fs1, data = wavfile.read(name)
            if (gender=="male"):
                male_cepstrums.append(compute_human_cepstrum(data))
            else:
                female_cepstrums.append(compute_human_cepstrum(data))
    return (
        male_cepstrums,
        female_cepstrums,
    )



(
    male_cepstrums,
    female_cepstrums,
) = get_cepstrum_arrays()

mean_male,mean_female=get_mean()

get_mean()



'''for n in ('a','e','i','o','u'):
    list = []
    flist = []
    for gender in ('male', 'female'):
        for i in ('0', '1', '2', '3', '4'):
            name = "./" + gender + "/normalized_data_" + gender + "_" + gender + "_" + i + "_"+n+".wav"
            fs1, data = wavfile.read(name)
            if (gender == 'male'):
                list.append(compute_human_cepstrum(data))
            elif (gender == 'female'):
                flist.append(compute_human_cepstrum(data))
    compare_all_cepstrum(list[0], list[1], list[2], list[3], list[4])
    compare_all_cepstrum(flist[0], flist[1], flist[2], flist[3], flist[4])'''



compare_two_mean_cepstrum(mean_male[A], mean_female[A], A)
compare_two_mean_cepstrum(mean_male[E], mean_female[E], E)
compare_two_mean_cepstrum(mean_male[I], mean_female[I], I)
compare_two_mean_cepstrum(mean_male[O], mean_female[O], O)
compare_two_mean_cepstrum(mean_male[U], mean_female[U], U)

'''compare_two_cepstrum(male_cepstrums[A], female_cepstrums[A], A)
compare_two_cepstrum(male_cepstrums[E], female_cepstrums[E], E)
compare_two_cepstrum(male_cepstrums[I], female_cepstrums[I], I)
compare_two_cepstrum(male_cepstrums[O], female_cepstrums[O], O)
compare_two_cepstrum(male_cepstrums[U], female_cepstrums[U], U)'''