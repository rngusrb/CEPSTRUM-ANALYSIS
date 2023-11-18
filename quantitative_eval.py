from scipy.spatial.distance import euclidean, cosine
from scipy.stats import ttest_ind
import numpy as np


# 모두 소수점 아래 4자리까지 출력합니다.
def compute_cpp(cepstrum):
    peak = np.argmax(cepstrum)
    prominence = cepstrum[peak] - min(cepstrum)
    return round(prominence, 4)


def compute_mean_std(cepstrum):
    mean = np.mean(cepstrum)
    std = np.std(cepstrum)
    return round(mean, 4), round(std, 4)


def compute_correlation(cepstrum1, cepstrum2):
    correlation = np.corrcoef(cepstrum1, cepstrum2)[0, 1]
    return round(correlation, 4)


def compute_distances(cepstrum1, cepstrum2):
    euclidean_distance = euclidean(cepstrum1, cepstrum2)
    cosine_similarity = 1 - cosine(cepstrum1, cepstrum2)  # 코사인 유사도
    return round(euclidean_distance, 4), round(cosine_similarity, 4)


def perform_t_test(group1, group2):
    t_statistic, p_value = ttest_ind(group1, group2)
    return round(t_statistic, 4), round(p_value, 4)
