import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import os
from pandas.api.types import is_numeric_dtype
from scipy import stats

np.random.seed(42)


def reject_outliers(data, m=3):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def mad_based_outlier(points, thresh=3.5):
    if len(points.shape) == 1:
        points = points[:, None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)
    modified_z_score = 0.6745 * diff / med_abs_deviation
    return modified_z_score > thresh
    # 출처: https://pythonanalysis.tistory.com/7 [Python 데이터 분석]


def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    result = a[np.where((a >= quartileSet[0]) & (a <= quartileSet[1]))]

    return result


def normal_dist_test(df):
    for col in list(df.columns):
        # col = df.columns[0]
        print(col)
        x1 = df.loc[df[col].notnull(), col]
        print(x1.shape)
        x1 = x1[x1 != 0.0]
        print(x1.shape)
        x1 = removeOutliers(x1, 1.5)
        print(x1.shape)
        sns.distplot(x1, rug=True, label=col)

        # N = x1.shape[0]
        # x2 = sp.stats.norm(0, 1).rvs(N)
        # sns.distplot(x2, rug=True, label='norm')

        plt.legend(title="compare dist")

        plt.figure(figsize=(12, 6))
        plt.title(col)
        plt.show()

        # plt.hist(x1, 12)

        # print(sp.stats.ks_2samp(x1, x2).pvalue < 0.05)  # Kolmogorov-Smirnov 검정 0.05, 이하라면 둘은 다른 분포.
        print(sp.stats.shapiro(x1)[1] < 0.05)  # shapiro wilks 검정 0.05, 이하라면 정규분포가 아니다.



def shpiro_dist_test(x1):
    # print(sp.stats.ks_2samp(x1, x2).pvalue < 0.05)  # Kolmogorov-Smirnov 검정 0.05, 이하라면 둘은 다른 분포.
    # print(sp.stats.shapiro(x1)[1] < 0.05)
    return sp.stats.shapiro(x1)[1] > 0.05  # 0.05보다 크면 정규분포라는 귀무가설 기각


def compare_mean(x1, x2):
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    print("mean of first  one is ", mean1)
    print("mean of second one is ", mean2)
    if mean1 < mean2:
        print("so mean2 is bigger")
    else:
        print("so mean1 is bigger")


def equ_var_test_and_unpaired_t_test(x1, x2):  # 모든 조합으로 독립표본 t-test 실시. 일단 다른 변수로 감안.(같다면 등분산 t-test라고 생각)
    # 등분산성 확인. 가장 기본적인 방법은 F분포를 사용하는 것이지만 실무에서는 이보다 더 성능이 좋은 bartlett, fligner, levene 방법을 주로 사용.
    # https://datascienceschool.net/view-notebook/14bde0cc05514b2cae2088805ef9ed52/
    alpha = 0.05
    if stats.levene(x1, x2).pvalue < 0.05:  # 이보다 적으면 등분산.
        tTestResult = stats.ttest_ind(x1, x2, equal_var=True)
        print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        # if tTestResult.pvalue < 0.05:
        if (tTestResult[0] < 0) & (tTestResult[1]/2 < alpha):
            compare_mean(x1, x2)
            print("reject null hypothesis, mean of {} is less than mean of {}".format('X1', 'X2'))
        else:
            compare_mean(x1, x2)
            print("two sample mean is same h0 accepted")
    else:
        tTestResult = stats.ttest_ind(x1, x2, equal_var=False)  # 등분산이 아니므로 Welch’s t-test
        print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        # if tTestResult.pvalue < 0.05:
        if (tTestResult[0] < 0) & (tTestResult[1]/2 < alpha):
            compare_mean(x1, x2)
            print("reject null hypothesis, mean of {} is less than mean of {}".format('X1', 'X2'))
        else:
            compare_mean(x1, x2)
            print("two sample mean is same h0 accepted")
    return tTestResult

# 전이 후보다 작다는걸 보여야함. 즉 x1

# df = pd.read_csv('notebook_result.csv')
#
# print('nb_acc_komoran_result')
# nb_acc_komoran_result = equ_var_test_and_unpaired_t_test(list(df.iloc[:, 0]), list(df.iloc[:, 2]))
# print('nb_f1_komoran_result')
# nb_f1_komoran_result = equ_var_test_and_unpaired_t_test(list(df.iloc[:, 1]), list(df.iloc[:, 3]))
# print('nb_acc_hannanum_result')
# nb_acc_hannanum_result = equ_var_test_and_unpaired_t_test(list(df.iloc[:, 0]), list(df.iloc[:, 4]))
# print('nb_f1_hannanum_result')
# nb_f1_hannanum_result = equ_var_test_and_unpaired_t_test(list(df.iloc[:, 1]), list(df.iloc[:, 5]))
#
#
# print('svm_acc_komoran_result')
# svm_acc_komoran_result = equ_var_test_and_unpaired_t_test(list(df.iloc[:, 6]), list(df.iloc[:, 8]))
# print('svm_f1_komoran_result')
# svm_f1_komoran_result = equ_var_test_and_unpaired_t_test(list(df.iloc[:, 7]), list(df.iloc[:, 9]))

