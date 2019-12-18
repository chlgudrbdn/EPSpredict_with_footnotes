import math
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
import numpy as np
import os, sys
import preprocess_footnotes_data as pfd
import random as rn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from scipy import stats



def adjusted_rmse(y_true, y_pred):  # 사실 2009년엔 없는 단어여서 그런지 모르겠지만 이걸 Mean Square Percentage Error 라고 함. mape보다 인기는 없어보인다.
    # return np.sqrt((((targets - predictions)/targets) ** 2).mean())
    return np.sq10uare(((y_true - y_pred)/y_true)).mean() * 100


def Mape(y_true, y_pred):  # 0에 가까운 값에 약해(규모에 민감)서 사실 그닥 성능은 좋지 않아보인다.
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Mpe(y_true, y_pred):  # underperformance 인지 overperformance 인지 판단 할 수 있다는 것입니다.
    return np.mean((y_true - y_pred) / y_true) * 100


def excel_to_df_list(file_name, sheet_num):
    df_list = []
    for i in range(sheet_num):
        df_list.append(pd.read_excel(file_name, sheet_name='try'+str(i)))
    return df_list


def get_error_metric_per_q(predict_df):
    # time_series = [(2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4),
    #                (2018, 1), (2018, 2), (2018, 3)]
    testRmseList = []
    testmaeList = []
    testmapeList = []
    testadjrmseList = []

    for df in predict_df:
        for t in range(1, 5):
            print('-----------------------')
            # year_q = (2016,1)  # for test
            test = df[(df['주기'] == t)]
            print('test ', test.shape)
            predictY = test['pred'].values
            true = test['true'].values
            testScore = math.sqrt(mean_squared_error(true, predictY))
            print('Test Score after more val: %.4f RMSE' % testScore)
            mspe = adjusted_rmse(true, predictY)
            print('adjusted_rmse: ', mspe)
            mae = mean_absolute_error(true, predictY)
            print('mae: {0:.4f}'.format(round(mae, 3)))
            mape = Mape(true, predictY)
            print('mape: {0:.4f}'.format(round(mape, 3)))
            mpe = Mpe(true, predictY.flatten())
            if mpe < 0:
                print(mpe, 'over perform')
            else:
                print(mpe, 'under perform')
            testRmseList.append(testScore)
            testadjrmseList.append(mspe)
            testmaeList.append(mae)
            testmapeList.append(mape)
    return testRmseList, testmaeList, testmapeList, testadjrmseList


def get_error_metric_per_q10(predict_df):
    time_series = [(2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1), (2018, 2), (2018, 3)]
    testRmseList = []
    testmaeList = []
    testmapeList = []
    testadjrmseList = []

    for df in predict_df:
        for t in range(len(time_series) - 1):  # 일종의 10-fold가 된다. 튜플들은 validation에 쓰는걸 기준. (2018, 3)은 test
            test = df[(df['회계년'] == time_series[t + 1][0]) & (df['주기'] == time_series[t + 1][1])]
            print('-----------------------', time_series[t + 1])
            print('test ', test.shape)
            predictY = test['pred'].values
            true = test['true'].values
            testScore = math.sqrt(mean_squared_error(true, predictY))
            print('Test Score after more val: %.4f RMSE' % testScore)
            mspe = adjusted_rmse(true, predictY)
            print('adjusted_rmse: ', mspe)
            mae = mean_absolute_error(true, predictY)
            print('mae: {0:.4f}'.format(round(mae, 3)))
            mape = Mape(true, predictY)
            print('mape: {0:.4f}'.format(round(mape, 3)))
            mpe = Mpe(true, predictY.flatten())
            if mpe < 0:
                print(mpe, 'over perform')
            else:
                print(mpe, 'under perform')
            testRmseList.append(testScore)
            testadjrmseList.append(mspe)
            testmaeList.append(mae)
            testmapeList.append(mape)
    return testRmseList, testmaeList, testmapeList, testadjrmseList

def get_error_metric_per_i(predict_df):
    testRmseList = []
    testmaeList = []
    testmapeList = []
    testadjrmseList = []
    ind_crp = pd.read_csv('ind_string_match_crp.csv', encoding='cp949')
    ind_list = list(ind_crp['ind'].unique())
    print(len(ind_list))
    for i in range(len(predict_df)):
        df = pd.merge(left=predict_df[i], right=ind_crp, on=['Symbol'])
        for ind in ind_list:  # 59개 산업이 있다.
            print('-----------------------try', i)
            test = df[(df['ind'] == ind)]
            print(ind, ' test ', test.shape)
            predictY = test['pred'].values
            true = test['true'].values
            testScore = math.sqrt(mean_squared_error(true, predictY))
            print('Test Score after more val: %.4f RMSE' % testScore)
            mspe = adjusted_rmse(true, predictY)
            print('adjusted_rmse: ', mspe)
            mae = mean_absolute_error(true, predictY)
            print('mae: {0:.4f}'.format(round(mae, 3)))
            mape = Mape(true, predictY)
            print('mape: {0:.4f}'.format(round(mape, 3)))
            mpe = Mpe(true, predictY.flatten())
            if mpe < 0:
                print(mpe, 'over perform')
            else:
                print(mpe, 'under perform')
            testRmseList.append(testScore)
            testadjrmseList.append(mspe)
            testmaeList.append(mae)
            testmapeList.append(mape)

    return testRmseList, testmaeList, testmapeList, testadjrmseList


def mean_per_n(result_list, n):
    result = []
    average_per_n = []
    for li in result_list:
        result.append(li)
        if len(result) == n:
            average_per_n.append(np.mean(result))
            result = []
    return average_per_n


def pick_per_n(result_list, n):
    result = []
    for i in range(n):  # 1분기, 2분기 또는 산업
        tmp = []
        for j in range(i, len(result_list), n):
            tmp.append(result_list[j])
        result.append(tmp)
    return result



def chk_row_cnt(df):
    for year in range(2013, 2019):
        print(year, '년', df[df['회계년'] == year].shape)  # 738
        for q in range(1,5):
            print(year, '년', q, '분기', df[(df['회계년'] == year) & (df['주기'] == q)].shape)  # 972  # 2013년 14년은 뭔가 이상한 데이터가 많았다.


def equ_var_test_and_unpaired_t_test(x1, x2):  # 모든 조합으로 독립표본 t-test 실시. 일단 다른 변수로 감안.(같다면 등분산 t-test라고 생각)
    # 등분산성 확인. 가장 기본적인 방법은 F분포를 사용하는 것이지만 실무에서는 이보다 더 성능이 좋은 bartlett, fligner, levene 방법을 주로 사용.
    # https://datascienceschool.net/view-notebook/14bde0cc05514b2cae2088805ef9ed52/
    alpha = 0.05
    if stats.levene(x1, x2).pvalue > 0.05:  # 이보다 적으면 이분산
        tTestResult = stats.ttest_ind(x1, x2, equal_var=True)
        print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
        # 출처: https://thenotes.tistory.com/entry/Ttest-in-python
        # if tTestResult.pvalue < 0.05:
        if (tTestResult[0] > 0) & (tTestResult[1] < alpha):
            # compare_mean(x1, x2)
            # print("reject null hypothesis, mean of {} is large than mean of {}".format('X1', 'X2'))
            pass
        else:
            # compare_mean(x1, x2)
            # print("two sample mean is same h0 accepted")
            pass
    else:
        tTestResult = stats.ttest_ind(x1, x2, equal_var=False)  # 등분산이 아니므로 Welch’s t-test
        print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        # if tTestResult.pvalue < 0.05:
        if (tTestResult[0] > 0) & (tTestResult[1] < alpha):
            # compare_mean(x1, x2)
            # print("reject null hypothesis, mean of {} is large than mean of {}".format('X1', 'X2'))
            pass
        else:
            # compare_mean(x1, x2)
            # print("two sample mean is same h0 accepted")
            pass
    return tTestResult



if __name__ == '__main__':
    base_dir = os.getcwd()

    predict_df1 = excel_to_df_list(base_dir + '/predict_df1.xlsx', 30)
    predict_df2 = excel_to_df_list(base_dir + '/predict_df2.xlsx', 30)
    predict_df3 = excel_to_df_list(base_dir + '/predict_df3.xlsx', 30)
    predict_df4 = excel_to_df_list(base_dir + '/predict_df4.xlsx', 30)

    q_rmseList1, q_maeList1, q_mapeList1, q_mspeList1 = get_error_metric_per_q(predict_df1)  # 4*30개씩 리턴. 1~4분기. 다음 시드 1~4분기
    q_rmseList2, q_maeList2, q_mapeList2, q_mspeList2 = get_error_metric_per_q(predict_df2)
    q_rmseList3, q_maeList3, q_mapeList3, q_mspeList3 = get_error_metric_per_q(predict_df3)
    q_rmseList4, q_maeList4, q_mapeList4, q_mspeList4 = get_error_metric_per_q(predict_df4)

    chk_row_cnt(predict_df1[0])

    cnt = 4
    avg_q_rmseList1 = pick_per_n(q_rmseList1, cnt)
    avg_q_maeList1 = pick_per_n(q_maeList1, cnt)
    avg_q_mapeList1 = pick_per_n(q_mapeList1, cnt)
    avg_q_mspeList1 = pick_per_n(q_mspeList1, cnt)

    avg_q_rmseList2 = pick_per_n(q_rmseList2, cnt)
    avg_q_maeList2 = pick_per_n(q_maeList2, cnt)
    avg_q_mapeList2 = pick_per_n(q_mapeList2, cnt)
    avg_q_mspeList2 = pick_per_n(q_mspeList2, cnt)

    avg_q_rmseList3 = pick_per_n(q_rmseList3, cnt)
    avg_q_maeList3 = pick_per_n(q_maeList3, cnt)
    avg_q_mapeList3 = pick_per_n(q_mapeList3, cnt)
    avg_q_mspeList3 = pick_per_n(q_mspeList3, cnt)

    avg_q_rmseList4 = pick_per_n(q_rmseList4, cnt)
    avg_q_maeList4 = pick_per_n(q_maeList4, cnt)
    avg_q_mapeList4 = pick_per_n(q_mapeList4, cnt)
    avg_q_mspeList4 = pick_per_n(q_mspeList4, cnt)

    i_rmseList1, i_maeList1, i_mapeList1, i_mspeList1 = get_error_metric_per_i(predict_df1)  # 58*30개씩 리턴
    i_rmseList2, i_maeList2, i_mapeList2, i_mspeList2 = get_error_metric_per_i(predict_df2)
    i_rmseList3, i_maeList3, i_mapeList3, i_mspeList3 = get_error_metric_per_i(predict_df3)
    i_rmseList4, i_maeList4, i_mapeList4, i_mspeList4 = get_error_metric_per_i(predict_df4)
    ind_crp = pd.read_csv('ind_string_match_crp.csv', encoding='cp949')
    ind_list = list(ind_crp['ind'].unique())
    print(len(ind_list))
    cnt = len(ind_list)
    avg_i_rmseList1 = pick_per_n(i_rmseList1, cnt)
    avg_i_maeList1 = pick_per_n(i_maeList1, cnt)
    avg_i_mapeList1 = pick_per_n(i_mapeList1, cnt)
    avg_i_mspeList1 = pick_per_n(i_mspeList1, cnt)

    avg_i_rmseList2 = pick_per_n(i_rmseList2, cnt)
    avg_i_maeList2 = pick_per_n(i_maeList2, cnt)
    avg_i_mapeList2 = pick_per_n(i_mapeList2, cnt)
    avg_i_mspeList2 = pick_per_n(i_mspeList2, cnt)

    avg_i_rmseList3 = pick_per_n(i_rmseList3, cnt)
    avg_i_maeList3 = pick_per_n(i_maeList3, cnt)
    avg_i_mapeList3 = pick_per_n(i_mapeList3, cnt)
    avg_i_mspeList3 = pick_per_n(i_mspeList3, cnt)

    avg_i_rmseList4 = pick_per_n(i_rmseList4, cnt)
    avg_i_maeList4 = pick_per_n(i_maeList4, cnt)
    avg_i_mapeList4 = pick_per_n(i_mapeList4, cnt)
    avg_i_mspeList4 = pick_per_n(i_mspeList4, cnt)

    for q in range(4):
        pfd.equ_var_test_and_unpaired_t_test(avg_q_mapeList1[q], avg_q_mapeList2[q])  # 독립 t-test 단방향 검정 # 유의
    for q in range(4):
        pfd.equ_var_test_and_unpaired_t_test(avg_q_mspeList1[q], avg_q_mspeList2[q])  # 독립 t-test 단방향 검정 # 유의

    q10_rmseList1, q10_maeList1, q10_mapeList1, q10_mspeList1 = get_error_metric_per_q10(predict_df1)  # 4*30개씩 리턴. 1~4분기. 다음 시드 1~4분기
    q10_rmseList2, q10_maeList2, q10_mapeList2, q10_mspeList2 = get_error_metric_per_q10(predict_df2)
    q10_rmseList3, q10_maeList3, q10_mapeList3, q10_mspeList3 = get_error_metric_per_q10(predict_df3)
    q10_rmseList4, q10_maeList4, q10_mapeList4, q10_mspeList4 = get_error_metric_per_q10(predict_df4)
    cnt = 10
    avg_q10_rmseList1 = pick_per_n(q10_rmseList1, cnt)
    avg_q10_maeList1 = pick_per_n(q10_maeList1, cnt)
    avg_q10_mapeList1 = pick_per_n(q10_mapeList1, cnt)
    avg_q10_mspeList1 = pick_per_n(q10_mspeList1, cnt)

    avg_q10_rmseList2 = pick_per_n(q10_rmseList2, cnt)
    avg_q10_maeList2 = pick_per_n(q10_maeList2, cnt)
    avg_q10_mapeList2 = pick_per_n(q10_mapeList2, cnt)
    avg_q10_mspeList2 = pick_per_n(q10_mspeList2, cnt)

    avg_q10_rmseList3 = pick_per_n(q10_rmseList3, cnt)
    avg_q10_maeList3 = pick_per_n(q10_maeList3, cnt)
    avg_q10_mapeList3 = pick_per_n(q10_mapeList3, cnt)
    avg_q10_mspeList3 = pick_per_n(q10_mspeList3, cnt)

    avg_q10_rmseList4 = pick_per_n(q10_rmseList4, cnt)
    avg_q10_maeList4 = pick_per_n(q10_maeList4, cnt)
    avg_q10_mapeList4 = pick_per_n(q10_mapeList4, cnt)
    avg_q10_mspeList4 = pick_per_n(q10_mspeList4, cnt)

    for q in range(10):
        pfd.equ_var_test_and_unpaired_t_test(avg_q10_mapeList1[q], avg_q10_mapeList2[q])  # 독립 t-test 단방향 검정 # 유의
    for q in range(10):
        pfd.equ_var_test_and_unpaired_t_test(avg_q10_mspeList1[q], avg_q10_mspeList2[q])  # 독립 t-test 단방향 검정 # 유의

    ind_reuslt_t = []
    ind_reuslt_p = []
    for i in range(len(ind_list)):
        result = pfd.equ_var_test_and_unpaired_t_test(avg_i_mapeList1[i], avg_i_mapeList2[i])  # 독립 t-test 단방향 검정 # 유의
        ind_reuslt_t.append(result[0])
        ind_reuslt_p.append(result[1])
