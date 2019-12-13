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
from datetime import datetime
import shutil
from math import sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool


def add_one_hot(df, col_name):
    df = pd.concat([df, pd.get_dummies(df[col_name], dummy_na=False, prefix=col_name)], axis=1)
    # df.drop([col_name], axis=1, inplace=True)
    return df


def MPE(y_true, y_pred):
    return np.mean((y_true - y_pred) / y_true) * 100


def match_t_minums4():
    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , dtype=object, sheet_name='indvar_weiZhang')  # 종속변수까지 포함됨.
    t_minus_n = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , dtype=object, sheet_name='ind_weiZhang')  # 종속변수까지 포함됨.
    df = t_minus_n[['Symbol', 'Name', '결산월', '회계년', '주기',
                           't_1M000601002_EPS(원)', 't_1M000601005_EPS(보통주)(원)', 't_1M000601003_수정EPS(원)', 't_1M000601006_수정EPS(보통주)(원)']].copy()
    # 't_1M000601002_EPS(원)', 't_1M000601042_EPS(계속사업)(원)','t_1M000601005_EPS(보통주)(원)', 't_1M000601003_수정EPS(원)', 't_1M000601043_수정EPS(계속사업)(원)', 't_1M000601006_수정EPS(보통주)(원)']]

    # var_name = '상장주식수_보통주'
    cd_list = list(df['Symbol'].unique())
    check_var_name = ['M000601002_EPS(원)', 'M000601005_EPS(보통주)(원)', 'M000601003_수정EPS(원)', 'M000601006_수정EPS(보통주)(원)']
    for var_name in check_var_name:
        df["t_4"+var_name] = ""

    for cd in cd_list:
        for year in range(2012, 2019):
            for quarter in range(1, 5):
                tminus_quarter = quarter - 3  # 결산월 보다 쿼터 쪽이 신뢰도 있음
                quarter = str(quarter) + 'Q'
                tminus_year = year
                if tminus_quarter == -2:  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
                    tminus_year = year - 1  # 결산월 보다 쿼터 쪽이 신뢰도 있음
                    tminus_quarter = '2Q'
                elif tminus_quarter == -1:  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
                    tminus_year = year - 1  # 결산월 보다 쿼터 쪽이 신뢰도 있음
                    tminus_quarter = '3Q'
                elif tminus_quarter == 0:  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
                    tminus_year = year - 1  # 결산월 보다 쿼터 쪽이 신뢰도 있음
                    tminus_quarter = '4Q'
                else:
                    tminus_quarter = str(tminus_quarter) + 'Q'

                tminus_data = df[(df['Symbol'] == cd) & (df['회계년'] == tminus_year) & (df['주기'] == tminus_quarter)]
                t_data = df[(df['Symbol'] == cd) & (df['회계년'] == year) & (df['주기'] == quarter)]
                for var_name in check_var_name:
                    new_var_name1 = "t_4"+var_name
                    new_var_name2 = "t_1"+var_name
                    df.loc[t_data.index[0], new_var_name1] = float(tminus_data.iloc[0][new_var_name2])
    df.to_csv('change_check.csv', encoding='cp949')
    # df.drop(columns=['t_1M000601002_EPS(원)', 't_1M000601005_EPS(보통주)(원)', 't_1M000601003_수정EPS(원)', 't_1M000601006_수정EPS(보통주)(원)'], inplace=True)
    # df.drop(columns=['t_4M000601002_EPS(원)', 't_4M000601005_EPS(보통주)(원)', 't_4M000601003_수정EPS(원)', 't_4M000601006_수정EPS(보통주)(원)'], inplace=True)
    # df.columns = ['Symbol', 'Name', '결산월', '회계년', '주기', 't_4M000601002_EPS(원)', 't_4M000601005_EPS(보통주)(원)', 't_4M000601003_수정EPS(원)', 't_4M000601006_수정EPS(보통주)(원)']
    # merged_df = pd.merge(left=t_minus_n, right=df, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    # merged_df.to_csv('change_check.csv', encoding='cp949')


def parse_rptnm(qual_ind_var):
    for index, row in qual_ind_var.iterrows():  # 무식하지만 결국 이 틀에서 벗어나기 어렵다.
        rpt_nm = row['rpt_nm']
        t_closing_date = rpt_nm[rpt_nm.find("(") + 1:rpt_nm.find(")")].split('.')
        t_year = int(t_closing_date[0])
        t_month = int(t_closing_date[1])

        rpt = rpt_nm.split()
        if rpt[0] == '반기보고서' and t_month == 6:
            # t_quarter = '2Q'
            t_quarter = 2
        elif rpt[0] == '사업보고서' and t_month == 12:
            # t_quarter = '4Q'
            t_quarter = 4
        elif rpt[0] == '분기보고서' and t_month == 3:  # 주기와 맞는다는 보장은 없다. 이게 맞길 바래야함.
            # t_quarter = '1Q'
            t_quarter = 1
        elif rpt[0] == '분기보고서' and t_month == 9:
            # t_quarter = '3Q'
            t_quarter = 3
        else:  # 혹시 모르니 일단 예외처리.  # 직접 확인한 결과 612건. 예를들어 4월부터 9월까지를 반기로 치는 중소기업이 있었다(000220). 무시해도 좋다고 판단함.
            print('exeception idx:', index, ' month:', t_month, ' rpt_nm:', rpt_nm)
            t_quarter = ''
            continue
        qual_ind_var.loc[index, '주기'] = t_quarter
        qual_ind_var.loc[index, '회계년'] = t_year
        qual_ind_var.loc[index, 'Symbol'] = 'A' + str(row['crp_cd'])
    return qual_ind_var


def match_quanti_and_qual_data(qual_ind_var, quanti_ind_var, file_name):  # 정량 독립 및 정량 종속 변수와 정성 독립 변수 매칭
    result_df = pd.DataFrame()
    valid_df_idx_list = []

    pool = Pool(8)
    df_split = np.array_split(qual_ind_var, 8)
    qual_ind_var = pd.concat(pool.map(parse_rptnm, df_split))
    pool.close()
    pool.join()

    matched_quanti_and_qual_data = pd.merge(left=quanti_ind_var, right=qual_ind_var, how='left', on=['Symbol', '회계년', '주기'], sort=False)

    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    matched_quanti_and_qual_data.to_pickle(directory_name+'/'+file_name)
    print(len(valid_df_idx_list))
    print(matched_quanti_and_qual_data.shape)
    return matched_quanti_and_qual_data


def add_one_hot_with_ind_cd(df):
    sector_detailed = pd.read_excel('한국표준산업분류(10차)_표.xlsx', sheet_name='Sheet2', dtype=object)
    # print(sector_detailed.info())
    # df = matched_quanti_and_qual_data.copy()  # FOR TEST
    df['ind_cd'] = df['ind_cd'].replace(np.nan, "")

    for index, row in sector_detailed.iterrows():
        print(row['sector2'])
        tmp = df[df.ind_cd.str.contains('^' + str(row['range']))]  # 앞의 두자리만 알면 중분류를 알 수 있다.
        if tmp.empty:
            continue
        print(tmp.shape)
        for idx, r in tmp.iterrows():
            # df.loc[idx, 'ind'] = row['sector']  # 대분류
            df.loc[idx, 'ind'] = row['sector2']  # 중분류
        # try:
        # except Exception as e:
        #     print(e)
    df = add_one_hot(df, 'ind')
    df.drop(['ind_cd'], axis=1, inplace=True)
    return df


def per_crp_rolling(tmp):
    identifier1 = ['Symbol',
                    '회계년',
                    '주기',
                    't4_M000901012_재고자산(천원)',
                    't4_M000901006_매출채권(천원)',
                    't4_M_CAPEX',
                    't4_M000904007_매출총이익(천원)',
                    't4_M000904017_판매비와관리비(천원)',
                    # 't4_ETR',
                    't4_M000911020_유효세율(%)',
                    't4_LF_salesDivEmp',
                    't4_t_1M000601002_EPS(원)',
                    't4_t_1M000601005_EPS(보통주)(원)',
                    't4_t_1M000601003_수정EPS(원)',
                    't4_t_1M000601006_수정EPS(보통주)(원)',
                    't4_t_1M000908001_당기순이익(천원)',
                    't3_M000901012_재고자산(천원)',
                    't3_M000901006_매출채권(천원)',
                    't3_M_CAPEX',
                    't3_M000904007_매출총이익(천원)',
                    't3_M000904017_판매비와관리비(천원)',
                    # 't3_ETR',
                    't3_M000911020_유효세율(%)',
                    't3_LF_salesDivEmp',
                    't3_t_1M000601002_EPS(원)',
                    't3_t_1M000601005_EPS(보통주)(원)',
                    't3_t_1M000601003_수정EPS(원)',
                    't3_t_1M000601006_수정EPS(보통주)(원)',
                    't3_t_1M000908001_당기순이익(천원)',
                    't2_M000901012_재고자산(천원)',
                    't2_M000901006_매출채권(천원)',
                    't2_M_CAPEX',
                    't2_M000904007_매출총이익(천원)',
                    't2_M000904017_판매비와관리비(천원)',
                    # 't2_ETR',
                    't2_M000911020_유효세율(%)',
                    't2_LF_salesDivEmp',
                    't2_t_1M000601002_EPS(원)',
                    't2_t_1M000601005_EPS(보통주)(원)',
                    't2_t_1M000601003_수정EPS(원)',
                    't2_t_1M000601006_수정EPS(보통주)(원)',
                    't2_t_1M000908001_당기순이익(천원)',
                    't1_M000901012_재고자산(천원)',
                    't1_M000901006_매출채권(천원)',
                    't1_M_CAPEX',
                    't1_M000904007_매출총이익(천원)',
                    't1_M000904017_판매비와관리비(천원)',
                    # 't1_ETR',
                    't1_M000911020_유효세율(%)',
                    't1_LF_salesDivEmp',
                    't1_t_1M000601002_EPS(원)',
                    't1_t_1M000601005_EPS(보통주)(원)',
                    't1_t_1M000601003_수정EPS(원)',
                    't1_t_1M000601006_수정EPS(보통주)(원)',
                    't1_t_1M000908001_당기순이익(천원)']
    identifier2 = ['Symbol',
                   '회계년',
                   '주기',
                   'M000901012_재고자산(천원)',
                   'M000901006_매출채권(천원)',
                   'M_CAPEX',
                   'M000904007_매출총이익(천원)',
                   'M000904017_판매비와관리비(천원)',
                   # 'ETR',
                   'M000911020_유효세율(%)',
                   'LF_salesDivEmp',
                   't_1M000601002_EPS(원)',
                   't_1M000601005_EPS(보통주)(원)',
                   't_1M000601003_수정EPS(원)',
                   't_1M000601006_수정EPS(보통주)(원)',
                   't_1M000908001_당기순이익(천원)']
    ind_var_list_for_rolling = ['M000901012_재고자산(천원)',
                                'M000901006_매출채권(천원)',
                                'M_CAPEX',
                                'M000904007_매출총이익(천원)',
                                'M000904017_판매비와관리비(천원)',
                                # 'ETR',
                                'M000911020_유효세율(%)',
                                'LF_salesDivEmp',
                                't_1M000601002_EPS(원)',
                                't_1M000601005_EPS(보통주)(원)',
                                't_1M000601003_수정EPS(원)',
                                't_1M000601006_수정EPS(보통주)(원)',
                                't_1M000908001_당기순이익(천원)']
    result_df = pd.DataFrame(columns=identifier1)
    for idx in range(3, tmp.shape[0]):  # 3 ~ 4*8(2011, 2012,2013,2014, 2015,2016,2017,2018개 년)
        # idx = 3  # for test
        tmp4 = tmp.loc[idx - 3:idx, :].copy()
        tmp4 = tmp4.loc[:, identifier2]
        tmp4.reset_index(drop=True, inplace=True)
        # print(list(tmp4.columns))
        # print(tmp4.shape)
        # print(tmp4.at[3,'Symbol'])
        ident = [tmp4.iloc[3]['Symbol'], tmp4.iloc[3]['회계년'], tmp4.iloc[3]['주기']]
        flatten = list(tmp4[ind_var_list_for_rolling].values.flatten())  # 36 = 4 * 9
        ident.extend(flatten)  # 39 = 3 + 36
        new_row = pd.DataFrame(columns=identifier1, data=[ident])
        result_df = result_df.append(new_row)
    return result_df


def rolling_t4(df, ind_var_list):
    cd_list = list(df['Symbol'].unique())
    identifier1 = ['Symbol', '회계년', '주기']
    identifier2 = identifier1.copy()
    cols = ind_var_list.copy()
    new_cols = []
    for t in ['t4_', 't3_', 't2_', 't1_']:
        new_cols.extend([t+col for col in cols])  # 9*4 = 36
    identifier1.extend(new_cols)  # 39 = 3+36
    # identifier1.append(dep)  # 40 = 39+40
    identifier2.extend(ind_var_list)  # 12 = 3 + 9
    # identifier2.append(dep)  # 13 = 3 + 9 +1

    # result_df = pd.DataFrame(columns=identifier1)
    tmp_list = []
    pool = Pool(8)

    for cd in cd_list:
        # cd = 'A000020'  # FOR TEST
        tmp = df[(df['Symbol'] == cd)].copy()
        tmp.reset_index(drop=True, inplace=True)
        tmp_list.append(tmp)
    print(len(tmp_list))
    del df
    df_li = pool.map(per_crp_rolling, tmp_list)
    result_df = pd.concat(df_li, axis=0, ignore_index=True, sort=False)
    result_df.reset_index(drop=True, inplace=True)
    pool.close()
    pool.join()
    return result_df


def mean_per_n(result_list, n):
    result = []
    average_per_n = []
    for li in result_list:
        result.append(li)
        if len(result) == n:
            average_per_n.append(np.mean(result))
            result = []
    return average_per_n


def mean_absolute_percentage_error(y_true, y_pred):
    # y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100
    # return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Mpe(y_true, y_pred): # underperformance 인지 overperformance 인지 판단 할 수 있다는 것입니다.
    return np.mean((y_true - y_pred) / y_true) * 100


def adjusted_rmse(y_true, y_pred):  # 사실 2009년엔 없는 단어여서 그런지 모르겠지만 이걸 Mean Square Percentage Error 라고 함. mape보다 인기는 없어보인다.
    # return np.sqrt((((targets - predictions)/targets) ** 2).mean())
    return math.sqrt(np.square(np.divide(np.subtract(y_true, y_pred), y_true)).mean())


def ols(df, x_var_names, y_var_name, constant):
    testRmseList = []
    testmapeList = []
    testadjrmseList = []
    predict_df_li = []
    for year_q in [(2016, 1), (2016, 2), (2016, 3), (2016, 4),
                 (2017, 1), (2017, 2), (2017, 3), (2017, 4),
                 (2018, 1), (2018, 2), (2018, 3)]:  # 일종의 3-fold가 된다.
        # df = data2.copy()
        # year_q = (2016,1)  # for test
        # x_var_names = ind_var_at_paper
        # y_var_name = dep
        print('------------------------ year: ', year_q)
        test = df[(df['회계년'] == year_q[0]) & (df['주기'] == year_q[1])]
        train = df[df.index < test.index[0]]
        print('train ', train.shape)
        print('test ', test.shape)
        predict_df = test.iloc[:, :3]
        print(predict_df.shape)
        train_X = train[x_var_names]
        train_y = train[y_var_name]
        if constant == True:
            train_X = sm.add_constant(train_X)  ###########  constant
        result = sm.OLS(train_y, train_X).fit()
        # print('r_square : ', result.rsquared)
        print(result.summary())
        predict_df.reset_index(drop=True, inplace=True)
        # print(result.predict(X))
        test_X = test[x_var_names]
        test_y = test[y_var_name]
        if constant == True:
            test_X = sm.add_constant(test_X)  ###########  constant
        predictY = result.predict(test_X)
        rmse = sqrt(mean_squared_error(test_y, predictY))
        print('rmse: ', rmse)
        adjrmse = adjusted_rmse(test_y, predictY)
        print('adjusted_rmse: ', rmse)
        mape = Mape(test_y, predictY)
        print('mape: ', mape)
        mpe = MPE(test.iloc[:, -1], predictY)
        if mpe < 0:
            print(mpe, 'over perform')
        else:
            print(mpe, 'under perform')
        testadjrmseList.append(adjrmse)
        testRmseList.append(rmse)
        testmapeList.append(mape)
        # print(len(list(test_y)))
        # print(len(predictY))
        print(predict_df.shape)
        predict_df['true'] = pd.Series(list(test_y))
        predict_df['pred'] = pd.Series(list(predictY))
        predict_df_li.append(predict_df)
    # f = open("OLS_result"+" ".join(x_var_names)+".txt", 'w')
    # f.write(str(result.summary()))
    # f.close()
    predict_df = pd.concat(predict_df_li, axis=0, ignore_index=True, sort=False)
    return testRmseList, testadjrmseList, testmapeList, predict_df


def q_replace_int(df):
    df.loc[df['주기'] == '1Q', '주기'] = 1
    df.loc[df['주기'] == '2Q', '주기'] = 2
    df.loc[df['주기'] == '3Q', '주기'] = 3
    df.loc[df['주기'] == '4Q', '주기'] = 4
    return df


def ols_all(data, x_var_names, y_var_name, constant):
    # year= 2016  # for test
    print('------------------------\nols ALL: ')
    predict_df = data.iloc[:, :3]

    X = data[x_var_names]
    y = data[y_var_name]
    if constant == True:
        X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    # print('r_square : ', result.rsquared)
    print(result.summary())

    # print(result.predict(X))
    predictY = result.predict(X)
    rmse = sqrt(mean_squared_error(y, predictY))
    print('rmse: ', rmse)
    adjrmse = adjusted_rmse(y, predictY)
    print('adjusted_rmse: ', adjrmse)
    mape = Mape(y, predictY)
    print('mape: ', mape)
    mpe = MPE(y, predictY)
    if mpe < 0:
        print(mpe, 'over perform')
    else:
        print(mpe, 'under perform')
    predict_df['true'] = pd.Series(list(y))
    predict_df['pred'] = pd.Series(list(predictY))

    # f = open("OLS_result"+" ".join(x_var_names)+".txt", 'w')
    # f.write(str(result.summary()))
    # f.close()
    return rmse, adjrmse, mape, predict_df


def match_tplus_data_modif(quanti_ind_var, dep_vars, dep_var, ind_var_list, file_name):
    identifier = ['Symbol', '회계년', '주기']
    dep_vars = pd.concat([dep_vars.loc[:, identifier],
                         dep_vars.loc[:, dep_var]], axis=1)  # 일단 폼은 유지하되 필요한 변수 하나만 떼다 쓰기 위함.
    # dep_vars.dropna(thresh=len(list(dep_vars.columns)) - len(identifier), inplace=True)  # 식별 위한 정보를 제외한 것이 하나도 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # identifier.extend(ind_var_list)
    quanti_ind_var = quanti_ind_var[ind_var_list].copy()
    print(quanti_ind_var.shape)
    # quanti_ind_var = quanti_ind_var.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까.
    # quanti_ind_var.dropna(thresh=len(list(dep_vars.columns)) - len(identifier), inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(quanti_ind_var.columns)
    # print('quanti_ind_var.info() : ', quanti_ind_var.dtypes)
    for index, row in quanti_ind_var.iterrows():
        # print(row)
        tplus_quarter = row['주기']+1   # 결산월 보다 쿼터 쪽이 신뢰도 있음
        if tplus_quarter == 5:  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
            tplus_year = int(row['회계년']) + 1   # 결산월 보다 쿼터 쪽이 신뢰도 있음
            tplus_quarter = 1
        else:
            tplus_year = int(row['회계년'])
        tplus_data = dep_vars[(dep_vars['Symbol'] == row['Symbol'])  # t+1 종속변수를 끌어온다.
                              & (dep_vars['회계년'] == tplus_year)
                              & (dep_vars['주기'] == tplus_quarter)].copy()
        if tplus_data.shape[0] > 1:  # 중복된 분기의 데이터 없는 문제 확인.
            print('tplus_data :', tplus_data)  # 일단 미리 없애놨으니 나타날린 없지만 그래도 체크.
        if tplus_data.empty:  # 없으면 체크
            quanti_ind_var.loc[index, dep_var] = np.nan
            # print('empty', index)
            # print(row)
            continue
        for dep in dep_var:
            quanti_ind_var.loc[index, dep] = tplus_data.iloc[0][dep]  # t+1 분기와 매칭. 이후 독립변수 데이터 프레임 옆에다 종속변수를 붙인다.
    # result_df = result_df.drop(columns=['Symbol', 'Name', '결산월', '회계년'])  # 종속변수 쪽 식별 정보는 필요 없음.
    # result_df.reset_index(drop=True, inplace=True)
    # quanti_ind_var.reset_index(drop=True, inplace=True)
    # quanti_ind_var = quanti_ind_var.drop(columns=['Name'])  # 그냥 종목번호보다 보기좋아서 냅둔거라. 지워도 상관없음.

    # quanti_ind_var = pd.concat([quanti_ind_var, result_df], axis=1)
    # quanti_ind_var.dropna(inplace=True)  # 이전 단계에서 걸렀을 가능성이 높지만 그래도 1~2개 없는 경우를 거르기 위함.
    # directory_name = 'merged_FnGuide ind_var/'
    # if not os.path.exists(directory_name):
    #     os.mkdir(directory_name)
    # df1.to_excel(directory_name+'/merged_FnGuide ind_var '+keyword+'.xlsx')
    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    # np.save('./merged_FnGuide/'+file_name, quanti_ind_var.values)
    quanti_ind_var.to_excel(directory_name+'/'+file_name, encoding='cp949')
    print(quanti_ind_var.columns)
    print(quanti_ind_var.shape)
    print(quanti_ind_var.info())
    return quanti_ind_var


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    # path_dir = 'C:/Users/lab515/PycharmProjects/eps_predict/'
    path_dir = os.getcwd()
    ind_var_list = [
                    # 'M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)',
                    # 'M000904017_판매비와관리비(천원)', 'M000911020_유효세율(%)', 'LF_salesDivEmp'  # 기존 EPS 계산시 사용된 수치들.
                    # 'M000909001_영업활동으로인한현금흐름(천원)'  # 황선희 2006 연구
                    'CFO_ratio'  # 황선희 2006 연구
                    , 'tminus_inv_M000901001_총자산(천원)', 'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)'   # Dechow 1995  # 16개 변수
                    , 'BIG', 'SIZE', 'LEV', 'ROA', 'LOSS', 'MTB', 'LARGE', 'FOREIGN'  # SIZE, LEV, MTB, ROA, LOSS, CFO, BIG, LARGE, Foreign,      YD, ID, e  # 모예린 2018  # CFO는 이미 포함. 근데 영 유의하지 않아서 빼는걸 고려.
                    ]  # 여기에 사용할 eps를 더 추가
    version_number = 1
    # constant = True
    constant = False  # 모예린 있을 때만
    dep_var = ['dep_M000601002_EPS(원)', 'dep_M000601005_EPS(보통주)(원)',
               'dep_M000601003_수정EPS(원)', 'dep_M000601006_수정EPS(보통주)(원)']  # 쓸걸로 예상
    eps_versions =[['t_1M000601002_EPS(원)', 't_4M000601002_EPS(원)'],
                   ['t_1M000601005_EPS(보통주)(원)', 't_4M000601005_EPS(보통주)(원)'],
                   ['t_1M000601003_수정EPS(원)', 't_4M000601003_수정EPS(원)'],
                   ['t_1M000601006_수정EPS(보통주)(원)', 't_4M000601006_수정EPS(보통주)(원)']]    ## option ##
    dep = dep_var[version_number]
    eps_version = eps_versions[version_number]
    main_ind_var = 't_1q_cos_dist'
    # main_ind_var = 't_1y_cos_dist'
    # main_ind_var = 'diff_q_cos_per_ind'  # 산업 평균 대비 분기 cos거리
    # main_ind_var = 'diff_y_cos_per_ind'  # 산업 평균 대비 연 cos거리
    ## option ##
    ind_var_list_for_rolling = ind_var_list.copy()
    # flat_list = [item for sublist in eps_versions for item in sublist]
    ind_var_list_for_rolling.extend(eps_versions)
    # ind_var_list_for_rolling = eps_versions
    # ind_var_list.append(eps_version)  # 9개 변수
    # quanti_qual_matched_file_name = 'quanti_qaul_komoran_dnn.pkl'  # 정성 데이터 포함시킨 데이터.
    # quanti_qual_matched_file_name = 'revision_quanti_qaul_komoran_dnn.pkl'  # 정성 데이터 포함시킨 데이터.
    quanti_qual_matched_file_name = 'revisionAll_quanti_qaul_komoran_dnn.pkl'  # 정성 데이터 포함시킨 데이터.
    """           
    df8 = pd.read_pickle(path_dir + '/divide_by_sector/filter8 komoran_for_cosine_distance.pkl')  # 독립변수중 산업이나 코사인거리
    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\ind_wei_revision.xlsx'
                                   , dtype=object, sheet_name='ind_wei_revision')  # 미리 종속변수까지 붙여놓아 번거로운 작업을 할 필요는 없음.
    d = quanti_ind_var.describe()
    matched_quanti_and_qual_data = match_quanti_and_qual_data(df8, quanti_ind_var, 'processing_'+quanti_qual_matched_file_name)
    # matched_quanti_and_qual_data.dropna(how='all', inplace=True)
    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/processing_' + quanti_qual_matched_file_name)
    """
    """        
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/processing_'+quanti_qual_matched_file_name)

    matched_quanti_and_qual_data['회계년'] = matched_quanti_and_qual_data['회계년'].astype('int')
    matched_quanti_and_qual_data = matched_quanti_and_qual_data[(matched_quanti_and_qual_data['회계년']) < 2019]

    ## 산업코드 부여 안된 빈 공간 미리 처리.
    crp_ind_match = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\crp_ind_match.xlsx'
                                   , dtype=object, sheet_name='Sheet1')  # 미리 종속변수까지 붙여놓아 번거로운 작업을 할 필요는 없음.
    cd_list = list(matched_quanti_and_qual_data['Symbol'].unique())
    for cd in cd_list:
        # cd = 'A005930'  # for test
        try:
            matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data['Symbol'] == cd), 'ind_cd'] = crp_ind_match.loc[crp_ind_match['Symbol'] == cd, 'ind_cd'].values[0]
        except Exception as e:
            # print(cd)
            # print(e)
            pass

    columns = ['crp_cd', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt', 't_minus_index', 't_minus_year_index',
               'Name', '결산월']  # Symbol, 회계년, 주기, ind_cd는 차후 식별등을 위해 일단 남긴다.  # 'ind','crp_cls'나중에 지운다.
    matched_quanti_and_qual_data.drop(columns, inplace=True, axis=1)

    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].replace("", np.nan)
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')
    #
    # matched_quanti_and_qual_data['t_1y_cos_dist'] = matched_quanti_and_qual_data['t_1y_cos_dist'].replace("", np.nan)
    # matched_quanti_and_qual_data['t_1y_cos_dist'] = matched_quanti_and_qual_data['t_1y_cos_dist'].astype('float')

    matched_quanti_and_qual_data = add_one_hot(matched_quanti_and_qual_data, '주기')
    matched_quanti_and_qual_data = add_one_hot(matched_quanti_and_qual_data, 'crp_cls')
    matched_quanti_and_qual_data = add_one_hot_with_ind_cd(matched_quanti_and_qual_data)  # 중분류 산업코드 사용. 대분류 제조업은

    # ind_cosine_dict = {}
    # for ind_col_name in list(matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^ind')]):
    #     print(ind_col_name)
    #     if ind_col_name == 'ind':
    #         continue
    #     for year in range(2013, 2019):
    #         for quarter in ['주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q']:
    #             avg_q_dist = matched_quanti_and_qual_data[(matched_quanti_and_qual_data['회계년'] == year) & (matched_quanti_and_qual_data[ind_col_name] == 1) & (matched_quanti_and_qual_data[quarter] == 1)]['t_1q_cos_dist'].mean(skipna=True)
    #             avg_y_dist = matched_quanti_and_qual_data[(matched_quanti_and_qual_data['회계년'] == year) & (matched_quanti_and_qual_data[ind_col_name] == 1) & (matched_quanti_and_qual_data[quarter] == 1)]['t_1y_cos_dist'].mean(skipna=True)
    #             ind_cosine_dict[ind_col_name+'$'+str(year)+'$'+quarter] = [avg_q_dist, avg_y_dist]
    # matched_quanti_and_qual_data.drop(['q_cos_mean', 'y_cos_mean', 'diff_q_cos_per_ind', 'diff_y_cos_per_ind'], inplace=True, axis=1)
    # cols = matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^q_cos_ind')]
    # matched_quanti_and_qual_data.drop(cols, inplace=True, axis=1)
    # cols = matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^y_cos_ind')]
    # matched_quanti_and_qual_data.drop(cols, inplace=True, axis=1)

    # for key in ind_cosine_dict:
    #     # key = 'ind_항공 운송업'  # for test
    #     matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data['회계년'] == int(key.split('$')[1])) & (matched_quanti_and_qual_data[key.split('$')[0]] == 1) & (matched_quanti_and_qual_data[key.split('$')[2]] == 1), "q_cos_mean"] = ind_cosine_dict[key][0]
    #     matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data['회계년'] == int(key.split('$')[1])) & (matched_quanti_and_qual_data[key.split('$')[0]] == 1) & (matched_quanti_and_qual_data[key.split('$')[2]] == 1), "y_cos_mean"] = ind_cosine_dict[key][1]
    # matched_quanti_and_qual_data.dropna(inplace=True)
    # matched_quanti_and_qual_data['diff_q_cos_per_ind'] = matched_quanti_and_qual_data['t_1q_cos_dist'] - matched_quanti_and_qual_data['q_cos_mean']
    # matched_quanti_and_qual_data['diff_y_cos_per_ind'] = matched_quanti_and_qual_data['t_1y_cos_dist'] - matched_quanti_and_qual_data['y_cos_mean']

    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/diffPerInd_ols' + quanti_qual_matched_file_name)
    """
    """     
    # matched_quanti_and_qual_data.drop(['t_1y_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data.drop(['t_1q_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/diffPerInd_ols'+quanti_qual_matched_file_name)
    # print(matched_quanti_and_qual_data.dtypes)
    # matched_quanti_and_qual_data.sort_values(['Symbol', '회계년', '주기'], ascending=['True', 'True', 'True'], inplace=True)
    matched_quanti_and_qual_data['LF_salesDivEmp'] = matched_quanti_and_qual_data['LF_salesDivEmp'].astype('float')
    matched_quanti_and_qual_data['LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['LF_salesDivEmp'])

    # etr_calc = pd.read_excel(path_dir+'/financial ratio for independent variable/ETR_calc_table.xlsx', sheet_name='Sheet1')  # 3번 조건. 2번은 이미 충족됨.
    # matched_quanti_and_qual_data = pd.merge(left=matched_quanti_and_qual_data, right=etr_calc[['Symbol', '회계년', '주기', 'ETR', 'ETR_treat_minus']],
    #                                         how='left', on=['Symbol', '회계년', '주기'], sort=False)
    matched_quanti_and_qual_data.replace(to_replace='N/A(IFRS)', value=np.nan, inplace=True)
    matched_quanti_and_qual_data_t4Flatten = rolling_t4(matched_quanti_and_qual_data, ind_var_list_for_rolling)
    print(matched_quanti_and_qual_data.shape[0])
    print(matched_quanti_and_qual_data_t4Flatten.shape[0])
    # matched_quanti_and_qual_data_t4Flatten.reset_index(drop=True, inplace=True)
    # result_df.dropna(inplace=True)
    """
    """       
    columns = ['Symbol', '회계년', '주기', '주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q'
               # 'M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)',
               # 'M000904017_판매비와관리비(천원)', 'M000911020_유효세율(%)', 'LF_salesDivEmp'  # 기존 EPS 계산시 사용된 수치들.
                , 'M000909001_영업활동으로인한현금흐름(천원)'  # 황선희 2006 연구
                , 'tminus_inv_M000901001_총자산(천원)', 'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)'
                , 'BIG', 'SIZE', 'LEV', 'ROA', 'LOSS', 'MTB', 'LARGE', 'FOREIGN',
                'dep_M000601002_EPS(원)', 'dep_M000601005_EPS(보통주)(원)',
               'dep_M000601003_수정EPS(원)', 'dep_M000601006_수정EPS(보통주)(원)',
               'dep_M000908001_당기순이익(천원)',
               't_1q_cos_dist', 't_1y_cos_dist', # 'diff_q_cos_per_ind', 'diff_y_cos_per_ind',
               'crp_cls_K', 'crp_cls_Y', 'ind',  # 식별용으로 남김.
               'ind_1차 금속 제조업', 'ind_가구 제조업',
               'ind_가죽, 가방 및 신발 제조업', 'ind_건축 기술, 엔지니어링 및 기타 과학기술 서비스업',
               'ind_고무 및 플라스틱제품 제조업', 'ind_교육 서비스업', 'ind_금속 가공제품 제조업; 기계 및 가구 제외',
               'ind_금융업', 'ind_기타 개인 서비스업', 'ind_기타 기계 및 장비 제조업', 'ind_기타 운송장비 제조업',
               'ind_기타 전문, 과학 및 기술 서비스업', 'ind_기타 제품 제조업', 'ind_농업', 'ind_담배 제조업',
               'ind_도매 및 상품 중개업', 'ind_목재 및 나무제품 제조업; 가구 제외', 'ind_방송업',
               'ind_보험 및 연금업', 'ind_부동산업', 'ind_비금속 광물제품 제조업', 'ind_비금속광물 광업; 연료용 제외',
               'ind_사업 지원 서비스업', 'ind_사업시설 관리 및 조경 서비스업', 'ind_석탄, 원유 및 천연가스 광업',
               'ind_섬유제품 제조업; 의복 제외', 'ind_소매업; 자동차 제외', 'ind_수상 운송업', 'ind_숙박업',
               'ind_스포츠 및 오락관련 서비스업', 'ind_식료품 제조업', 'ind_어업', 'ind_연구개발업',
               'ind_영상ㆍ오디오 기록물 제작 및 배급업', 'ind_우편 및 통신업', 'ind_육상 운송 및 파이프라인 운송업',
               'ind_음료 제조업', 'ind_음식점 및 주점업', 'ind_의료, 정밀, 광학 기기 및 시계 제조업',
               'ind_의료용 물질 및 의약품 제조업', 'ind_의복, 의복 액세서리 및 모피제품 제조업',
               'ind_인쇄 및 기록매체 복제업', 'ind_임대업; 부동산 제외', 'ind_자동차 및 부품 판매업',
               'ind_자동차 및 트레일러 제조업', 'ind_전기, 가스, 증기 및 공기 조절 공급업', 'ind_전기장비 제조업',
               'ind_전문 서비스업', 'ind_전문직별 공사업', 'ind_전자 부품, 컴퓨터, 영상, 음향 및 통신장비 제조업',
               'ind_정보서비스업', 'ind_종합 건설업', 'ind_창고 및 운송관련 서비스업',
               'ind_창작, 예술 및 여가관련 서비스업', 'ind_출판업', 'ind_컴퓨터 프로그래밍, 시스템 통합 및 관리업',
               'ind_코크스, 연탄 및 석유정제품 제조업', 'ind_펄프, 종이 및 종이제품 제조업',
               'ind_폐기물 수집, 운반, 처리 및 원료 재생업', 'ind_항공 운송업',
               'ind_화학 물질 및 화학제품 제조업; 의약품 제외', 'ind_환경 정화 및 복원업']  # Symbol, 회계년, 주기, ind_cd는 차후 식별등을 위해 일단 남긴다.  # 'ind','crp_cls'나중에 지운다.

    matched_quanti_and_qual_data_t4Flatten = q_replace_int(matched_quanti_and_qual_data_t4Flatten)
    matched_quanti_and_qual_data = q_replace_int(matched_quanti_and_qual_data)
    merged_df = pd.merge(left=matched_quanti_and_qual_data_t4Flatten, right=matched_quanti_and_qual_data[columns],
                         how='left', on=['Symbol', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    merged_df = merged_df[(merged_df['회계년']) > 2011]
    matched_quanti_and_qual_data = merged_df.copy()
    # matched_quanti_and_qual_data_t4Flatten.drop(['dep_M000601002_EPS(원)'], inplace=True, axis=1)
    # matched_quanti_and_qual_data_m = pd.read_pickle('./merged_FnGuide/processing_'+quanti_qual_matched_file_name)
    # matched_quanti_and_qual_data_m['회계년'] = matched_quanti_and_qual_data_m['회계년'].astype('int')
    # matched_quanti_and_qual_data_m = matched_quanti_and_qual_data_m[(matched_quanti_and_qual_data_m['회계년']) < 2019]

    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['t1_LF_salesDivEmp'] <= 0, 't1_LF_salesDivEmp'] = np.nan
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['t2_LF_salesDivEmp'] <= 0, 't2_LF_salesDivEmp'] = np.nan
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['t3_LF_salesDivEmp'] <= 0, 't3_LF_salesDivEmp'] = np.nan
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['t4_LF_salesDivEmp'] <= 0, 't4_LF_salesDivEmp'] = np.nan

    # matched_quanti_and_qual_data['t1_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t1_LF_salesDivEmp'])
    # matched_quanti_and_qual_data['t2_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t2_LF_salesDivEmp'])
    # matched_quanti_and_qual_data['t3_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t3_LF_salesDivEmp'])
    # matched_quanti_and_qual_data['t4_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t4_LF_salesDivEmp'])

    common_share_cnt = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , sheet_name='Sheet1')
    common_share_cnt['상장주식수_보통주'] = common_share_cnt['상장주식수_보통주'].astype('float')
    common_share_cnt = q_replace_int(common_share_cnt)

    matched_quanti_and_qual_data = pd.merge(left=matched_quanti_and_qual_data, right=common_share_cnt[['Symbol', '회계년', '주기', '상장주식수_보통주']],
                                            how='left', on=['Symbol', '회계년', '주기'], sort=False)
    for i in range(1, 5):
        for ind_var in ['M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)', 'M000904017_판매비와관리비(천원)']:
            div_var_name = 't' + str(i) + '_' + ind_var
            matched_quanti_and_qual_data[div_var_name] = matched_quanti_and_qual_data[div_var_name] / matched_quanti_and_qual_data['상장주식수_보통주']
            
    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/t4_ols'+quanti_qual_matched_file_name)
    """
    """          """
    matched_quanti_and_qual_data = pd.read_pickle(path_dir + '/merged_FnGuide/t4_ols' + quanti_qual_matched_file_name)

    ## 13~18년 합병 이벤트 있는 기업 제외. 시작
    merge_log = pd.read_excel(path_dir+'/합병_내역.xlsx', dtype=object, sheet_name='crp_cd_merge_happend')  # 3번 조건. 2번은 이미 충족됨.
    merge_crp = ['A'+str(x) for x in list(merge_log['거래소코드'].unique())]
    matched_quanti_and_qual_data = matched_quanti_and_qual_data.loc[~matched_quanti_and_qual_data['Symbol'].isin(merge_crp)]
    ## 13~18년 합병 이벤트 있는 기업 제외. 끝

    # matched_quanti_and_qual_data_without_merge[dep].replace('N/A(IFRS)', np.nan, inplace=True)
    # matched_quanti_and_qual_data_without_merge[dep] = matched_quanti_and_qual_data_without_merge[dep].astype('float')

    ## eps 0 이상인 기업만 포함. 시작
    # matched_quanti_and_qual_data_without_minus_eps = matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data[dep] > 0, :]  # 1번 조건
    ## eps 0 이상인 기업만 포함. 끝

    ## 유상증자가 없는 기업만 포함 시작.
    # matched_quanti_and_qual_data_without_cnt_change = matched_quanti_and_qual_data_without_minus_eps[matched_quanti_and_qual_data_without_minus_eps['cnt_change'] != 'diff']  # 4번 조건
    ## 유상증자가 없는 기업만 포함 끝

    # idx_until_diff_list = []
    # crp_cd_li = list(matched_quanti_and_qual_data_without_minus_eps['Symbol'].unique())
    # for crp_cd in crp_cd_li:
    #     print(crp_cd)
    #     tmp = matched_quanti_and_qual_data_without_minus_eps[matched_quanti_and_qual_data_without_minus_eps['Symbol'] == crp_cd]
    #     tmp.sort_values(['회계년', 'rpt_nm', 'rcp_dt'], ascending=['True', 'True', 'True'], inplace=True)
    #
    #     for index, row in tmp.iterrows():
    #         if row['cnt_change'] != 'diff':
    #             idx_until_diff_list.append(index)
    #         else:
    #             break
    # matched_quanti_and_qual_data_without_cnt_change = matched_quanti_and_qual_data_without_minus_eps.loc[idx_until_diff_list, :]

    # no_diff_crp = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
    #                                , dtype=object, sheet_name='no_diff_crp')
    # no_diff_crp_list = list(no_diff_crp['Symbol'].unique())
    # matched_quanti_and_qual_data_without_cnt_change = matched_quanti_and_qual_data_without_minus_eps.loc[matched_quanti_and_qual_data_without_minus_eps['Symbol'].isin(no_diff_crp_list)]
    # matched_quanti_and_qual_data_without_cnt_change.drop(columns = ['cnt_change'], inplace=True)
    ## 유상증자가 없는 기업만 포함 끝.
    print(matched_quanti_and_qual_data.shape)

    inv_var_for_ctrl = ['주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q',
                        # 'crp_cls_K', 'crp_cls_Y',
                        'ind_1차 금속 제조업', 'ind_가구 제조업', 'ind_가죽, 가방 및 신발 제조업',
                        'ind_건축 기술, 엔지니어링 및 기타 과학기술 서비스업',
                        'ind_고무 및 플라스틱제품 제조업', 'ind_교육 서비스업', 'ind_금속 가공제품 제조업; 기계 및 가구 제외',
                        'ind_금융업', 'ind_기타 개인 서비스업', 'ind_기타 기계 및 장비 제조업', 'ind_기타 운송장비 제조업',
                        'ind_기타 전문, 과학 및 기술 서비스업', 'ind_기타 제품 제조업', 'ind_농업', 'ind_담배 제조업',
                        'ind_도매 및 상품 중개업', 'ind_목재 및 나무제품 제조업; 가구 제외', 'ind_방송업', 'ind_보험 및 연금업',
                        'ind_부동산업', 'ind_비금속 광물제품 제조업', 'ind_비금속광물 광업; 연료용 제외', 'ind_사업 지원 서비스업',
                        'ind_사업시설 관리 및 조경 서비스업', 'ind_석탄, 원유 및 천연가스 광업', 'ind_섬유제품 제조업; 의복 제외',
                        'ind_소매업; 자동차 제외', 'ind_수상 운송업', 'ind_숙박업', 'ind_스포츠 및 오락관련 서비스업',
                        'ind_식료품 제조업', 'ind_어업', 'ind_연구개발업', 'ind_영상ㆍ오디오 기록물 제작 및 배급업',
                        'ind_우편 및 통신업', 'ind_육상 운송 및 파이프라인 운송업', 'ind_음료 제조업', 'ind_음식점 및 주점업',
                        'ind_의료, 정밀, 광학 기기 및 시계 제조업', 'ind_의료용 물질 및 의약품 제조업',
                        'ind_의복, 의복 액세서리 및 모피제품 제조업', 'ind_인쇄 및 기록매체 복제업', 'ind_임대업; 부동산 제외',
                        'ind_자동차 및 부품 판매업', 'ind_자동차 및 트레일러 제조업', 'ind_전기, 가스, 증기 및 공기 조절 공급업',
                        'ind_전기장비 제조업', 'ind_전문 서비스업', 'ind_전문직별 공사업', 'ind_전자 부품, 컴퓨터, 영상, 음향 및 통신장비 제조업',
                        'ind_정보서비스업', 'ind_종합 건설업', 'ind_창고 및 운송관련 서비스업', 'ind_창작, 예술 및 여가관련 서비스업', 'ind_출판업',
                        'ind_컴퓨터 프로그래밍, 시스템 통합 및 관리업', 'ind_코크스, 연탄 및 석유정제품 제조업',
                        'ind_펄프, 종이 및 종이제품 제조업', 'ind_폐기물 수집, 운반, 처리 및 원료 재생업', 'ind_항공 운송업',
                        'ind_화학 물질 및 화학제품 제조업; 의약품 제외', 'ind_환경 정화 및 복원업']
    for ctrl in inv_var_for_ctrl:
        matched_quanti_and_qual_data[ctrl] = matched_quanti_and_qual_data[ctrl].astype('category')

    for ctrl in inv_var_for_ctrl:
        if matched_quanti_and_qual_data[(matched_quanti_and_qual_data[ctrl] == 1)].shape[0] == 0:
            print(ctrl)
            matched_quanti_and_qual_data.drop([ctrl], inplace=True, axis=1)
    # matched_quanti_and_qual_data['t1_M000911020_유효세율(%)'] = matched_quanti_and_qual_data['t1_M000911020_유효세율(%)'] / 100.0
    # matched_quanti_and_qual_data['t2_M000911020_유효세율(%)'] = matched_quanti_and_qual_data['t2_M000911020_유효세율(%)'] / 100.0
    # matched_quanti_and_qual_data['t3_M000911020_유효세율(%)'] = matched_quanti_and_qual_data['t3_M000911020_유효세율(%)'] / 100.0
    # matched_quanti_and_qual_data['t4_M000911020_유효세율(%)'] = matched_quanti_and_qual_data['t4_M000911020_유효세율(%)'] / 100.0
    #
    # test = matched_quanti_and_qual_data.sort_values(['Symbol', '회계년', '주기', 't4_ETR'], ascending=['True', 'True', 'False', 'False', 'False', 'False'], inplace=True)

    print(matched_quanti_and_qual_data.shape)
    matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data[  # 이 조건을 만족하지 않는다면 DB가 뭔가 이상한거다.
        (matched_quanti_and_qual_data['t1_LF_salesDivEmp'] > 0)
        # & (matched_quanti_and_qual_data['t1_M000904017_판매비와관리비(천원)'] > 0)
        # & (matched_quanti_and_qual_data['t1_M000904007_매출총이익(천원)'] > 0)    ############### 4
        # & (matched_quanti_and_qual_data['t1_tminus_inv_M000901001_총자산(천원)'] > 0)
        # & (matched_quanti_and_qual_data['t1_M000901006_매출채권(천원)'] > 0)

        & (matched_quanti_and_qual_data[dep] != 0)
        ].copy()
    # matched_quanti_and_qual_data_fin.dropna(subset=[main_ind_var], inplace=True)
    print(matched_quanti_and_qual_data_fin.shape)
    # matched_quanti_and_qual_data_fin.replace(to_replace='N/A(IFRS)', value=np.nan, inplace=True)
    matched_quanti_and_qual_data_fin.loc[:, ~matched_quanti_and_qual_data_fin.columns.isin(['Symbol', '주기', 'ind'])] = \
        matched_quanti_and_qual_data_fin.loc[:, ~matched_quanti_and_qual_data_fin.columns.isin(['Symbol', '주기', 'ind'])].apply(pd.to_numeric)
    # list(matched_quanti_and_qual_data_fin.columns)

    new_eps_version = eps_version.copy()
    # new_eps_version = []
    # for eps in range(1, 5):
    # for eps in [1, 4]:
    #     new_eps_version.append('t'+str(eps)+'_' + eps_version)
    # ind_var_at_paper = ['t1_'+ind_var for ind_var in ind_var_list]
    # ind_var_at_paper.extend(new_eps_version)
    # if constant == False:
    #     ind_var_at_paper.extend(inv_var_for_ctrl)
    matched_quanti_and_qual_data_fin.sort_values(['회계년', '주기'], ascending=['True', 'True'], inplace=True)
    matched_quanti_and_qual_data_fin.reset_index(drop=True, inplace=True)
    matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data_fin[(matched_quanti_and_qual_data_fin['회계년'] > 2012)]


    TA_calc = pd.read_excel('./TA_calc.xlsx')  # 3번 조건. 2번은 이미 충족됨.
    TA_calc = q_replace_int(TA_calc)
    matched_quanti_and_qual_data_fin = pd.merge(left=matched_quanti_and_qual_data_fin, right=TA_calc,
                                            how='left', on=['Symbol', '회계년', '주기'], sort=False)

    identifier = ['Symbol', '회계년', '주기']
    indVar_withoutFootnote = identifier.copy()
    indVar_withoutFootnote.extend(ind_var_list)
    indVar_withoutFootnote.append(dep)

    indVar_withFootnote = identifier.copy()
    indVar_withFootnote.extend(ind_var_list)
    indVar_withFootnote.append(main_ind_var)  # footnotes
    # ind_var_at_paper.append(main_ind_var)
    indVar_withFootnote.append(dep)

    df = matched_quanti_and_qual_data_fin.copy()
    df = df[indVar_withFootnote].copy()
    df.dropna(inplace=True)

    getDA = matched_quanti_and_qual_data_fin[['Symbol', '회계년', '주기', 'tminus_inv_M000901001_총자산(천원)',
                                              'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)', 'TA_divLaggedAsset']].copy()
    getDA = getDA[~((getDA['회계년'] == 2018) & (getDA['주기'] == 4))]
    getDA.dropna(inplace=True)
    print(getDA.shape)
    print(getDA.isna().sum())
    # start_time = datetime.now()

    predict_df = getDA.iloc[:, :3]
    X = getDA[['tminus_inv_M000901001_총자산(천원)', 'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)']]
    y = getDA[['TA_divLaggedAsset']]

    result = sm.OLS(y, X).fit()
    # print('r_square : ', result.rsquared)
    print(result.summary())

    # print(result.predict(X))
    predict_df = getDA.iloc[:, :3]
    predictY = result.predict(X)
    predict_df['true'] = pd.Series(list(y.values.flatten()))
    predict_df['pred'] = pd.Series(list(predictY))
    predict_df['error'] = predict_df['true'] - predict_df['pred']
    result2 = result.get_robustcov_results()
    print('robust!')
    print(result2.summary())
    result.resid
    #### Dechow end
    #### moyerin start

    moyerin_ind_var = identifier.copy()
    major_moyerin_ind_var = [main_ind_var, 'CFO_ratio', 'BIG', 'SIZE', 'LEV', 'ROA', 'LOSS', 'MTB', 'LARGE', 'FOREIGN']
    moyerin_ind_var.extend(major_moyerin_ind_var)
    ctrl_ind = ['ind_1차 금속 제조업',
                'ind_가구 제조업',
                'ind_가죽, 가방 및 신발 제조업',
                'ind_건축 기술, 엔지니어링 및 기타 과학기술 서비스업',
                'ind_고무 및 플라스틱제품 제조업',
                'ind_교육 서비스업',
                'ind_금속 가공제품 제조업; 기계 및 가구 제외',
                'ind_금융업',
                'ind_기타 기계 및 장비 제조업',
                'ind_기타 운송장비 제조업',
                'ind_기타 전문, 과학 및 기술 서비스업',
                'ind_기타 제품 제조업',
                'ind_농업',
                'ind_담배 제조업',
                'ind_도매 및 상품 중개업',
                'ind_목재 및 나무제품 제조업; 가구 제외',
                'ind_방송업',
                'ind_보험 및 연금업',
                'ind_부동산업',
                'ind_비금속 광물제품 제조업',
                'ind_비금속광물 광업; 연료용 제외',
                'ind_사업 지원 서비스업',
                'ind_사업시설 관리 및 조경 서비스업',
                'ind_석탄, 원유 및 천연가스 광업',
                'ind_섬유제품 제조업; 의복 제외',
                'ind_소매업; 자동차 제외',
                'ind_수상 운송업',
                'ind_숙박업',
                'ind_스포츠 및 오락관련 서비스업',
                'ind_식료품 제조업',
                'ind_어업',
                'ind_연구개발업',
                'ind_영상ㆍ오디오 기록물 제작 및 배급업',
                'ind_우편 및 통신업',
                'ind_육상 운송 및 파이프라인 운송업',
                'ind_음료 제조업',
                'ind_의료, 정밀, 광학 기기 및 시계 제조업',
                'ind_의료용 물질 및 의약품 제조업',
                'ind_의복, 의복 액세서리 및 모피제품 제조업',
                'ind_인쇄 및 기록매체 복제업',
                'ind_임대업; 부동산 제외',
                'ind_자동차 및 부품 판매업',
                'ind_자동차 및 트레일러 제조업',
                'ind_전기, 가스, 증기 및 공기 조절 공급업',
                'ind_전기장비 제조업',
                'ind_전문 서비스업',
                'ind_전문직별 공사업',
                'ind_전자 부품, 컴퓨터, 영상, 음향 및 통신장비 제조업',
                'ind_정보서비스업',
                'ind_종합 건설업',
                'ind_창고 및 운송관련 서비스업',
                'ind_창작, 예술 및 여가관련 서비스업',
                'ind_출판업',
                'ind_컴퓨터 프로그래밍, 시스템 통합 및 관리업',
                'ind_코크스, 연탄 및 석유정제품 제조업',
                'ind_펄프, 종이 및 종이제품 제조업',
                'ind_폐기물 수집, 운반, 처리 및 원료 재생업',
                'ind_항공 운송업',
                'ind_화학 물질 및 화학제품 제조업; 의약품 제외']
    moyerin_ind_var.extend(ctrl_ind)
    major_moyerin_ind_var.extend(ctrl_ind)

    calc_cosdist_relation = matched_quanti_and_qual_data_fin[moyerin_ind_var].copy()
    calc_cosdist_relation.dropna(subset=[main_ind_var], inplace=True)
    calc_cosdist_relation['주기'] = calc_cosdist_relation['주기'].astype('int')
    # moyerin_modif = match_tplus_data_modif(calc_cosdist_relation, predict_df, ['error'], moyerin_ind_var, 'moyerin_backup.xlsx')
    # moyerin_modif = pd.merge(left=predict_df, right=calc_cosdist_relation,
    #                          how='left', on=['Symbol', '회계년', '주기'], sort=False)
    moyerin_modif = pd.read_excel('./merged_FnGuide/moyerin_backup.xlsx')
    moyerin_modif.dropna(inplace=True)
    moyerin_modif = add_one_hot(moyerin_modif, '회계년')
    major_moyerin_ind_var.extend(['회계년_2013', '회계년_2014', '회계년_2015', '회계년_2016', '회계년_2017', '회계년_2018'])

    new_X = moyerin_modif[major_moyerin_ind_var]
    new_X = sm.add_constant(new_X)
    new_y = moyerin_modif[['error']]
    result3 = sm.OLS(new_y, new_X).fit()
    print(result3.summary())
    result4 = result3.get_robustcov_results()
    print(result4.summary())

    moyerin_modif.loc[:, 'abs_error'] = np.abs(moyerin_modif['error'].copy())
    new_y2 = moyerin_modif[['abs_error']]
    result5 = sm.OLS(new_y2, new_X).fit()
    print(result5.summary())
    result6 = result5.get_robustcov_results()
    print(result6.summary())

    qual_dat = pd.read_excel('before_divide1and2.xlsx')
    matched_quanti_and_qual_data_fin = q_replace_int(matched_quanti_and_qual_data_fin)

    matched = pd.merge(left=qual_dat[['Symbol', '회계년', '주기', 'index']].copy(), right=moyerin_modif,
                       how='inner', on=['Symbol', '회계년', '주기'], sort=False)
    new_X2 = matched[major_moyerin_ind_var]
    new_X2 = sm.add_constant(new_X2)
    new_y3 = matched['error']
    new_y4 = matched['abs_error']

    result7 = sm.OLS(new_y3, new_X2).fit()
    print(result7.summary())
    result8 = result7.get_robustcov_results()
    print(result8.summary())

    result9 = sm.OLS(new_y4, new_X2).fit()
    print(result9.summary())
    result10 = result9.get_robustcov_results()
    print(result10.summary())

    # 유의하지 않은 변수 제외하고 다시
    new_X3 = matched[[main_ind_var]]
    new_X3 = sm.add_constant(new_X3)

    result11 = sm.OLS(new_y3, new_X3).fit()
    print(result11.summary())
    result12 = result11.get_robustcov_results()
    print(result12.summary())

    result13 = sm.OLS(new_y4, new_X3).fit()
    print(result13.summary())
    result14 = result13.get_robustcov_results()
    print(result14.summary())


    """
    rmse1, adjrmse1, mape1, predict_df1 = ols(data1, ind_var_list, dep, constant)
    print(rmse1)
    print(mape1)
    rmse_all1, adjrmse_all1, mape_all1, predict_df_all1 = ols_all(data1, ind_var_list, dep, constant)

    data2 = matched_quanti_and_qual_data_fin[indVar_withFootnote].copy()
    data2 = data2.dropna()
    print(data2.shape)
    print(data2.isna().sum())
    print(data1.shape[1] == data2.shape[1]-1)
    rmse2, adjrmse2, mape2, predict_df2 = ols(data2, ind_var_list, dep, constant)
    print(rmse2)
    print(mape2)
    rmse_all2, adjrmse_all2, mape_all2, predict_df_all2 = ols_all(data2, ind_var_list, dep, constant)

    np.random.seed(42)  # for placebo
    data3 = data2.copy()
    data3[main_ind_var] = np.random.uniform(0, 2, size=data3.shape[0])  # for placebo
    print(data3.shape)
    print(data3.isna().sum())
    print(data3.shape[1] == data2.shape[1])
    rmse3, adjrmse3, mape3, predict_df3 = ols(data3, ind_var_list, dep, constant)
    print(rmse3)
    print(mape3)
    rmse_all3, adjrmse_all3, mape_all3, predict_df_all3 = ols_all(data3, ind_var_list, dep, constant)

    
    vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(data2.loc[:, ind_var_at_paper].values, i) for i in range(data2.loc[:, ind_var_at_paper].shape[1])]
    col = ['t1_M000901012_재고자산(천원)', 't1_M000901006_매출채권(천원)', 't1_M_CAPEX', 't1_M000904007_매출총이익(천원)',
           't1_M000904017_판매비와관리비(천원)', 't1_ETR', 't1_LF_salesDivEmp',
           't1_M000909001_영업활동으로인한현금흐름(천원)', 't1_tminus_inv_M000901001_총자산(천원)', 't1_deltaREV_minus_deltaREC',
           't1_div_A_M000901017_유형자산(천원)', 't1_BIG', 't1_SIZE', 't1_LEV', 't1_ROA', 't1_LOSS', 't1_MTB', 't1_LARGE', 't1_FOREIGN',
           't1_t_1M000601005_EPS(보통주)(원)', 't2_t_1M000601005_EPS(보통주)(원)',
           't3_t_1M000601005_EPS(보통주)(원)', 't4_t_1M000601005_EPS(보통주)(원)', 't_1q_cos_dist']
    vif["VIF Factor"] = [variance_inflation_factor(data2.loc[:, col].values, i) for i in range(data2.loc[:, col].shape[1])]
    vif["features"] = col
    print(vif)
    """
    # cor_matrix = data2.loc[:, col].corr()
    # cor_matrix.to_csv('cor_mat.csv', encoding='cp949')

    # print('ols All result')
    # print(rmse_all1)
    # print(adjrmse_all1)
    # print(mape_all1)
    # print(rmse_all2)
    # print(adjrmse_all2)
    # print(mape_all2)
    # print(rmse_all3)
    # print(adjrmse_all3)
    # print(mape_all3)

    # rmse1_avg = mean_per_n(rmse1, 3)
    # mape1_avg = mean_per_n(mape1, 3)
    # rmse2_avg = mean_per_n(rmse2, 3)
    # mape2_avg = mean_per_n(mape2, 3)
    # rmse3_avg = mean_per_n(rmse3, 3)
    # mape3_avg = mean_per_n(mape3, 3)



    """
    print('-------------------------------------------------')
    print('rmse check')
    result1 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse2)  # 독립 t-test 단방향 검정
    print('adjrmse check')
    result2 = pfd.equ_var_test_and_unpaired_t_test(adjrmse1, adjrmse2)  # 독립 t-test 단방향 검정
    print('mape check')
    result3 = pfd.equ_var_test_and_unpaired_t_test(mape1, mape2)  # 독립 t-test 단방향 검정


    print('placebo check-------------------------------------------------')
    print('rmse check')
    result4 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse3)  # placebo 효과 확인.
    print('adjrmse check')
    result5 = pfd.equ_var_test_and_unpaired_t_test(adjrmse1, adjrmse3)  # 독립 t-test 단방향 검정
    print('mape check')
    result6 = pfd.equ_var_test_and_unpaired_t_test(mape1, mape3)  # placebo 효과 확인.

    ind_var_at_paper_new = ind_var_at_paper.copy()
    # ind_var_at_paper_new.append(main_ind_var)
    col_new = col.copy()

    # ind_var_at_paper_new.remove('t1_SIZE')
    # col_new.remove('t1_SIZE')
    ind_var_at_paper_new.remove('t1_tminus_inv_M000901001_총자산(천원)')
    col_new.remove('t1_tminus_inv_M000901001_총자산(천원)')

    ind_var_at_paper_new.remove('t1_deltaREV_minus_deltaREC')
    col_new.remove('t1_deltaREV_minus_deltaREC')

    ind_var_at_paper_new.remove('')
    ind_var_at_paper_new.remove('')
    ind_var_at_paper_new.remove('')
    ind_var_at_paper_new.remove('')
    ind_var_at_paper_new.remove('')


    rmse_all, adjrmse_all, mape_all, predict_df_all = ols_all(data2, ind_var_at_paper_new, dep, constant)
    new_vif = pd.DataFrame()
    new_vif["VIF Factor"] = [variance_inflation_factor(data2.loc[:, col_new].values, i) for i in range(data2.loc[:, col_new].shape[1])]
    new_vif["features"] = col_new
    print(new_vif)
    """