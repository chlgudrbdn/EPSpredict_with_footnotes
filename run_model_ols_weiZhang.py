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
from scipy.linalg import toeplitz
import seaborn as sns
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.diagnostic import het_goldfeldquandt


def adjusted_rmse(y_true, y_pred):  # 사실 2009년엔 없는 단어여서 그런지 모르겠지만 이걸 Mean Square Percentage Error 라고 함. mape보다 인기는 없어보인다.
    # return np.sqrt((((targets - predictions)/targets) ** 2).mean())
    return math.sqrt(np.square(np.divide(np.subtract(y_true, y_pred), y_true)).mean())


def Mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


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


def match_quanti_and_qual_data(qual_ind_var, quanti_ind_var, file_name):  # 정량 독립 및 정량 종속 변수와 정성 독립 변수 매칭
    result_df = pd.DataFrame()
    valid_df_idx_list = []
    for index, row in qual_ind_var.iterrows():  # 무식하지만 결국 이 틀에서 벗어나기 어렵다.
        rpt_nm = row['rpt_nm']
        t_closing_date = rpt_nm[rpt_nm.find("(")+1:rpt_nm.find(")")].split('.')
        t_year = int(t_closing_date[0])
        t_month = int(t_closing_date[1])

        rpt = rpt_nm.split()
        if rpt[0] == '반기보고서' and t_month == 6:
            t_quarter = '2Q'
        elif rpt[0] == '사업보고서' and t_month == 12:
            t_quarter = '4Q'
        elif rpt[0] == '분기보고서' and t_month == 3:  # 주기와 맞는다는 보장은 없다. 이게 맞길 바래야함.
            t_quarter = '1Q'
        elif rpt[0] == '분기보고서' and t_month == 9:
            t_quarter = '3Q'
        else:  # 혹시 모르니 일단 예외처리.  # 직접 확인한 결과 612건. 예를들어 4월부터 9월까지를 반기로 치는 중소기업이 있었다(000220). 무시해도 좋다고 판단함.
            print('exeception idx:', index, ' month:', t_month, ' rpt_nm:', rpt_nm)
            t_quarter = ''
            result_df = result_df.append(pd.Series(), ignore_index=True)
            continue
        qual_ind_var.loc[index, '주기'] = t_quarter
        qual_ind_var.loc[index, '회계년'] = t_year
        qual_ind_var.loc[index, 'Symbol'] = 'A'+str(row['crp_cd'])
        # tplus_data = quanti_ind_var[(quanti_ind_var['Symbol'] == 'A'+str(row['crp_cd'])) &
        #                          (quanti_data['결산월'] == t_closing_date.month) &
                                 # (quanti_ind_var['주기'] == t_quarter) &
                                 # (quanti_ind_var['회계년'] == t_year)]
        # if tplus_data.shape[0] > 1:
        #     print('duplicated ', index)  # 없겠지만 중복이 생길 경우 예외처리를 위함.

        # if tplus_data.empty:
        #     print('empty ', index)  # 약 2260건. 필연적으로 어딘가 비면 생길 수 밖에 없는 문제다.
        #     result_df = result_df.append(pd.Series(), ignore_index=True)
        #     continue
        # valid_df_idx_list.append(index)  # 최종적으로는 매칭에 이것만 있으면 된다. # 일단 적절한 값이 없는 경우 알아서 생략되도록 앞의 코드에서 처리.
        # result_df = result_df.append(tplus_data, ignore_index=True)
    # qual_ind_var.reset_index(drop=True, inplace=True)
    # result_df.reset_index(drop=True, inplace=True)
    matched_quanti_and_qual_data = pd.merge(left=quanti_ind_var, right=qual_ind_var, how='left', on=['Symbol', '회계년', '주기'], sort=False)

    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    matched_quanti_and_qual_data.to_pickle(directory_name+'/'+file_name)
    print(len(valid_df_idx_list))
    print(matched_quanti_and_qual_data.shape)
    return matched_quanti_and_qual_data


def add_one_hot(df, col_name):
    df = pd.concat([df, pd.get_dummies(df[col_name], dummy_na=False, prefix=col_name)], axis=1)
    # df.drop([col_name], axis=1, inplace=True)
    return df


def q_replace_int(df):
    df.loc[df['주기'] == '1Q', '주기'] = 1
    df.loc[df['주기'] == '2Q', '주기'] = 2
    df.loc[df['주기'] == '3Q', '주기'] = 3
    df.loc[df['주기'] == '4Q', '주기'] = 4
    return df


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

    result_df = pd.DataFrame(columns=identifier1)
    for cd in cd_list:
        # cd = 'A000020'  # FOR TEST
        tmp = df[(df['Symbol'] == cd)].copy()
        tmp.reset_index(drop=True, inplace=True)
        for idx in range(3, tmp.shape[0]):  # 3 ~ 4*8(2011, 2012,2013,2014, 2015,2016,2017,2018개 년)
            # idx = 3  # for test
            tmp4 = tmp.loc[idx-3:idx, :].copy()
            tmp4 = tmp4.loc[:, identifier2]  # 13
            tmp4.reset_index(drop=True, inplace=True)
            # print(list(tmp4.columns))
            # print(tmp4.shape)
            # print(tmp4.at[3,'Symbol'])
            ident = [tmp4.iloc[3]['Symbol'], tmp4.iloc[3]['회계년'], tmp4.iloc[3]['주기']]
            flatten = list(tmp4[ind_var_list].values.flatten())  # 36 = 4 * 9
            # print(len(flatten) == tmp4.shape[0] * (tmp4.shape[1]-4))
            ident.extend(flatten)  # 39 = 3 + 36
            # ident.append(tmp4.loc[3, dep])  # 40 = 39 + 1
            new_row = pd.DataFrame(columns=identifier1, data=[ident])
            # print(new_row)
            result_df = result_df.append(new_row)
        result_df.reset_index(drop=True, inplace=True)
        # result_df.dropna(inplace=True)
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


def Mae(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))


def Mpe(y_true, y_pred): # underperformance 인지 overperformance 인지 판단 할 수 있다는 것입니다.
    return np.mean((y_true - y_pred) / y_true) * 100


def ols(df, x_var_names, y_var_name):
    testRmseList = []
    testmaeList = []
    predict_df_li = []
    for year in [2016, 2017, 2018]:  # 일종의 3-fold가 된다.
        # df = data2.copy()
        # year= 2016  # for test
        # x_var_names = ind_var_at_paper
        # y_var_name = dep
        print('------------------------\nseed - year: ', year)
        train = df[(df['회계년'] < year)]
        test = df[(df['회계년'] == year)]
        print('train ', train.shape)
        print('test ', test.shape)
        predict_df = test.iloc[:, :3]
        print(predict_df.shape)
        train_X = train[x_var_names]
        train_y = train[y_var_name]
        train_X = sm.add_constant(train_X)
        result = sm.OLS(train_y, train_X).fit()
        # print('r_square : ', result.rsquared)
        print(result.summary())
        predict_df.reset_index(drop=True, inplace=True)
        # print(result.predict(X))
        test_X = test[x_var_names]
        test_y = test[y_var_name]
        test_X = sm.add_constant(test_X)
        predictY = result.predict(test_X)
        rmse = sqrt(mean_squared_error(test_y, predictY))
        print('rmse: ', rmse)
        mae = Mae(test_y, predictY)
        print('mae: ', mae)
        mpe = Mpe(test.iloc[:, -1], predictY)
        if mpe < 0:
            print(mpe, 'over perform')
        else:
            print(mpe, 'under perform')
        testRmseList.append(rmse)
        testmaeList.append(mae)
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
    return testRmseList, testmaeList, predict_df


def ols_all(data, x_var_names, y_var_name):
    # year= 2016  # for test
    print('------------------------\nols ALL: ')
    predict_df = data.iloc[:, :3]

    X = data[x_var_names]
    y = data[y_var_name]
    X = sm.add_constant(X)
    result = sm.OLS(y, X).fit()
    # print('r_square : ', result.rsquared)
    print(result.summary())

    # print(result.predict(X))
    predictY = result.predict(X)
    rmse = sqrt(mean_squared_error(y, predictY))
    print('rmse: ', rmse)
    mae = mean_absolute_error(y, predictY)
    print('mae: {0:.4f}'.format(round(mae, 3)))
    mape = Mape(y, predictY)
    print('mape: ', mape)
    adjrmse = adjusted_rmse(y, predictY)
    print('adjusted_rmse: ', adjrmse)
    mpe = Mpe(y, predictY)
    if mpe < 0:
        print('over perform')
    else:
        print('under perform')
    predict_df['true'] = pd.Series(list(y))
    predict_df['pred'] = pd.Series(list(predictY))

    return rmse, mae, mape, adjrmse, predict_df, result


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    # path_dir = 'C:/Users/lab515/PycharmProjects/eps_predict/'
    path_dir = os.getcwd()
    ind_var_list = ['M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)',
                    'M000904017_판매비와관리비(천원)', 'M000911020_유효세율(%)', 'LF_salesDivEmp'
        # , 'CFO_ratio'  # 황선희 2006 연구
        #             # , 'tminus_inv_M000901001_총자산(천원)', 'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)'   # Dechow 1995  # 16개 변수
        # , 'BIG', 'SIZE', 'LEV', 'ROA', 'LOSS', 'MTB', 'LARGE', 'FOREIGN'  # SIZE, LEV, MTB, ROA, LOSS, CFO, BIG, LARGE, Foreign,      YD, ID, e  # 모예린 2018  # CFO는 이미 포함. 근데 영 유의하지 않아서 빼는걸 고려.
                    ]  # 여기에 사용할 eps를 더 추가
    version_number = 1
    dep_var = ['dep_M000601002_EPS(원)', 'dep_M000601005_EPS(보통주)(원)',
               'dep_M000601003_수정EPS(원)', 'dep_M000601006_수정EPS(보통주)(원)']  # 쓸걸로 예상
    eps_versions = ['t_1M000601002_EPS(원)', 't_1M000601005_EPS(보통주)(원)',
                    't_1M000601003_수정EPS(원)', 't_1M000601006_수정EPS(보통주)(원)']
    ## option ##
    dep = dep_var[version_number]
    eps_version = eps_versions[version_number]
    main_ind_var = 't_1q_cos_dist'
    # main_ind_var = 't_1y_cos_dist'
    # main_ind_var = 'diff_q_cos_per_ind'  # 산업 평균 대비 분기 cos거리
    # main_ind_var = 'diff_y_cos_per_ind'  # 산업 평균 대비 연 cos거리
    ## option ##

    # for i in range(1,5):
    #     't'+str(i)+'_'+eps_version
    # ind_var_list.extend(eps_version)  # 9개 변수

    # quanti_qual_matched_file_name = 'quanti_qaul_komoran_dnn.pkl'  # 정성 데이터 포함시킨 데이터.
    quanti_qual_matched_file_name = 't4_olsrevisionAll_quanti_qaul_komoran_dnn.pkl'  # 정성 데이터 포함시킨 데이터.
    """     
    # dep_vars = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for dependent variable\\EPS_rawData.xlsx'
    #                         , dtype=object, sheet_name='Sheet1')  # 종속변수
    df8 = pd.read_pickle(path_dir + '/divide_by_sector/filter8 komoran_for_cosine_distance.pkl')  # 독립변수중 산업이나 코사인거리
    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , dtype=object, sheet_name='ind_weiZhang_t4')  # 미리 종속변수까지 붙여놓아 번거로운 작업을 할 필요는 없음.

    matched_quanti_and_qual_data = match_quanti_and_qual_data(df8, quanti_ind_var, 'processing_'+dep+quanti_qual_matched_file_name)
    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/processing_'+dep +"_"+ quanti_qual_matched_file_name)
    """
    """     
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/processing_'+dep+"_"+quanti_qual_matched_file_name)

    matched_quanti_and_qual_data['회계년'] = matched_quanti_and_qual_data['회계년'].astype('int')
    matched_quanti_and_qual_data = matched_quanti_and_qual_data[(matched_quanti_and_qual_data['회계년']) < 2019]

    # matched_quanti_and_qual_data_for_test = matched_quanti_and_qual_data.copy()
    # matched_quanti_and_qual_data_for_test.dropna(subset=['ind_cd'], inplace=True)
    # matched_quanti_and_qual_data_for_test = matched_quanti_and_qual_data_for_test[['Symbol', 'ind_cd']]
    # matched_quanti_and_qual_data_for_test = matched_quanti_and_qual_data_for_test.drop_duplicates()
    # matched_quanti_and_qual_data_for_test.shape
    # matched_quanti_and_qual_data_for_test.to_excel('crp_ind_match.xlsx')

    crp_ind_match = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\crp_ind_match.xlsx'
                                   , dtype=object, sheet_name='Sheet1')  # 미리 종속변수까지 붙여놓아 번거로운 작업을 할 필요는 없음.
    cd_list = list(matched_quanti_and_qual_data['Symbol'].unique())
    for cd in cd_list:
        # cd = 'A005930'  # for test
        try:
            matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data['Symbol'] == cd), 'ind_cd'] = crp_ind_match.loc[crp_ind_match['Symbol'] == cd, 'ind_cd'].values[0]
        except Exception as e:
            print(cd)
            print(e)

    columns = ['crp_cd', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt', 't_minus_index', 't_minus_year_index',
               'Name', '결산월']  # Symbol, 회계년, 주기, ind_cd는 차후 식별등을 위해 일단 남긴다.  # 'ind','crp_cls'나중에 지운다.

    matched_quanti_and_qual_data.drop(columns, inplace=True, axis=1)

    matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].replace("", np.nan)
    matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')

    matched_quanti_and_qual_data['t_1y_cos_dist'] = matched_quanti_and_qual_data['t_1y_cos_dist'].replace("", np.nan)
    matched_quanti_and_qual_data['t_1y_cos_dist'] = matched_quanti_and_qual_data['t_1y_cos_dist'].astype('float')

    matched_quanti_and_qual_data = add_one_hot(matched_quanti_and_qual_data, '주기')
    matched_quanti_and_qual_data = add_one_hot(matched_quanti_and_qual_data, 'crp_cls')
    matched_quanti_and_qual_data = add_one_hot_with_ind_cd(matched_quanti_and_qual_data)  # 중분류 산업코드 사용. 대분류 제조업은

    ind_cosine_dict = {}
    for ind_col_name in list(matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^ind')]):
        print(ind_col_name)
        if ind_col_name == 'ind':
            continue
        for year in range(2013, 2019):
            for quarter in ['주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q']:
                avg_q_dist = matched_quanti_and_qual_data[(matched_quanti_and_qual_data['회계년'] == year) & (matched_quanti_and_qual_data[ind_col_name] == 1) & (matched_quanti_and_qual_data[quarter] == 1)]['t_1q_cos_dist'].mean(skipna=True)
                avg_y_dist = matched_quanti_and_qual_data[(matched_quanti_and_qual_data['회계년'] == year) & (matched_quanti_and_qual_data[ind_col_name] == 1) & (matched_quanti_and_qual_data[quarter] == 1)]['t_1y_cos_dist'].mean(skipna=True)
                ind_cosine_dict[ind_col_name+'$'+str(year)+'$'+quarter] = [avg_q_dist, avg_y_dist]
    # matched_quanti_and_qual_data.drop(['q_cos_mean', 'y_cos_mean', 'diff_q_cos_per_ind', 'diff_y_cos_per_ind'], inplace=True, axis=1)
    # cols = matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^q_cos_ind')]
    # matched_quanti_and_qual_data.drop(cols, inplace=True, axis=1)
    # cols = matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^y_cos_ind')]
    # matched_quanti_and_qual_data.drop(cols, inplace=True, axis=1)

    for key in ind_cosine_dict:
        # key = 'ind_항공 운송업'  # for test
        matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data['회계년'] == int(key.split('$')[1])) & (matched_quanti_and_qual_data[key.split('$')[0]] == 1) & (matched_quanti_and_qual_data[key.split('$')[2]] == 1), "q_cos_mean"] = ind_cosine_dict[key][0]
        matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data['회계년'] == int(key.split('$')[1])) & (matched_quanti_and_qual_data[key.split('$')[0]] == 1) & (matched_quanti_and_qual_data[key.split('$')[2]] == 1), "y_cos_mean"] = ind_cosine_dict[key][1]
    # matched_quanti_and_qual_data.dropna(inplace=True)
    matched_quanti_and_qual_data['diff_q_cos_per_ind'] = matched_quanti_and_qual_data['t_1q_cos_dist'] - matched_quanti_and_qual_data['q_cos_mean']
    matched_quanti_and_qual_data['diff_y_cos_per_ind'] = matched_quanti_and_qual_data['t_1y_cos_dist'] - matched_quanti_and_qual_data['y_cos_mean']

    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/diffPerInd_'+dep+"_"+ quanti_qual_matched_file_name)
    """
    """   
    # matched_quanti_and_qual_data.drop(['t_1y_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data.drop(['t_1q_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/diffPerInd_'+dep +"_"+quanti_qual_matched_file_name)
    # print(matched_quanti_and_qual_data.dtypes)
    # matched_quanti_and_qual_data.sort_values(['Symbol', '회계년', '주기'], ascending=['True', 'True', 'True'], inplace=True)
    matched_quanti_and_qual_data_t4Flatten = rolling_t4(matched_quanti_and_qual_data, ind_var_list)
    # matched_quanti_and_qual_data_t4Flatten.reset_index(drop=True, inplace=True)
    # result_df.dropna(inplace=True)

    columns = ['Symbol', '회계년', '주기', '주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q',
               # 'M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)', 'M000904017_판매비와관리비(천원)', 'M000911020_유효세율(%)', 'LF_salesDivEmp',
               'dep_M000601002_EPS(원)', 'dep_M000601005_EPS(보통주)(원)',
               'dep_M000601003_수정EPS(원)', 'dep_M000601006_수정EPS(보통주)(원)',
               't_1q_cos_dist', 't_1y_cos_dist', 'diff_q_cos_per_ind', 'diff_y_cos_per_ind',
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

    merged_df = pd.merge(left=matched_quanti_and_qual_data_t4Flatten, right=matched_quanti_and_qual_data[columns],
                         how='left', on=['Symbol', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    merged_df = merged_df[(merged_df['회계년']) > 2012]


    # matched_quanti_and_qual_data_t4Flatten.drop(['dep_M000601002_EPS(원)'], inplace=True, axis=1)

    # matched_quanti_and_qual_data_m = pd.read_pickle('./merged_FnGuide/processing_'+quanti_qual_matched_file_name)
    # matched_quanti_and_qual_data_m['회계년'] = matched_quanti_and_qual_data_m['회계년'].astype('int')
    # matched_quanti_and_qual_data_m = matched_quanti_and_qual_data_m[(matched_quanti_and_qual_data_m['회계년']) < 2019]

    # matched_quanti_and_qual_data_for_test = matched_quanti_and_qual_data.copy()
    # matched_quanti_and_qual_data_for_test.dropna(subset=['ind_cd'], inplace=True)
    # matched_quanti_and_qual_data_for_test = matched_quanti_and_qual_data_for_test[['Symbol', 'ind_cd']]
    # matched_quanti_and_qual_data_for_test = matched_quanti_and_qual_data_for_test.drop_duplicates()
    # matched_quanti_and_qual_data_for_test.shape
    # matched_quanti_and_qual_data_for_test.to_excel('crp_ind_match.xlsx')

    merged_df.to_pickle(path_dir + '/merged_FnGuide/t4_'+dep +"_"+ quanti_qual_matched_file_name)
    # matched_quanti_and_qual_data  = merged_df
    """
    """    """
    matched_quanti_data = pd.read_pickle('./merged_FnGuide/'+quanti_qual_matched_file_name)
    matched_quanti_and_qual_data = matched_quanti_data.copy()

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
    matched_quanti_and_qual_data.replace(to_replace='N/A(IFRS)', value=np.nan, inplace=True)

    # 2009 논문 참조
    # 1분기전 EPS, 4분기전 EPS, 1분기전 (IV, AR, CAPX,GM,SA,ETR,LFP)
    # 1분기전 EPS, 4분기전 EPS, 4분기전 (IV, AR, CAPX,GM,SA,ETR,LFP)
    # matched_quanti_and_qual_data.loc[:, ~matched_quanti_and_qual_data.columns.isin(['Symbol', '주기', 'ind'])] = \
    #     matched_quanti_and_qual_data.loc[:, ~matched_quanti_and_qual_data.columns.isin(['Symbol', '주기', 'ind'])].apply(pd.to_numeric)

    matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data[
        ## first
        # (matched_quanti_and_qual_data['t1_M000904007_매출총이익(천원)'] > 0)  # 이 조건을 만족하지 않는다면 DB가 뭔가 이상한거다.
        # & (matched_quanti_and_qual_data['t1_M000904017_판매비와관리비(천원)'] > 0)
        (matched_quanti_and_qual_data['t1_LF_salesDivEmp'] > 0)
    ].copy()
    ## second        ## third
    # (matched_quanti_and_qual_data['t4_M000904007_매출총이익(천원)'] > 0)
    # & (matched_quanti_and_qual_data['t4_M000904017_판매비와관리비(천원)'] > 0)
    # & (matched_quanti_and_qual_data['t4_LF_salesDivEmp'] > 0)].copy()

    print(matched_quanti_and_qual_data_fin.shape)
    # matched_quanti_and_qual_data_fin.dropna(subset=[main_ind_var], inplace=True)
    print(matched_quanti_and_qual_data_fin.shape)
    matched_quanti_and_qual_data_fin['년주기'] = matched_quanti_and_qual_data_fin['회계년'].map(str)+matched_quanti_and_qual_data_fin['주기'].map(str)
    matched_quanti_and_qual_data_fin = add_one_hot(matched_quanti_and_qual_data_fin, '년주기')

    # matched_quanti_and_qual_data.sort_values(['Symbol', '회계년', '주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q'], ascending=['True', 'True', 'False', 'False', 'False', 'False'], inplace=True)
    # matched_quanti_and_qual_data_fin.replace(to_replace='N/A(IFRS)', value=np.nan, inplace=True)

    # list(matched_quanti_and_qual_data_fin.columns)


    ## first
    # new_eps_version = []
    # for eps in eps_version:
    #     new_eps_version.append('t1_'+eps)  # 8 *4 = 36
    # ind_var_at_paper = ['t1_M000901012_재고자산(천원)', 't1_M000901006_매출채권(천원)', 't1_M_CAPEX', 't1_M000904007_매출총이익(천원)',
    #                     't1_M000904017_판매비와관리비(천원)', 't1_M000911020_유효세율(%)', 't1_LF_salesDivEmp'
    #                     # 't1_t_1M000601002_EPS(원)', 't1_t_4M000601002_EPS(원)'
    #                     ]
    # ind_var_at_paper.extend(new_eps_version)
    ## second
    # new_eps_version = []
    # for eps in eps_version:
    #     new_eps_version.append('t1_' + eps)  # 8 *4 = 36
    #
    # ind_var_at_paper = ['t4_M000901012_재고자산(천원)','t4_M000901006_매출채권(천원)', 't4_M_CAPEX', 't4_M000904007_매출총이익(천원)',
    #                     't4_M000904017_판매비와관리비(천원)', 't4_M000911020_유효세율(%)', 't4_LF_salesDivEmp'
    #                     # 't1_t_1M000601002_EPS(원)', 't1_t_4M000601002_EPS(원)',
    #                     ]
    # ind_var_at_paper.extend(new_eps_version)
    ## third
    new_eps_version = []
    for t in [1, 4]:
        new_eps_version.append('t'+str(t)+'_' + eps_version)  # 8 *4 = 36
    add_ind_var_list = ind_var_list.copy()
    ind_var_at_paper = []  # 1~4분기전 데이터
    for i in [4]:  #
        ind_var_at_paper.extend(['t'+str(i)+'_'+ind_var for ind_var in add_ind_var_list])
    ind_var_at_paper.extend(new_eps_version)

    inv_var_for_ctrl = ['ind_1차 금속 제조업',
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
    ctrl_q = ['년주기_20121',
              '년주기_20122',
              '년주기_20123',
              '년주기_20124',
              '년주기_20131',
              '년주기_20132',
              '년주기_20133',
              '년주기_20134',
              '년주기_20141',
              '년주기_20142',
              '년주기_20143',
              '년주기_20144',
              '년주기_20151',
              '년주기_20152',
              '년주기_20153',
              '년주기_20154',
              '년주기_20161',
              '년주기_20162',
              '년주기_20163',
              '년주기_20164',
              '년주기_20171',
              '년주기_20172',
              '년주기_20173',
              '년주기_20174',
              '년주기_20181',
              '년주기_20182',
              '년주기_20183']
    # for ctrl in inv_var_for_ctrl:
    #     matched_quanti_and_qual_data[ctrl] = matched_quanti_and_qual_data[ctrl].astype('category')
    matched_quanti_and_qual_data_fin.sort_values(['회계년', '주기'], ascending=['True', 'True'], inplace=True)

    # for ctrl in inv_var_for_ctrl:
    #     if matched_quanti_and_qual_data_fin[(matched_quanti_and_qual_data[ctrl] == 1)].shape[0] == 0:
    #         print(ctrl)
    #         matched_quanti_and_qual_data_fin.drop([ctrl], inplace=True, axis=1)

    identifier = ['Symbol', '회계년', '주기']
    indVar_withoutFootnote = identifier.copy()
    indVar_withoutFootnote.extend(ind_var_at_paper)
    # indVar_withoutFootnote.extend(inv_var_for_ctrl)  # optional for contorl
    indVar_withoutFootnote.append(dep)

    moyerin_ind_var = ['M000909001_영업활동으로인한현금흐름(천원)'
                        , 'CFO_ratio', 'BIG', 'SIZE', 'LEV', 'ROA', 'LOSS', 'MTB', 'LARGE', 'FOREIGN'
                        ,'tminus_inv_M000901001_총자산(천원)', 'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)'
                       ]

    data1 = matched_quanti_and_qual_data_fin[indVar_withoutFootnote].copy()
    data1.dropna(inplace=True)
    # print(data1.shape)
    # print(data1.isna().sum())
    # start_time = datetime.now()
    # rmse1, mae1, predict_df1 = ols(data1, ind_var_at_paper, dep)
    # print(rmse1)
    # print(mae1)

    rmse_all1, mae_all1, mape_all1, adjrmse_all1, predict_df_all1, result1 = ols_all(data1, ind_var_at_paper, dep)
    # matched_quanti_and_qual_data_fin = add_one_hot(matched_quanti_and_qual_data_fin, '회계년')
    result1_1 = result1.get_robustcov_results()
    print(result1_1.summary())


    qual_dat = pd.read_excel('before_divide1and2.xlsx')
    TA_calc = pd.read_excel('./TA_calc.xlsx')  # 3번 조건. 2번은 이미 충족됨.
    TA_calc = q_replace_int(TA_calc)
    matched_quanti_and_qual_data_fin = pd.merge(left=matched_quanti_and_qual_data_fin, right=TA_calc[['Symbol', '회계년', '주기', 'CFO_ratio']].copy(),
                                            how='left', on=['Symbol', '회계년', '주기'], sort=False)
    matched_quanti_and_qual_data_fin['t1_M000911020_유효세율(%)'] = matched_quanti_and_qual_data_fin[
                                                                't1_M000911020_유효세율(%)'] / 100.0
    matched = pd.merge(left=qual_dat[['Symbol', '회계년', '주기', 'index']].copy(),
                       right=matched_quanti_and_qual_data_fin, how='left', on=['Symbol', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    data4 = matched[indVar_withoutFootnote].copy()
    data4.dropna(inplace=True)

    indVar_withFootnote = identifier.copy()
    indVar_withFootnote.append(main_ind_var)  # footnotes
    indVar_withFootnote.extend(ind_var_at_paper)
    indVar_withFootnote.extend(moyerin_ind_var)
    indVar_withFootnote.extend(inv_var_for_ctrl)  # optional for contorl
    # indVar_withFootnote.extend(['회계년_2013', '회계년_2014', '회계년_2015', '회계년_2016', '회계년_2017', '회계년_2018'])  # optional for contorl
    indVar_withFootnote.extend(ctrl_q)  # optional for contorl
    # indVar_withFootnote.extend(['주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q'])  # optional for contorl
    indVar_withFootnote.append(dep)
    # matched_quanti_and_qual_data.loc['년주기'] = str(
    #     matched_quanti_and_qual_data['회계년'] + str(matched_quanti_and_qual_data['주기'])

    # df = matched_quanti_and_qual_data_fin[indVar_withoutFootnote].copy()
    # df.dropna(inplace=True)
    # print(df.isna().sum())

    data2 = matched_quanti_and_qual_data_fin[indVar_withFootnote].copy()
    data2 = data2.dropna()
    # print(data2.shape)
    # print(data2.isna().sum())
    # print(data1.shape[1] == data2.shape[1]-1)
    # rmse2, mae2, predict_df2 = ols(data2, ind_var_at_paper, dep)
    # print(rmse2)
    # print(mae2)

    columns = []
    columns.append(main_ind_var)  # footnotes
    columns.extend(ind_var_at_paper)
    columns.extend(moyerin_ind_var)
    columns.extend(inv_var_for_ctrl)  # optional for contorl
    columns.extend(ctrl_q)
    # columns.extend(['주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q'])
    # columns.extend(['회계년_2013', '회계년_2014', '회계년_2015', '회계년_2016', '회계년_2017', '회계년_2018'])
    columns.remove('SIZE')  # 원본 연구에서도 유의하지 않아서 제거 #아마도 다중공선성으로 보임.
    columns.remove('BIG')  # 원본 연구에서도 유의하지 않아서 제거
    columns.remove('LARGE')  # 원본 연구에서도 유의하지 않아서 제거

    columns.remove('년주기_20163')  # 더미변수니까 하나 적당히 지움
    # columns.remove('주기_4Q')  # 더미변수니까 하나 적당히 지움
    columns.remove('ind_비금속 광물제품 제조업')  # 더미변수니까 하나 적당히 지움
    # columns.remove('t1_M000901006_매출채권(천원)')
    # columns.remove('t1_M000901012_재고자산(천원)')
    # columns.remove('t1_M_CAPEX')
    # columns.remove('t1_M000904007_매출총이익(천원)')
    columns.remove('M000909001_영업활동으로인한현금흐름(천원)')
    columns.remove('LEV')
    columns.remove('MTB')

    # columns.remove('t4_M000901006_매출채권(천원)')
    # columns.remove('t4_M000901012_재고자산(천원)')
    # columns.remove('t4_M_CAPEX')

    # columns.remove('t1_t_1M000601005_EPS(보통주)(원)')  # 당기순이익이라 ROA=당기순이익/총자산와 충돌
    # columns.remove('t2_t_1M000601005_EPS(보통주)(원)')
    # columns.remove('t3_t_1M000601005_EPS(보통주)(원)')
    # columns.remove('t4_t_1M000601005_EPS(보통주)(원)')
    columns.remove('ROA')

    columns.remove('tminus_inv_M000901001_총자산(천원)')  # 위의 3개와 중복되는 부분이 있다. Size와 상당히 유관.
    columns.remove('deltaREV_minus_deltaREC')  # 위의 3개와 중복되는 부분이 있다. 매출채권 delta는 AR과 LF_salesDivEmp와 겹치고
    columns.remove('div_A_M000901017_유형자산(천원)')  # 위의 3개와 중복되는 부분이 있다. M_CAPEX가 특히

    """
    rmse_all2, mae_all2, predict_df_all2, result2 = ols_all(data2, columns, dep)
    result4 = result2.get_robustcov_results()
    print(result4.summary())
    """
    data3 = matched.copy()
    data3 = data3.dropna()
    # data3.to_csv('for_gls.csv')
    columns3 = columns.copy()
    print('----------------data3')
    rmse_all3, mae_all3, predict_df_all3, result3 = ols_all(data3, columns3, dep)
    bp_test = het_breuschpagan(result3.resid, data3[columns3])
    labels = ['LM Statistic', 'LM - Test p - value', 'F - Statistic', 'F - Test p - value']
    print(dict(zip(labels, bp_test)))

    name = ['F statistic', 'p-value']
    test = het_goldfeldquandt(data3[dep], data3[columns3])
    print(dict(zip(name, test)))

    result5 = result3.get_robustcov_results()
    print(result5.summary())

    X = sm.add_constant(data3[columns3])
    # result3.resid.res
    # X.shape
    white_test = het_white(result3.resid, X)
    # labels = ['LM Statistic', 'LM - Test p - value', 'F - Statistic', 'F - Test p - value']
    # print(dict(zip(labels, white_test)))
    #
    rmse_all4, mae_all4, predict_df_all4, result4 = ols_all(data3, ind_var_at_paper, dep)
    result4_1 = result4.get_robustcov_results()
    print(result4_1.summary())


    # ridge_reg = Ridge(alpha=1, solver="cholesky")  # 쇼레스키 분해 용법 사용
    # res = ridge_reg.fit(data3[columns3], data3[dep])


    glm_res = sm.GLM(data3[dep].values, X).fit()
    print(glm_res.summary())
    r_glm_res = glm_res.get_robustcov_results()
    # X = sm.add_constant(data2[columns3].values)
    # glm_res = sm.GLM(data2[dep].values, X).fit()
    # print(glm_res.summary())

    # plt.scatter(data3[main_ind_var], result3.resid)
    # plt.show()

    col = columns.copy()
    inv_var_for_ctrl_set = set(inv_var_for_ctrl)  # optional for contorl
    ctrl_q_set = set(ctrl_q)
    list1 = [ele for ele in col if ele not in inv_var_for_ctrl_set]
    list2 = [ele for ele in list1 if ele not in ctrl_q_set]
    # columns.extend(inv_var_for_ctrl)  # optional for contorl
    # columns.extend(ctrl_q)
    vif = pd.DataFrame()
    # vif["VIF Factor"] = [variance_inflation_factor(data2.loc[:, ind_var_at_paper].values, i) for i in range(data2.loc[:, ind_var_at_paper].shape[1])]
    vif["VIF Factor"] = [variance_inflation_factor(data3.loc[:, list2].values, i) for i in range(data3.loc[:, list2].shape[1])]
    vif["features"] = list2
    print(vif)

    sns.set(style="whitegrid")


    # np.random.seed(42)  # for placebo
    # data3 = data2.copy()
    # data3[main_ind_var] = np.random.uniform(0, 2, size=data3.shape[0])  # for placebo
    # print(data3.shape)
    # print(data3.isna().sum())
    # print(data3.shape[1] == data2.shape[1])
    # rmse3, mae3, predict_df3 = ols(data3, ind_var_at_paper, dep)
    # print(rmse3)
    # print(mae3)
    # rmse_all3, mae_all3, predict_df_all3 = ols_all(data3, ind_var_at_paper, dep)


    # print('ols All result')
    # print(rmse_all1)
    # print(mae_all1)
    # print(rmse_all2)
    # print(mae_all2)
    # print(rmse_all3)
    # print(mae_all3)

    # rmse1_avg = mean_per_n(rmse1, 3)
    # mae1_avg = mean_per_n(mae1, 3)
    # rmse2_avg = mean_per_n(rmse2, 3)
    # mae2_avg = mean_per_n(mae2, 3)
    # rmse3_avg = mean_per_n(rmse3, 3)
    # mae3_avg = mean_per_n(mae3, 3)

    # print('rmse check-------------------------------------------------')
    # result1 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse2)  # 독립 t-test 단방향 검정
    # print('placebo check-------------------------------------------------')
    # result2 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse3)  # placebo 효과 확인.
    # print('mae check-------------------------------------------------')
    # result3 = pfd.equ_var_test_and_unpaired_t_test(mae1, mae2)  # 독립 t-test 단방향 검정
    # print('placebo check-------------------------------------------------')
    # result4 = pfd.equ_var_test_and_unpaired_t_test(mae1, mae3)  # placebo 효과 확인.
