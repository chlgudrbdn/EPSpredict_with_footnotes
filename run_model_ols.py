import preprocess_footnotes_data as pfd
import pandas as pd
import join_pickle_data as jpd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from numpy import dot
from numpy.linalg import norm
import numpy as np
from scipy import sparse
import statsmodels.api as sm
import gzip
import pickle
import multiprocessing
from scipy import spatial
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


def adjusted_rmse(y_true, y_pred):  # 사실 2009년엔 없는 단어여서 그런지 모르겠지만 이걸 Mean Square Percentage Error 라고 함. mape보다 인기는 없어보인다.
    # return np.sqrt((((targets - predictions)/targets) ** 2).mean())
    return np.square(((y_true - y_pred)/y_true)).mean() * 100


def Mape(y_true, y_pred):  # 0에 가까운 값에 약해(규모에 민감)서 사실 그닥 성능은 좋지 않아보인다.
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Mpe(y_true, y_pred):  # underperformance 인지 overperformance 인지 판단 할 수 있다는 것입니다.
    return np.mean((y_true - y_pred) / y_true) * 100


def rearrange_calender_data(df):  # 전처리 위한 코드. eps 변화율 데이터에 적자로 전환, 흑자 유지 같은 식으로 나와서 수정 eps를 그대로 계산.
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , dtype=object, sheet_name='주식수')
    df.drop(columns=['Kind', 'Item', 'Item Name ', 'Frequency'], inplace=True)
    arrange_df = pd.DataFrame(columns=['Symbol', 'Name', '결산월', '회계년', '주기', '상장주식수_보통주'])
    i = 0
    col_names = df.columns[2:]  # 행에 회사 열에 분기말 날짜.
    for index, row in df.iterrows():
        for col in col_names:  # 한 줄 내려가고 오른쪽으로 이동하며 하나씩 쌓는다.
            arrange_df.loc[i] = [row.Symbol, row['Symbol Name'], col.month, col.year, str(int(col.month/3))+"Q", row[col]]
            i += 1
    print(df.shape)
    print(arrange_df.shape)
    # set_diff_df = pd.concat([arrange_df[(arrange_df['회계년']==2019)&(arrange_df['결산월']==9)]
    #                             , arrange_df]).drop_duplicates(keep=False)
    # print(set_diff_df.shape[0]/df.shape[0] == (4*len([2013,2014,2015,2016,2017,2018])+2))
    print(arrange_df.shape[0]/df.shape[0] == (4*len([2013,2014,2015,2016,2017,2018])+3))
    # set_diff_df.to_csv('price.csv', encoding='cp949')
    arrange_df.to_csv('rearrange.csv', encoding='cp949')


def match_tplus_data(quanti_ind_var, dep_vars, dep_var, ind_var_list, file_name):
    identifier = ['Symbol', 'Name', '결산월', '회계년', '주기']
    dep_vars = pd.concat([dep_vars.loc[:, identifier],
                         dep_vars.loc[:, dep_var]], axis=1)  # 일단 폼은 유지하되 필요한 변수 하나만 떼다 쓰기 위함.
    dep_vars.dropna(thresh=len(list(dep_vars.columns)) - len(identifier), inplace=True)  # 식별 위한 정보를 제외한 것이 하나도 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    identifier.extend(ind_var_list)
    quanti_ind_var = quanti_ind_var[identifier]
    print(quanti_ind_var.shape)
    quanti_ind_var = quanti_ind_var.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까.
    quanti_ind_var.dropna(thresh=len(list(dep_vars.columns)) - len(identifier), inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(quanti_ind_var.columns)
    print('quanti_ind_var.info() : ', quanti_ind_var.shape)
    for index, row in quanti_ind_var.iterrows():
        tplus_quarter = str(int(row['주기'][0])+1)+'Q'   # 결산월 보다 쿼터 쪽이 신뢰도 있음
        if tplus_quarter == '5Q':  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
            tplus_year = int(row['회계년']) + 1   # 결산월 보다 쿼터 쪽이 신뢰도 있음
            tplus_quarter = '1Q'
        else:
            tplus_year = int(row['회계년'])
        tplus_data = dep_vars[(dep_vars['Symbol'] == row['Symbol'])
                              & (dep_vars['회계년'] == tplus_year)
                              & (dep_vars['주기'] == tplus_quarter)]
        if tplus_data.shape[0] > 1:  # 중복된 분기의 데이터 없는 문제 확인.
            print('tplus_data :', tplus_data)  # 일단 미리 없애놨으니 나타날린 없지만 그래도 체크.
        if tplus_data.empty:
            quanti_ind_var.loc[index, dep_var] = np.nan
            print('empty', index)
            # print(row)
            continue
        # result_df = result_df.append(tplus_data, ignore_index=True)  # 찾은 결과를 한줄씩 붙인뒤 나중에 옆으로 붙일 예정.
        # print('tplus_data : ', tplus_data.values)
        # print('tplus_data : ', tplus_data[dep_var].iloc[0])
        # print('tplus_data : ', tplus_data.loc[:, dep_var].values[0].tolist())
        # print('tplus_data : ', int(tplus_data.loc[:, '수정PER3분할_10y_20p']))
        for dep in dep_var:
            quanti_ind_var.loc[index, dep] = tplus_data.iloc[0][dep]  # t+1 분기와 매칭
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
    quanti_ind_var.to_pickle(directory_name+'/'+file_name)
    print(quanti_ind_var.columns)
    print(quanti_ind_var.shape)
    print(quanti_ind_var.info())

    return quanti_ind_var  # 다음에 쓰려고 ndarray로 반환하지 않음.


def check_difference(a, b):
    if np.isnan(a) and np.isnan(b):  # 둘다 비어서 같은건 그냥 사용할 수 없는 데이터로 취급해야한다.
        return 'no_dat'

    if a == b:
        return ""
    else:
        return 'diff'


def get_change_from_timus_to_t(df, check_var_name, var_name):  # 전처리 위한 코드. eps 변화율 데이터에 적자로 전환, 흑자 유지 같은 식으로 나와서 수정 eps를 그대로 계산.
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                       , dtype=object, sheet_name='주식수가 변치 않은 기업 리스트')  # 'price_rearrange.csv'
    check_var_name = 'cnt_change'
    var_name = '상장주식수_보통주'
    cd_list = list(df['Symbol'].unique())
    df[check_var_name] = ""
    for cd in cd_list:
        for year in range(2013, 2019):
            for quarter in range(1, 5):
                tminus_quarter = quarter - 1  # 결산월 보다 쿼터 쪽이 신뢰도 있음
                quarter = str(quarter) + 'Q'
                tminus_year = year
                if tminus_quarter == 0:  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
                    tminus_year = year - 1  # 결산월 보다 쿼터 쪽이 신뢰도 있음
                    tminus_quarter = '4Q'
                else:
                    tminus_quarter = str(tminus_quarter) + 'Q'

                tminus_data = df[(df['Symbol'] == cd) & (df['회계년'] == tminus_year) & (df['주기'] == tminus_quarter)]
                t_data = df[(df['Symbol'] == cd) & (df['회계년'] == year) & (df['주기'] == quarter)]
                df.loc[t_data.index, check_var_name] = check_difference(tminus_data.iloc[0][var_name], t_data.iloc[0][var_name])
    df.to_csv('change_check.csv', encoding='cp949')


def match_quanti_and_qual_data(qual_ind_var, quanti_ind_var, file_name):  # 정량 독립 및 정량 종속 변수와 정성 독립 변수 매칭
    result_df = pd.DataFrame()
    valid_df_idx_list = []
    for index, row in qual_ind_var.iterrows():
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
        tplus_data = quanti_ind_var[(quanti_ind_var['Symbol'] == 'A'+str(row['crp_cd'])) &
                                 # (quanti_data['결산월'] == t_closing_date.month) &
                                 (quanti_ind_var['주기'] == t_quarter) &
                                 (quanti_ind_var['회계년'] == t_year)]
        if tplus_data.shape[0] > 1:
            print('duplicated ', index)  # 없겠지만 중복이 생길 경우 예외처리를 위함.

        if tplus_data.empty:
            print('empty ', index)  # 약 2260건. 필연적으로 어딘가 비면 생길 수 밖에 없는 문제다.
            result_df = result_df.append(pd.Series(), ignore_index=True)
            continue
        valid_df_idx_list.append(index)  # 최종적으로는 매칭에 이것만 있으면 된다. # 일단 적절한 값이 없는 경우 알아서 생략되도록 앞의 코드에서 처리.
        result_df = result_df.append(tplus_data, ignore_index=True)
    qual_ind_var.reset_index(drop=True, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    matched_quanti_and_qual_data = pd.concat([qual_ind_var, result_df], axis=1)

    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    matched_quanti_and_qual_data.to_pickle(directory_name+'/'+file_name)
    print(len(valid_df_idx_list))
    print(matched_quanti_and_qual_data.shape)
    # np.save('./merged_FnGuide/'+file_name, df.values)
    # for test
    # directory_name = 'merged_FnGuide'
    # file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
    # df = pd.read_pickle(directory_name+'/merged_FnGuide '+file_name)
    # print(df.shape)
    # main_dat = df.loc[:, df.columns.str.contains('^M')]
    # df = df.dropna(thresh=len(df.columns) - len(main_dat.columns)+1)  # 종속변수가 모두 nan이려면 non-NaN인 값이 5보다 적은 경우이다.
    # print(main_dat.dropna(how='all').shape)
    # print(df.shape)
    # df.to_pickle(directory_name+'/merged_FnGuide '+file_name)
    # print(df1.loc[0])
    # for test
    return matched_quanti_and_qual_data


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
    result2 = result.get_robustcov_results()
    print('robust!')
    print(result2.summary())
    predictY = result.predict(X)
    rmse = sqrt(mean_squared_error(y, predictY))
    print('rmse: ', rmse)
    adjrmse = adjusted_rmse(y, predictY)
    print('adjusted_rmse: ', adjrmse)
    mae = mean_absolute_error(y, predictY)
    print('mae: ', mae)
    mape = Mape(y, predictY)
    print('mape: ', mape)
    mpe = Mpe(y, predictY)
    if mpe < 0:
        print(mpe, 'over perform')
    else:
        print(mpe, 'under perform')
    predict_df['true'] = pd.Series(list(y))
    predict_df['pred'] = pd.Series(list(predictY))
    return rmse, adjrmse, mae, mape, predict_df


def q_replace_int(df):
    df.loc[df['주기'] == '1Q', '주기'] = 1
    df.loc[df['주기'] == '2Q', '주기'] = 2
    df.loc[df['주기'] == '3Q', '주기'] = 3
    df.loc[df['주기'] == '4Q', '주기'] = 4
    return df



if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    path_dir = os.getcwd()
    dep_var = ['dep_M000601002_EPS(원)', 'dep_M000601005_EPS(보통주)(원)', 'dep_M000601003_수정EPS(원)', 'dep_M000601006_수정EPS(보통주)(원)']  # 쓸걸로 예상
    ind_var_list = ['M000909001_영업활동으로인한현금흐름(천원)', 'M000909018_재무활동으로인한현금흐름(천원)', 'M000909015_투자활동으로인한현금흐름(천원)',
                    # '주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q',
                    # 'ind_1차 금속 제조업', 'ind_가구 제조업', 'ind_가죽, 가방 및 신발 제조업',
                    # 'ind_건축 기술, 엔지니어링 및 기타 과학기술 서비스업', 'ind_고무 및 플라스틱제품 제조업', 'ind_교육 서비스업',
                    # 'ind_금속 가공제품 제조업; 기계 및 가구 제외', 'ind_기타 개인 서비스업', 'ind_기타 기계 및 장비 제조업',
                    # 'ind_기타 운송장비 제조업', 'ind_기타 전문, 과학 및 기술 서비스업', 'ind_기타 제품 제조업', 'ind_농업',
                    # 'ind_담배 제조업', 'ind_도매 및 상품 중개업', 'ind_목재 및 나무제품 제조업; 가구 제외', 'ind_방송업',
                    # 'ind_부동산업', 'ind_비금속 광물제품 제조업', 'ind_비금속광물 광업; 연료용 제외',
                    # 'ind_사업 지원 서비스업', 'ind_사업시설 관리 및 조경 서비스업', 'ind_석탄, 원유 및 천연가스 광업',
                    # 'ind_섬유제품 제조업; 의복 제외', 'ind_소매업; 자동차 제외', 'ind_수상 운송업', 'ind_숙박업',
                    # 'ind_스포츠 및 오락관련 서비스업', 'ind_식료품 제조업', 'ind_어업', 'ind_연구개발업',
                    # 'ind_영상ㆍ오디오 기록물 제작 및 배급업', 'ind_우편 및 통신업', 'ind_육상 운송 및 파이프라인 운송업',
                    # 'ind_음료 제조업', 'ind_음식점 및 주점업', 'ind_의료, 정밀, 광학 기기 및 시계 제조업',
                    # 'ind_의료용 물질 및 의약품 제조업', 'ind_의복, 의복 액세서리 및 모피제품 제조업',
                    # 'ind_인쇄 및 기록매체 복제업', 'ind_임대업; 부동산 제외', 'ind_자동차 및 부품 판매업',
                    # 'ind_자동차 및 트레일러 제조업', 'ind_전기, 가스, 증기 및 공기 조절 공급업', 'ind_전기장비 제조업',
                    # 'ind_전문 서비스업', 'ind_전문직별 공사업', 'ind_전자 부품, 컴퓨터, 영상, 음향 및 통신장비 제조업',
                    # 'ind_정보서비스업', 'ind_종합 건설업', 'ind_창고 및 운송관련 서비스업',
                    # 'ind_창작, 예술 및 여가관련 서비스업', 'ind_출판업', 'ind_컴퓨터 프로그래밍, 시스템 통합 및 관리업',
                    # 'ind_코크스, 연탄 및 석유정제품 제조업', 'ind_펄프, 종이 및 종이제품 제조업',
                    # 'ind_폐기물 수집, 운반, 처리 및 원료 재생업', 'ind_항공 운송업',
                    # 'ind_화학 물질 및 화학제품 제조업; 의약품 제외', 'ind_환경 정화 및 복원업',
                    # '상장주식수_보통주', 'cnt_change'
                    ]  # 여기에 사용할 eps를 더 추가
    quanti_data_set_file_name = 'quanti_ols_hwang.xlsx'  # 오로지 정량 데이터(종속, 독립 둘다) 만 정리된 데이터.
    dep = dep_var[1]

    """    """
    # dep_vars = pd.read_excel(path_dir+'\\financial ratio for dependent variable\\EPS_rawData.xlsx'
    #                         , dtype=object, sheet_name='Sheet1')  # 종속변수
    # df8 = pd.read_pickle(path_dir + '/divide_by_sector/filter8 komoran_for_cosine_distance.pkl')  # 독립변수중 산업이나 코사인거리
    # quanti_ind_var = pd.read_excel(path_dir+'\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
    #                                , dtype=object, sheet_name='Sheet1')
    # cnt = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx', dtype=object, sheet_name='주식수가 변치 않은 기업 리스트')
    # quanti_ind_var = pd.merge(left=quanti_ind_var, right=dep_vars, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    # quanti_ind_var.to_csv('eps_ind.csv', encoding='cp949')
    # matched_quanti_data = match_tplus_data(quanti_ind_var, dep_vars, dep_var, ind_var_list, quanti_data_set_file_name)
    # dep_vars[matched_quanti_and_qual_data.columns.str.contains('(직전4분기)')].replace('N/A(IFRS)', np.nan, inplace=True)

    # dep_vars = dep_vars[['Symbol', 'Name', '결산월', '회계년', '주기'].extend(dep_var)]
    # dep_vars.columns = ['Symbol', 'Name', '결산월', '회계년', '주기',
    #                     't_M000601002_EPS(원)', 't_M000601042_EPS(계속사업)(원)', 't_M000601005_EPS(보통주)(원)', 't_M000601004_EPS(직전4분기)(원)',
    #                     't_M000601003_수정EPS(원)', 't_M000601043_수정EPS(계속사업)(원)', 't_M000601006_수정EPS(보통주)(원)', 't_M000601048_수정EPS(직전4분기)(원)']
    # matched_quanti_data = pd.merge(left=matched_quanti_data, right=dep_vars, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    matched_quanti_data = pd.read_excel('./'+quanti_data_set_file_name, sheet_name='Sheet1')
    matched_quanti_and_qual_data = matched_quanti_data.copy()
    # matched_quanti_and_qual_data = match_quanti_and_qual_data(df8, matched_quanti_data, quanti_qual_matched_file_name)
    # matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, 'crp_cls')
    # matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, '주기')
    # matched_quanti_and_qual_data = pfd.add_one_hot_with_ind_cd(matched_quanti_and_qual_data)  # 세분화된 산업코드 사용

    # columns = ['crp_cd', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt', 't_minus_index', 't_minus_year_index',
    #            'Name', '결산월_x', '결산월_y']  # Symbol, 회계년, 주기는 차후 식별등을 위해 일단 남긴다.
    # matched_quanti_and_qual_data.drop(columns, inplace=True, axis=1)

    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].replace("", np.nan)
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')

    # matched_quanti_and_qual_data['t_1y_cos_dist'] = matched_quanti_and_qual_data['t_1y_cos_dist'].replace("", np.nan)
    # matched_quanti_and_qual_data['t_1y_cos_dist'] = matched_quanti_and_qual_data['t_1y_cos_dist'].astype('float')

    # ind_cosine_dict = {}
    # for ind_col_name in list(matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^ind')]):
    #     print(ind_col_name)
    #     avg_q_dist = matched_quanti_and_qual_data[(matched_quanti_and_qual_data[ind_col_name] == 1)]['t_1q_cos_dist'].mean()
    #     avg_y_dist = matched_quanti_and_qual_data[(matched_quanti_and_qual_data[ind_col_name] == 1)]['t_1y_cos_dist'].mean()
    #     ind_cosine_dict[ind_col_name] = [avg_q_dist, avg_y_dist]

    # cols = matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^q_cos_ind')]
    # matched_quanti_and_qual_data.drop(cols, inplace=True, axis=1)
    # cols = matched_quanti_and_qual_data.columns[matched_quanti_and_qual_data.columns.str.contains('^y_cos_ind')]
    # matched_quanti_and_qual_data.drop(cols, inplace=True, axis=1)

    # for key in ind_cosine_dict:
        # key = 'ind_항공 운송업'  # for test
        # matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data[key] == 1), "q_cos_mean"] = ind_cosine_dict[key][0]
        # matched_quanti_and_qual_data.loc[(matched_quanti_and_qual_data[key] == 1), "y_cos_mean"] = ind_cosine_dict[key][1]
    # matched_quanti_and_qual_data.dropna(inplace=True)
    # matched_quanti_and_qual_data['diff_q_cos_per_ind'] = matched_quanti_and_qual_data['t_1q_cos_dist'] - matched_quanti_and_qual_data['q_cos_mean']
    # matched_quanti_and_qual_data['diff_y_cos_per_ind'] = matched_quanti_and_qual_data['t_1y_cos_dist'] - matched_quanti_and_qual_data['y_cos_mean']
    # matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/' + quanti_qual_matched_file_name)

    # matched_quanti_and_qual_data.drop(['t_1y_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data.drop(['t_1q_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')

    # matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/'+quanti_qual_matched_file_name)
    # matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/diffPerInd_' + quanti_qual_matched_file_name)

    # main_ind_var = 't_1q_cos_dist'
    # main_ind_var = 't_1y_cos_dist'
    # main_ind_var = 'diff_q_cos_per_ind'  # 산업 평균 대비 분기 cos거리
    # main_ind_var = 'diff_y_cos_per_ind'  # 산업 평균 대비 연 cos거리

    # ['M000601002_EPS(원)', 'M000601042_EPS(계속사업)(원)',	'M000601005_EPS(보통주)(원)', 'M000601004_EPS(직전4분기)(원)',
    # 'M000601003_수정EPS(원)', 'M000601043_수정EPS(계속사업)(원)', 'M000601006_수정EPS(보통주)(원)', 'M000601048_수정EPS(직전4분기)(원)']
    # dep = dep_var[0]
    # dep = dep_var[1]


    merge_log = pd.read_excel(path_dir+'/합병_내역.xlsx', dtype=object, sheet_name='crp_cd_merge_happend')  # 3번 조건. 2번은 이미 충족됨.
    merge_crp = ['A'+str(x) for x in list(merge_log['거래소코드'].unique())]
    matched_quanti_and_qual_data_without_merge = matched_quanti_and_qual_data.loc[~matched_quanti_and_qual_data['Symbol'].isin(merge_crp)].copy()

    matched_quanti_and_qual_data_without_merge[dep].replace('N/A(IFRS)', np.nan, inplace=True)
    matched_quanti_and_qual_data_without_merge[dep] = matched_quanti_and_qual_data_without_merge[dep].astype('float')

    ## eps 0 이상인 기업만 포함. 시작
    matched_quanti_and_qual_data_without_minus_eps = matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data[dep] > 0, :]  # 1번 조건
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

    no_diff_crp = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , dtype=object, sheet_name='no_diff_crp')
    no_diff_crp_list = list(no_diff_crp['Symbol'].unique())
    matched_quanti_and_qual_data_without_cnt_change = matched_quanti_and_qual_data_without_minus_eps.loc[matched_quanti_and_qual_data_without_minus_eps['Symbol'].isin(no_diff_crp_list)]
    # matched_quanti_and_qual_data_without_cnt_change.drop(columns=['cnt_change'], inplace=True)
    ## 유상증자가 없는 기업만 포함 끝.


    # matched_quanti_and_qual_data_without_merge[dep].replace('N/A(IFRS)', np.nan, inplace=True)
    # matched_quanti_and_qual_data_without_merge[dep] = matched_quanti_and_qual_data_without_merge[dep].astype('float')
    matched_quanti_and_qual_data_without_minus_eps = matched_quanti_and_qual_data_without_merge.loc[matched_quanti_and_qual_data_without_merge[dep] > 0, :]  # 1번 조건

    # matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data_without_cnt_change[matched_quanti_and_qual_data_without_cnt_change[main_ind_var] != np.nan]  # 미리 걸러서 별 의미는 없음.

    match_ind_crp = pd.read_excel('./match_ind_crp.xlsx')
    matched_quanti_and_qual_data_without_finance = pd.merge(left=matched_quanti_and_qual_data_without_minus_eps, right=match_ind_crp[['Symbol', '한국표준산업분류코드10차(대분류)']].copy(),
                                                            how='left', on=['Symbol'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    matched_quanti_and_qual_data_without_finance = matched_quanti_and_qual_data_without_finance[(matched_quanti_and_qual_data_without_finance['한국표준산업분류코드10차(대분류)'] !='K')]
    matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data_without_finance.copy()
    # ind_var_list.append('t_'+dep)
    matched_quanti_and_qual_data_fin.dropna(inplace=True)
    matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data_fin[(matched_quanti_and_qual_data_fin['회계년'] < 2019)
                                                                        & (matched_quanti_and_qual_data_fin['회계년'] > 2012)]
    print(matched_quanti_and_qual_data_fin.dtypes)
    print(matched_quanti_and_qual_data_fin.shape)


    # matched_quanti_and_qual_data_fin['t_'+dep] = matched_quanti_and_qual_data['t_'+dep].astype('float')

    # var_list_without_footnote.pop(-1)
    # var_list_with_footnote.pop(-1)
    var_list_without_footnote = ind_var_list.copy()
    # var_list_with_footnote = ind_var_list.copy()

    # var_list_with_footnote.append(main_ind_var)

    # var   _list_without_footnote.append(dep)
    # var_list_with_footnote.append(dep)
    qual_dat = pd.read_excel('before_divide1and2.xlsx')
    matched_quanti_and_qual_data_fin = q_replace_int(matched_quanti_and_qual_data_fin)

    matched = pd.merge(left=matched_quanti_and_qual_data_fin, right=qual_dat[['Symbol', '회계년', '주기', 't_1q_cos_dist']].copy(),
                       how='inner', on=['Symbol', '회계년', '주기'],
             sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.

    rmse_all1, mspe_all1, mae_all1, mape_all1, predict_df_all1 = ols_all(matched_quanti_and_qual_data_fin, ind_var_list, dep, True)
    rmse_all2, mspe_all2, mae_all2, mape_all2, predict_df_all2 = ols_all(matched, ind_var_list, dep, True)
    print(matched.shape)

    # rlm_model = sm.RLM(matched[dep], matched[ind_var_list], M=sm.robust.norms.HuberT())
    # rlm_results = rlm_model.fit()
    # rlm_results.summary()
    # data1 = matched_quanti_and_qual_data_fin[var_list_without_footnote]
    # data1 = data1.dropna()
    # rmse1, mape1 = ols(data1, ind_var_list, dep)
    #
    # data2 = matched_quanti_and_qual_data_fin[var_list_with_footnote]
    # data2 = data2.dropna()
    # ind_var_list.append(main_ind_var)
    # rmse2, mape2 = ols(data2, ind_var_list, dep)
    #
    # np.random.seed(10)  # for placebo
    # data3 = data2
    # data3[main_ind_var] = np.random.rand(data3.shape[0])  # for placebo
    # rmse3, mape3 = ols(data3, ind_var_list, dep)
    # matched_quanti_and_qual_data[main_ind_var] = matched_quanti_and_qual_data[main_ind_var].astype('float')
    # matched_quanti_and_qual_data[dep_var] = matched_quanti_and_qual_data[dep_var].astype('int8')

    # print(matched_quanti_and_qual_data.info())
