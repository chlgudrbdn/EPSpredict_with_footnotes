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
import preprocess_footnotes_data as pfd


def tf_idf_prerocess(foot_note_col):
    tf = TfidfVectorizer(max_df=0.95, min_df=0)
    tfidf_matrix = tf.fit_transform(foot_note_col)
    return tfidf_matrix


def make_category_by_quartile_with_10_year(df, col_name):
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   # , dtype=object, sheet_name='Sheet1')  # for test
                                   , dtype=object, sheet_name='PER_before_10 year')  # for test
    col_name = 'M수정PER'  # for test
    cd_list = list(df['Symbol'].unique())
    # per_categorize = []
    for cd in cd_list:
        # cd = cd_list[0]  # for test
        for year in range(2013, 2019):
            for quarter in range(1, 5):
                q = str(quarter)+'Q'
                end_index = df[(df['Symbol'] == cd) & (df['회계년'] == year) & (df['주기'] == q)].index[0]
                # tmp = df.iloc[end_index-40: end_index+1]
                tmp = df.loc[end_index-40: end_index, :]
                tmp = tmp.replace(0, np.nan)  # 이상한 경우.
                # tmp['q_cat'] = pd.qcut(tmp[col_name],  [0, .2, .8, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
                # tmp['q_cat'] = pd.qcut(tmp[col_name],  [0, .25, .75, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
                tmp['q_cat'] = pd.qcut(tmp[col_name],  [0, .33, .66, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
                # df.loc[end_index, '수정PER3분할_10y_20p'] = tmp.loc[end_index, 'q_cat']
                # df.loc[end_index, '수정PER3분할_10y_25p'] = tmp.loc[end_index, 'q_cat']
                df.loc[end_index, '수정PER3분할_10y_33p'] = tmp.loc[end_index, 'q_cat']
    # per_categorize.extend(list(q_cat))
    # df.loc['수정PER3분할'] = per_categorize
    # df = pd.concat([df, pd.concat(per_categorize).rename('수정PER계속사업3분할')], axis=1)
    # df.to_csv('PER_categorization.csv', encoding='cp949')
    # df.to_csv('PER_categorization_with_10y_20.csv', encoding='cp949')
    # df.to_csv('PER_categorization_with_10y_25.csv', encoding='cp949')
    df.to_csv('PER_categorization_with_10y_33.csv', encoding='cp949')


def make_category_by_quartile_with_per_quarter(df, col_name):  # 동기 PER 분위
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   # , dtype=object, sheet_name='Sheet1')  # for test
                                   , dtype=object, sheet_name='PER_before_10 year')  # for test
    col_name = 'M수정PER'  # for test
    df_list = []
    for year in range(2013, 2019):
        for quarter in range(1, 5):
            q = str(quarter)+'Q'
            tmp = df[(df['회계년'] == year) & (df['주기'] == q)]
            tmp.loc[:, col_name] = tmp[col_name].replace(0, np.nan)  # PER가 0인 이상한 경우.
            # tmp['수정PER3분할_1q_20p'] = pd.qcut(tmp[col_name],  [0, .2, .8, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
            # tmp['수정PER3분할_1q_25p'] = pd.qcut(tmp[col_name],  [0, .25, .75, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
            tmp['수정PER3분할_1q_33p'] = pd.qcut(tmp[col_name],  [0, .33, .66, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
            df_list.append(tmp)
    # per_categorize.extend(list(q_cat))
    # df.loc['수정PER3분할'] = per_categorize
    # df = pd.concat([df, pd.concat(per_categorize).rename('수정PER계속사업3분할')], axis=1)
    # df.to_csv('PER_categorization.csv', encoding='cp949')
    df = pd.concat(df_list, axis=0, ignore_index=False, sort=False)
    # df.to_csv('PER_categorization_with_1q_20.csv', encoding='cp949')
    # df.to_csv('PER_categorization_with_1q_25.csv', encoding='cp949')
    df.to_csv('PER_categorization_with_1q_33.csv', encoding='cp949')


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote/divide_by_sector'
    # 특정 형태소만 쓰는건 tf-idf로 바꿀때만 유용
    """    """
    df7 = pd.read_pickle(path_dir + '/filter7 komoran_filtered_pos.pkl')
    print(df7.shape)
    df7.sort_values(['crp_cd', 'rcp_dt'], ascending=['True', 'True'], inplace=True)
    result_df = pd.DataFrame()
    valid_df_idx_list = []


    quanti_data_set_file_name = './merged_FnGuide/quanti_per_dataset.pkl'
    quanti_ind_var = pd.read_pickle(quanti_data_set_file_name)


    for index, row in df7.iterrows():
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
            # print('exeception idx:', index, ' month:', t_month, ' rpt_nm:', rpt_nm)
            t_quarter = np.nan
            result_df = result_df.append(pd.Series(), ignore_index=True)
            valid_df_idx_list.append(index)  # 최종적으로는 매칭에 이것만 있으면 된다. # 일단 적절한 값이 없는 경우 알아서 생략되도록 앞의 코드에서 처리.
            continue
    print(len(valid_df_idx_list))
    valid_idx = set(df7.index) - set(valid_df_idx_list)
    valid = df7.loc[valid_idx, :]
    print(valid.shape)
    print(df7.shape)
    print(df7.shape[0] - valid.shape[0])
    valid.reset_index(drop=True, inplace=True)
    # 이전 분기와 매칭
    for index, row in valid.iterrows():
        t_rpt_nm = row['rpt_nm']
        t_closing_date = t_rpt_nm[t_rpt_nm.find("(") + 1:t_rpt_nm.find(")")].split('.')
        t_year = int(t_closing_date[0])
        t_month = int(t_closing_date[1])
        t_raw_rpt_nm = t_rpt_nm.split()[0]
        if t_month == 3:
            t_minus_year = t_year-1
            t_minus_month = 12
            t_minus_raw_rpt_nm = '사업보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        elif t_month == 6:
            t_minus_year = t_year
            t_minus_month = 3
            t_minus_raw_rpt_nm = '분기보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        elif t_month == 9:
            t_minus_year = t_year
            t_minus_month = 6
            t_minus_raw_rpt_nm = '반기보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        elif t_month == 12:
            t_minus_year = t_year
            t_minus_month = 9
            t_minus_raw_rpt_nm = '분기보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        else:
            print('something is wrong ', row)

        try:
            valid.loc[index, 't_minus_index'] = valid[(valid.crp_cd == row.crp_cd) &
                                                      (valid.rpt_nm == t_minus_rpt_nm)].index[0]
        except Exception as e:
            print(e)
            valid.loc[index, 't_minus_index'] = np.nan

    ## 이전 년도 동일 문서와 매칭
    for index, row in valid.iterrows():
        t_rpt_nm = row['rpt_nm']
        t_closing_date = t_rpt_nm[t_rpt_nm.find("(") + 1:t_rpt_nm.find(")")].split('.')
        t_year = int(t_closing_date[0])
        t_month = int(t_closing_date[1])
        t_raw_rpt_nm = t_rpt_nm.split()[0]

        if t_month == 3:
            t_minus_year = t_year-1
            t_minus_month = t_month
            t_minus_raw_rpt_nm = '분기보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        elif t_month == 6:
            t_minus_year = t_year-1
            t_minus_month = t_month
            t_minus_raw_rpt_nm = '반기보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        elif t_month == 9:
            t_minus_year = t_year-1
            t_minus_month = t_month
            t_minus_raw_rpt_nm = '분기보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        elif t_month == 12:
            t_minus_year = t_year-1
            t_minus_month = t_month
            t_minus_raw_rpt_nm = '사업보고서'
            t_minus_rpt_nm = t_minus_raw_rpt_nm+' ('+str(t_minus_year)+'.'+'{:02d}'.format(t_minus_month)+')'
        else:
            print('something is wrong ', t_month)

        try:
            valid.loc[index, 't_minus_year_index'] = valid[(valid.crp_cd == row.crp_cd) &
                                                      (valid.rpt_nm == t_minus_rpt_nm)].index[0]
        except Exception as e:
            print(e)
            # print('something is wrong ', row)
            valid.loc[index, 't_minus_year_index'] = np.nan

    ### make cosine distance with komoran ###
    foot_note_tf_idf = tf_idf_prerocess(valid['foot_note'])
    for index in valid.index:
        t_1q_index = valid.loc[index, 't_minus_index']
        t_1y_index = valid.loc[index, 't_minus_year_index']
        t_1q_result = np.nan
        t_1y_result = np.nan
        if not np.isnana(t_1q_index):
            t_1q_result = spatial.distance.cosine(foot_note_tf_idf[t_1q_index, :].todense(), foot_note_tf_idf[index, :].todense())
            # t_1q_result = cos_distance(foot_note_tf_idf[t_1q_index, :].todense(), foot_note_tf_idf[index, :].todense())
        if not np.isnana(t_1y_index):
            t_1y_result = spatial.distance.cosine(foot_note_tf_idf[t_1y_index, :].todense(), foot_note_tf_idf[index, :].todense())
            # t_1y_result = cos_distance(foot_note_tf_idf[t_1y_index, :].todense(), foot_note_tf_idf[index, :].todense())
        valid.loc[index, 't_1q_cos_dist'] = t_1q_result
        valid.loc[index, 't_1y_cos_dist'] = t_1y_result
    sparse.save_npz('./merged_FnGuide/tfidf_valid.npz', foot_note_tf_idf)
    valid.to_pickle(path_dir + '/filter8 komoran_for_cosine_distance.pkl')


    # foot_note_tf_idf = sparse.load_npz(path_dir + '/filter8 komoran_for_cosine_distance.pkl')
    valid = pd.read_pickle(path_dir + '/filter8 komoran_for_cosine_distance.pkl')
    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    dep_vars = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                            , dtype=object, sheet_name='Sheet1')


    ind_var_list = ['M000701061_수정PBR(배)', 'M000901001_ln총자산(천원)', 'debt_asset_ratio', 'eps_change_ratio', '수정주가분기수익률']
    dep_var = 'M수정PER'
    matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_vars, dep_var, ind_var_list, 'quanti_per_komoran.pkl')
    matched_quanti_and_qual_data, valid_df_idx_list = pfd.match_quanti_and_qual_data(valid, matched_quanti_data,
                                                                                     'quanti_qaul_per_komoran.pkl')
    matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, 'crp_cls')
    matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, '주기')
    matched_quanti_and_qual_data = pfd.add_one_hot_with_ind_cd(matched_quanti_and_qual_data)

    columns = ['crp_cd', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt', 't_minus_index',
               't_minus_year_index', 'Symbol', 'Name', '결산월', '회계년']  # 주기는 일단 남긴다.
    matched_quanti_and_qual_data.drop(columns, inplace=True, axis=1)
    matched_quanti_and_qual_data.dropna(inplace=True)

    matched_quanti_and_qual_data.drop(['t_1y_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data.drop(['t_1q_cos_dist'], inplace=True, axis=1)
    matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]
    matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')

    X = matched_quanti_and_qual_data.loc[:, matched_quanti_and_qual_data.columns != dep_var]
    y = matched_quanti_and_qual_data[dep_var]
    # X = sm.add_constant(X)
    result = sm.OLS(np.asarray(y), X).fit()
    # print(result.rsquared)
    # # 요약결과 출력
    print(result.summary())

    matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]

    matched_quanti_and_qual_data['EPSxPBR'] = matched_quanti_and_qual_data['eps_change_ratio'] * matched_quanti_and_qual_data['M000701061_수정PBR(배)']
    matched_quanti_and_qual_data['EPSxSIZE'] =matched_quanti_and_qual_data['eps_change_ratio'] * matched_quanti_and_qual_data['M000701061_수정PBR(배)']
    matched_quanti_and_qual_data['EPSxDBT'] =matched_quanti_and_qual_data['eps_change_ratio'] * matched_quanti_and_qual_data['debt_asset_ratio']
    X = pd.concat([
    #                matched_quanti_and_qual_data.eps_change_ratio,
    #                matched_quanti_and_qual_data.EPSxPBR,
    #                matched_quanti_and_qual_data.EPSxDBT,
    #                matched_quanti_and_qual_data.EPSxSIZE,
                   matched_quanti_and_qual_data.t_1q_cos_dist,
                   # matched_quanti_and_qual_data.t_1q_cos_dist
                   matched_quanti_and_qual_data[['ind_광업', 'ind_교육 서비스업', 'ind_농업, 임업 및 어업', 'ind_도매 및 소매업', 'ind_부동산업',
                                                 'ind_사업시설 관리, 사업 지원 및 임대 서비스업', 'ind_수도, 하수 및 폐기물 처리, 원료 재생업',
                                                 'ind_숙박 및 음식점업', 'ind_예술, 스포츠 및 여가관련 서비스업', 'ind_운수 및 창고업',
                                                 'ind_전기, 가스, 증기 및 공기 조절 공급업', 'ind_전문, 과학 및 기술 서비스업', 'ind_정보통신업',
                                                 'ind_제조업', 'ind_협회 및 단체, 수리 및 기타 개인 서비스업']]
                   ], axis=1)
    # df2 = matched_quanti_and_qual_data[matched_quanti_and_qual_data.columns.difference(['B', 'D'])]

    y = matched_quanti_and_qual_data['수정주가분기수익률']
    X = sm.add_constant(X)
    result = sm.OLS(np.asarray(y), X).fit()

    print(result.summary())




    # # 요약결과 저장
    # f = open("OLS_result.txt", 'w')
    # f.write(result.summary())
    # f.close()