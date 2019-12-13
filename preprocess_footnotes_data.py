import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import os
import math
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Okt
from konlpy.tag import Hannanum
import re
import gc
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.sparse
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from math import sqrt
from sklearn.model_selection import StratifiedKFold
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from scipy import stats
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
# from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier


# df.loc[i - 1, 'crp_cls'] = crp_cls  # 법인유형(유가증권Y, 코스닥K)
# df.loc[i - 1, 'crp_nm'] = crp_nm  # 공시대상회사(종목명)
# df.loc[i - 1, 'crp_cd'] = crp_cd  # 종목코드
# df.loc[i - 1, 'rpt_nm'] = rpt_nm  # 보고서명
# df.loc[i - 1, 'rcp_no'] = rcp_no  # 보고서 번호. 그러나 첨부파일 번호가 항상 해당 rcp_no에 종속되지 않음.
# df.loc[i - 1, 'dic_cls'] = dic_cls  # 공시구분
# df.loc[i - 1, 'dcm_no'] = dcm_no  # 만약 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
# df.loc[i - 1, 'col_dcm_no'] = col_dcm_no  # 만약 연결 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
# df.loc[i - 1, 'foot_note'] = foot_note  # 주석
# df.loc[i - 1, 'consolidated_foot_note'] = consolidated_foot_note  # 연결재무제표 주석. 주석과 연결재무제표주석이 있다면 일단 연결재무제표 주석을 우선시해서 크롤링.
# df.loc[i - 1, 'acc_crp'] = acc_crp  # 감사보고서를 작성한 회사.
# df.loc[i - 1, 'rcp_dt']


def filtering_from2013to2018(df):  # 2013~2018만 필터링. 설령 13년에 올라온 공시라도 12년 사업에 대한 것일 가능성이 있음
    # df = pd.read_pickle('total crawlDartFootNote data collection.pkl')
    df1_1 = df[df['rpt_nm'].str.contains('2013') | df['rpt_nm'].str.contains('2014') |
            df['rpt_nm'].str.contains('2015') | df['rpt_nm'].str.contains('2016') |
            df['rpt_nm'].str.contains('2017') | df['rpt_nm'].str.contains('2018')]
    # df3[(int(df3['rpt_nm'].split('(')[1].split('.')[0]) > 2012)]
    # for index, row in df.iterrows():
    #     if int(row['rpt_nm'].split('(')[1].split('.')[0]) < 2013:
    # df5 = df5.sort_values(by=['rcp_dt'], ascending=False)
    # save_with_xlsxwriter('filter from2013to2018.xlsx', df5)
    df['crp_cls'] = df.crp_cls.astype("category")
    df1_1.to_pickle('filter1_1 from2013to2018.pkl')
    return df1_1


def filtering_ind_(df1_1):  # 기업 종목코드 리스트 추출.
    # df1_1 = pd.read_pickle('filter1_1 from2013to2018.pkl')
    df1_2 = df1_1['crp_cd'].drop_duplicates().to_frame()
    df1_2.to_excel('crp_cd list.xlsx', index=False)  # 2062개
    return df1_2


def filtering_financial_industry(df1_1, df1_2):  # 금융 및 보험업 제거
    # df1_1 = pd.read_pickle('filter1_1 from2013to2018.pkl')
    # df1_2 = pd.read_excel('crp_ind_match.xlsx', dtype=str)
    df2 = pd.merge(df1_2, df1_1)
    print(df2[df2.ind_cd.str.contains('^64')].shape)
    print(df2[df2.ind_cd.str.contains('^65')].shape)
    print(df2[df2.ind_cd.str.contains('^66')].shape)
    df2_1 = df2[~(df2.ind_cd.str.contains('^64') | df2.ind_cd.str.contains('^65') | df2.ind_cd.str.contains('^66'))]
    df2_1 = df2_1.drop(columns=['acc_crp'])  # 일단 외부감사는 모두 받는 것도 아니고 코드가 완전한것도 아니므로 일단 acc_crp 제외
    df2_1.to_pickle('filter2_1 no_fiance_K.pkl')
    return df2_1


def filtering_less_text_data(df2_1):
    df3_1 = df2_1[df2_1['foot_note'] == 'NA']
    writer = pd.ExcelWriter('filter3_1 no_foot_note_case.xlsx', engine='xlsxwriter')  # import xlsxwriter 깔아야 사용 가능한 방법
    df3_1.to_excel(writer, 'Sheet1', index=False)
    writer.save()

    df3_2 = df2_1[~(df2_1['foot_note'] == 'NA')]
    df3_2.to_pickle('filter3_2 have_foot_note_case.pkl')

    return df3_1, df3_2


def revision_disclose_use_or_just_not_resvision(df3_2):  # 만약 다음 공시 이전까지 개정된 것이 있다면 이를 필터링. 그 이후의 개정된 것은 일단 삭제.
    # df3_2 = pd.read_pickle('filter3_2 have_foot_note_case.pkl')
    df4_1 = df3_2[df3_2['dic_cls'] == 'NA']

    # df4_2 = pd.DataFrame()
    df3_2.sort_values(['crp_cd', 'rpt_nm', 'rcp_dt'], ascending=['True', 'True', 'True'], inplace=True)
    pre_crp_cd = ""
    pre_rpt_nm = ""
    # pre_rpt_dt = ""
    pre_index = ""
    index_list_for_delete = []
    for index, row in df3_2.iterrows():
        # row = df3_2.loc[44881, :]  # for test
        cur_crp_cd = row['crp_cd']
        cur_rpt_nm = row['rpt_nm']
        cur_rcp_dt = row['rcp_dt']

        if cur_crp_cd == pre_crp_cd and cur_rpt_nm == pre_rpt_nm:  # 만약 중복되고
            t_closing_date = cur_rpt_nm[cur_rpt_nm.find("(")+1:cur_rpt_nm.find(")")].split('.')
            tplus_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]), 1)+ relativedelta(months=3)

            cur_rcp_dt_li = cur_rcp_dt.split('.')
            rpt_dt_datetime = datetime(int(cur_rcp_dt_li[0]), int(cur_rcp_dt_li[1]), int(cur_rcp_dt_li[2]))

            if rpt_dt_datetime < tplus_closing_date :  # 일단 데이터 누수 막기 위해 다음 분기 되기 전 데이터 한정
                # 일단 당장 지울 필요는 없지만 일단은 중복이 있다는 얘기. 즉 대체가 필요하다는 소리
                index_list_for_delete.append(pre_index)  # 즉 중복됐으면서 보다 이전날짜에 나온 공시를 지워야함.
            else:  # 만약 다음 분기 이후에 정정되었다면 일단 이건 쓸모가 없단 소리.
                index_list_for_delete.append(index)
        pre_crp_cd = cur_crp_cd
        pre_rpt_nm = cur_rpt_nm
        # pre_rpt_dt = cur_rcp_dt
        pre_index = index
    # 일단 복잡하므로 후순위로 구현.

    df4_2 = df3_2.drop(index_list_for_delete)
    print(df4_2.shape)

    df3_2.sort_values(['crp_cd', 'rpt_nm', 'rcp_dt', 'rcp_no'], ascending=['True', 'True', 'True', 'True'],
                      inplace=True)
    pre_crp_cd = ""
    pre_rpt_nm = ""
    # pre_rpt_dt = ""
    pre_index = ""
    index_list_for_delete = []
    for index, row in df3_2.iterrows():
        # row = df3_2.loc[44881, :]  # for test
        cur_crp_cd = row['crp_cd']
        cur_rpt_nm = row['rpt_nm']
        cur_rcp_dt = row['rcp_dt']
        if row['dic_cls'] != 'NA':
            if cur_crp_cd == pre_crp_cd and cur_rpt_nm == pre_rpt_nm:
                t_closing_date = cur_rpt_nm[cur_rpt_nm.find("(") + 1:cur_rpt_nm.find(")")].split('.')
                tplus_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]),
                                              monthrange(int(t_closing_date[0]), int(t_closing_date[1]))[1]) + relativedelta(months=3)

                cur_rcp_dt_li = cur_rcp_dt.split('.')
                rpt_dt_datetime = datetime(int(cur_rcp_dt_li[0]), int(cur_rcp_dt_li[1]), int(cur_rcp_dt_li[2]))
                # if cur_rpt_nm[:5] == '사업보고서':
                #     index_list_for_delete.append(pre_index)  # 사업 보고서의 경우 3개월 보다 더 오래 걸리기도 하는 모양
                # else:
                if rpt_dt_datetime <= tplus_closing_date:  # 일단 데이터 누수 막기 위해 다음 분기 되기 전 데이터 한정
                    # 일단 당장 지울 필요는 없지만 일단은 중복이 있다는 얘기. 즉 대체가 필요하다는 소리
                    index_list_for_delete.append(pre_index)  # 즉 중복됐으면서 보다 이전날짜에 나온 공시를 지워야함.
                else:  # 만약 다음 분기 이후에 정정되었다면 일단 이건 쓸모가 없단 소리.
                    index_list_for_delete.append(index)
        pre_crp_cd = cur_crp_cd
        pre_rpt_nm = cur_rpt_nm
        # pre_rpt_dt = cur_rcp_dt
        pre_index = index
    print(index_list_for_delete[0:5])

    df4_2 = df3_2.drop(index_list_for_delete)
    # 보통 t-1년 1~12월 이 4월의 사업보고서에 반영,

    df4_1.to_pickle('filter4_1 without_revision.pkl')
    df4_2.to_pickle('filter4_2 with_revision.pkl')

    return df4_1, df4_2


def if_has_colFootnotes_then_replace(df4_1, df4_2):  # 연결재무제표가 있으면 대체하는가 마는가.
    # df4_1 = pd.read_pickle('filter4_1 without_revision.pkl')
    # df4_2 = pd.read_pickle('filter4_2 with_revision.pkl')
    df5_1 = df4_1.drop(columns=['consolidated_foot_note'])
    df5_2 = df4_1
    df5_2['foot_note'] = df4_1[['foot_note', 'consolidated_foot_note']].apply(
        lambda x: x['consolidated_foot_note'] if x['consolidated_foot_note'] != 'NA' else x['foot_note'], axis=1)
    df5_2 = df5_2.drop(columns=['consolidated_foot_note'])

    df5_3 = df4_2.drop(columns=['consolidated_foot_note'])
    df5_4 = df4_2
    df5_4['foot_note'] = df4_2[['foot_note', 'consolidated_foot_note']].apply(
        lambda x: x['consolidated_foot_note'] if x['consolidated_foot_note'] != 'NA' else x['foot_note'], axis=1)

    df5_1.to_pickle('filter5_1 just footnote.pkl')
    df5_2.to_pickle('filter5_2 alternate col to footnotes.pkl')
    df5_3.to_pickle('filter5_3 just footnote.pkl')
    df5_4.to_pickle('filter5_4 alternate col to footnotes.pkl')
    return df5_1, df5_2, df5_3, df5_4


def divide_by_sector(df, filename, directory_name):  # 산업별로 분류에 사용? # 사용 보류 코드
    # df = pd.read_pickle('filter5_2 alternate col to footnotes.pkl')
    # filename = 'filter5_2 alternate col to footnotes'
    # directory_name = './divide_by_sector'
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    sector_list = pd.read_excel('한국표준산업분류(10차)_표.xlsx', sheet_name='Sheet1')
    sector_list = list(sector_list['sector'])
    sector_detailed = pd.read_excel('한국표준산업분류(10차)_표.xlsx', sheet_name='Sheet2', dtype=object)
    df_sector_list = []
    for sector in sector_list:
        # sector = sector_list[0] # for test
        detailed_sector_num = list(sector_detailed[sector_detailed['sector'] == sector]['range'])
        detailed_sector_num = [str(x) for x in detailed_sector_num]
        detailed_sector_regex = "|^".join(detailed_sector_num)  # 작성 의도가 잘 기억 안남. 다시 확인 요하는 코드.
        detailed_sector_regex = '^'+detailed_sector_regex
        df1 = df[df.ind_cd.str.contains(detailed_sector_regex, regex=True)]
        print(sector, df1.shape)
        # df1.to_pickle(directory_name+"/"+sector+'_'+filename+'.pkl')
        df_sector_list.append(df1)
    return df_sector_list, sector_list


def check_isnan_or_string(item):
    check_nan = True
    try:
        check_nan = math.isnan(item)  # nan이면 True
    except Exception:
        print(item)
    return check_nan


def substitute_main():  # fnguide에서 긁어온 종속 변수 손보기. Main인데 이상하게 구멍난애들은 메우는 방식.
    path_dir = 'C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable retrived 2019_05_15'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    file_df_list = []
    file_name_list = []
    for path, dirs, files in os.walk(path_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.xlsx':
                if 'substitute' in file.split(".")[0]:
                    tmp_df = pd.read_excel(path + "\\" + file, dtype=object)  # dtype=str option이 없으면 종목코드가 숫자로 처리됨.
                    if not tmp_df.empty:
                        print(file)
                        file_name_list.append(file)
                        file_df_list.append(tmp_df)
    for tmp_df, file_name in zip(file_df_list, file_name_list):
        # tmp_df = pd.read_excel(path_dir+"\\"+'PER_rawData.xlsx')  # dtype=str option이 없으면 종목코드가 숫자로 처리됨.
        main_dat = tmp_df.loc[:, tmp_df.columns.str.contains('^M')]
        noncon3_dat = tmp_df.loc[:, tmp_df.columns.str.contains('^3')]
        con6_dat = tmp_df.loc[:, tmp_df.columns.str.contains('^6')]
        for index, row in main_dat.iterrows():
            for main in list(main_dat.columns):
                if check_isnan_or_string(row[main]):  # nan인 경우 같은 줄, 같은 item에 연결 또는 단독 쪽 데이터를 체크해서 있으면 끌어온다.
                    item = main.split('-')[1]
                    for noncon3 in list(noncon3_dat.columns):
                        if item == noncon3.split('-')[1] and \
                            not check_isnan_or_string(noncon3_dat.loc[index, noncon3]):  # 혹시라도 같은 항목에 값이 있다면 보삽.
                            row[main] = noncon3_dat.loc[index, noncon3]  # 일단은 개별 재무제표 값이 있으면 이걸로 갈아 끼운다.
                    for con6 in list(con6_dat.columns):
                        if item == con6.split('-')[1] and \
                            not check_isnan_or_string(con6_dat.loc[index, con6]):  # 혹시라도 같은 항목에 값이 있다면 보삽.
                            row[main] = con6_dat.loc[index, con6]  # 일단은 연결 재무제표 값이 있으면 이걸로 갈아 끼운다.
        main_dat['결산월'] = list(tmp_df['결산월'])
        main_dat['회계년'] = list(tmp_df['회계년'])
        main_dat['Symbol'] = list(tmp_df['Symbol'])
        # file_name = 'PER_rawData.xlsx'  # for test
        main_dat.to_excel('./financial ratio for dependent variable retrived 2019_05_15/substitute_'+file_name)  # 보충하긴 했는데 이게 너무 생뚱맞는 값을 뱉는거 같다.
        print('save done')
    # tmp_df.loc[30371].at['M000701020-PER(연율화)(배)']
    # tmp_df.loc[30372].at['M000701020-PER(연율화)(배)']


def xplit(*delimiters):
    return lambda value: re.split('|'.join([re.escape(delimiter) for delimiter in delimiters]), value)


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
    return matched_quanti_and_qual_data, valid_df_idx_list


def match_fnguide_data_among_them(quanti_ind_var, dep_vars, dep_var, ind_var_list, file_name):
    # quanti_ind_var = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\previous research independant variable\\about_EPS_independant_var.xlsx', dtype=object, sheet_name='Sheet1')
    # dep_var = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable retrived 2019_05_15\\EPS_rawData.xlsx', dtype=object, sheet_name='Sheet1')
    # keyword = '수정EPS\(원\)'
    # file_name = ''
    identifier = ['Symbol', 'Name', '결산월', '회계년', '주기']
    dep_vars = pd.concat([dep_vars.loc[:, identifier],
                         dep_vars.loc[:, dep_var]], axis=1)  # 일단 폼은 유지하되 필요한 변수 하나만 떼다 쓰기 위함.
    # dep_vars = dep_vars.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까. 특히 거래가 없는 경우. # regression이라면 이 코드가 필요하지만 classification엔 필요 없는 일이었다. 0,1,2로 클래스가 나뉘는 마당에.
    dep_vars.dropna(thresh=6, inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(dep_var.info())

    # print(quanti_dep_var.columns)
    # print('ind_var : ', ind_var)
    identifier.extend(ind_var_list)
    # print('new ind_var : ', identifier)
    # quanti_ind_var = quanti_ind_var.drop(columns=list(quanti_ind_var.loc[:, quanti_ind_var.columns.str.contains('^주기')].columns))  # 주기는 불필요하니 제거
    quanti_ind_var = quanti_ind_var[identifier]
    print(quanti_ind_var.shape)
    quanti_ind_var = quanti_ind_var.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까.
    quanti_ind_var.dropna(thresh=len(identifier), inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(quanti_ind_var.columns)
    print('quanti_ind_var.info() : ', quanti_ind_var.shape)
    # print(dep_var)
    # result_df = pd.DataFrame()
    for index, row in quanti_ind_var.iterrows():
        # print(index)
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
            # result_df = result_df.append(pd.Series(), ignore_index=True)  # 매칭하는 날짜는 있는데 비어있는 경우 일단 빈칸으로 채우고 넘어간다.
            quanti_ind_var.loc[index, dep_var] = np.nan
            print('empty', index)
            # print(row)
            continue
        # result_df = result_df.append(tplus_data, ignore_index=True)  # 찾은 결과를 한줄씩 붙인뒤 나중에 옆으로 붙일 예정.
        # print('tplus_data : ', tplus_data)
        # print('tplus_data : ', tplus_data[dep_var].iloc[0])
        print('tplus_data : ', tplus_data)
        print('tplus_data : ', list(tplus_data.loc[:, dep_var]))
        quanti_ind_var.loc[index, dep_var] = tplus_data.loc[:, dep_var]  # t+1 분기와 매칭

    # result_df = result_df.drop(columns=['Symbol', 'Name', '결산월', '회계년'])  # 종속변수 쪽 식별 정보는 필요 없음.
    # result_df.reset_index(drop=True, inplace=True)
    # quanti_ind_var.reset_index(drop=True, inplace=True)
    # quanti_ind_var = quanti_ind_var.drop(columns=['Name'])  # 그냥 종목번호보다 보기좋아서 냅둔거라. 지워도 상관없음.

    # quanti_ind_var = pd.concat([quanti_ind_var, result_df], axis=1)
    quanti_ind_var.dropna(inplace=True)  # 이전 단계에서 걸렀을 가능성이 높지만 그래도 1~2개 없는 경우를 거르기 위함.

    # directory_name = 'merged_FnGuide ind_var/'
    # if not os.path.exists(directory_name):
    #     os.mkdir(directory_name)
    # df1.to_excel(directory_name+'/merged_FnGuide ind_var '+keyword+'.xlsx')
    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    # np.save('./merged_FnGuide/'+file_name, quanti_ind_var.values)
    print(quanti_ind_var.iloc[:, 5:].astype('float'))
    quanti_ind_var.to_pickle(directory_name+'/'+file_name)
    print(quanti_ind_var.columns)
    print(quanti_ind_var.shape)
    print(quanti_ind_var.info())

    return quanti_ind_var  # 다음에 쓰려고 ndarray로 반환하지 않음.


### evaluation ###
# revenueMatrix = np.array(RevenueDistributionPerWorld)
# 참고 http://www.dodomira.com/2016/04/02/r%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-t-test/
def equ_var_test_and_unpaired_t_test(x1, x2):  # 모든 조합으로 독립표본 t-test 실시. 일단 다른 변수로 감안.(같다면 등분산 t-test라고 생각)
    # 등분산성 확인. 가장 기본적인 방법은 F분포를 사용하는 것이지만 실무에서는 이보다 더 성능이 좋은 bartlett, fligner, levene 방법을 주로 사용.
    # https://datascienceschool.net/view-notebook/14bde0cc05514b2cae2088805ef9ed52/
    alpha = 0.05
    if stats.levene(x1, x2).pvalue > 0.05:  # 이보다 적으면 이분산
        tTestResult = stats.ttest_ind(x1, x2, equal_var=True)
        print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        # if tTestResult.pvalue < 0.05:
        if (tTestResult[0] > 0) & (tTestResult[1] < alpha):
            compare_mean(x1, x2)
            print("reject null hypothesis, mean of {} is large than mean of {}".format('X1', 'X2'))
        else:
            compare_mean(x1, x2)
            print("two sample mean is same h0 accepted")
    else:
        tTestResult = stats.ttest_ind(x1, x2, equal_var=False)  # 등분산이 아니므로 Welch’s t-test
        print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        # if tTestResult.pvalue < 0.05:
        if (tTestResult[0] > 0) & (tTestResult[1] < alpha):
            compare_mean(x1, x2)
            print("reject null hypothesis, mean of {} is large than mean of {}".format('X1', 'X2'))
        else:
            compare_mean(x1, x2)
            print("two sample mean is same h0 accepted")
    return tTestResult

def nested_cv(X, y, inner_cv, outer_cv, parameter_grid):
    outer_scores = []

    # y = y.reshape(-1, 1)
    # outer_cv의 분할을 순회하는 for 루프
    # (split 메소드는 훈련과 테스트 세트에 해당하는 인덱스를 리턴합니다)
    for training_samples, test_samples in outer_cv.split(X, y):
        # 최적의 매개변수를 찾습니다
        best_parms = {}
        best_score = -np.inf
        # 매개변수 그리드를 순회합니다
        # GridSearchCV(SVR)  # 병렬화. 일단 후순위.
        for parameters in parameter_grid:
            # 안쪽 교차 검증의 점수를 기록합니다
            cv_scores = []
            # inner_cv의 분할을 순회하는 for 루프
            for inner_train, inner_test in inner_cv.split(X[training_samples], y[training_samples]):
                # 훈련 데이터와 주어진 매개변수로 분류기를 만듭니다
                reg = SVC(**parameters)
                reg.fit(X[inner_train], y[inner_train])
                # 검증 세트로 평가합니다
                score = reg.score(X[inner_test], y[inner_test])  # SVC의 기본 성능 평가 척도는 정확도이다. 보통은 0~1
                cv_scores.append(score)
            # 안쪽 교차 검증의 평균 점수를 계산합니다
            print('inner cv_scores : ', cv_scores)
            mean_score = np.mean(cv_scores)
            print('mean cv_score : ', mean_score)
            if mean_score > best_score:
                # 점수가 더 높은면 매개변수와 함께 기록합니다
                best_score = mean_score
                best_params = parameters
        # 바깥쪽 훈련 데이터 전체를 사용해 분류기를 만듭니다
        reg = SVC(**best_params)
        print(best_params)
        reg.fit(X[training_samples], y[training_samples])
        # 테스트 세트를 사용해 평가합니다
        # outer_scores.append(reg.score(X[test_samples], y[test_samples]))
        outer_scores.append(sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples])))
        #, sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples]))))  # 이중 리스트로, 첫번째는 r^2 두번째는 rmse
    print('outer_scores: ', outer_scores)
    return np.mean(outer_scores)  # 전체 데이터 셋 대상으로 한 test의 예측값.


def compare_mean(x1, x2):
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    print("mean of first  one is ", mean1)
    print("mean of second one is ", mean2)
    if mean1 < mean2:
        print("so mean2 is bigger")
    else:
        print("so mean1 is bigger")


def previous_research_with_svm(dataset, try_cnt):
    # try_cnt = 30  # for test
    # param_grid = [{'kernel': ['rbf'],  # rbf면 c와 gamma 쓴다.
    #                'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                'gamma': ['auto']},
    #               {'kernel': ['linear'],  # c만 사용.
    #                'C': [0.001, 0.01, 0.1, 1, 10]},
    #               {'kernel': ['poly'],
    #                'C': [0.001, 0.01, 0.1, 1, 10],
    #                'gamma': ['auto'],
    #                'degree': [2, 3, 4],
    #                'epsilon': [0.01, 0.1, 0.2, 0.5, 1]}
    #               ]
    param_grid = [{'kernel': ['rbf'],
                   'gamma': ['auto']}
                  ]
    over_random_state_try = []
    over_random_state_try_f1 = []
    for seed in range(try_cnt):
        # seed = 42  # for test
        kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        average_kfold_train_test_score_with_highest_hyperparam_of_train_val, f1_score_mean = \
            nested_cv_multiprocess(dataset[:, :-1], dataset[:, -1].ravel(),
                                   kf, kf, param_grid, try_cnt)  # 최소 3*5*5*30=225회
            # nested_cv(scaler.fit_transform(dataset[:, 3:-1]), dataset[:, -1].ravel(), kf, kf, ParameterGrid(param_grid))
        # X_train, X_test, y_train, y_test = \
        #     train_test_split(df.iloc[:, 4:-1], df.iloc[:, -1], test_size=0.2, random_state=seed)  # 제대로 처리됐다면 ['Symbol', 'Name', '결산월', '회계년'] 순이 될 것.
        # best_score = 0
        print(' try :', seed, " ", average_kfold_train_test_score_with_highest_hyperparam_of_train_val)
        over_random_state_try.append(average_kfold_train_test_score_with_highest_hyperparam_of_train_val)
        over_random_state_try_f1.append(f1_score_mean)

        # print(rms)
        # if score > best_score:
        #     best_score = score
        #     best_parameters = {'C':C, 'gamma': gamma}
        # print("최고 점수: {:.2f}".format(best_score))
        # print("최적 매개점수: {}".format(best_parameters))
    # 교차검증 테스트 평균 점수가 같은데, 표준편차가 작다면 그게 더 좋다고 판단.
    # 탐색순서는 C가 고정되고 gamma가 변하는 식.
    return over_random_state_try, over_random_state_try_f1


def nested_cv_multiprocess(X, y, inner_cv, outer_cv, parameter_grid, seed):
    outer_scores = []
    f1_scores = []
    print(X.shape)
    # outer_cv의 분할을 순회하는 for 루프
    # (split 메소드는 훈련과 테스트 세트에 해당하는 인덱스를 리턴합니다)
    # X = X.toarray()  # 늦어질 뿐이다
    # 정량적인 데이터만 쓸 경우
    start_time = datetime.now()
    print("start_time : ", start_time)
    for training_samples, test_samples in outer_cv.split(X, y):
        # 최적의 매개변수를 찾습니다  # 파라미터 세트 하나만 시도해보는거니 생략. 결국 5-fold 30번.
        """
        if X.shape[1] < 1000:
            grid_search = GridSearchCV(SVC(), parameter_grid, cv=inner_cv, n_jobs=-1)
            grid_search.fit(X[training_samples], y[training_samples])
        # 정성적인데이터도 쓸 경우  # 그냥 파라미터 하나만 쓸 경우.
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            print('split done')
            svc = SVC(kernel='rbf', gamma='auto')
            svc.fit(X_train, y_train)
        # 바깥쪽 훈련 데이터 전체를 사용해 분류기를 만듭니다
        print("최적 매개변수:", grid_search.best_params_)
        print("최고 교차 검증 점수: {:.5f}".format(grid_search.best_score_))
        """
        # cls = SVC(**grid_search.best_params_)
        svc = SVC(kernel='rbf', gamma='auto')
        cls = BaggingClassifier(base_estimator=svc, n_estimators=5, n_jobs=-1, max_samples=1.0 / 5.0,)
        cls.fit(X[training_samples], y[training_samples])
        # 테스트 세트를 사용해 평가합니다
        # outer_scores.append(reg.score(X[test_samples], y[test_samples]))
        pred = cls.predict(X[test_samples])
        # print('mean acurracy of this param on testset : ', cls.score(X[test_samples], y[test_samples]))
        mean_acc = accuracy_score(y[test_samples], pred)
        f1 = f1_score(y[test_samples], pred, average='macro')
        print('mean acurracy of this param on testset :', mean_acc)
        print('confusion acurracy of this param on testset :\n', confusion_matrix(y[test_samples], pred))
        print('f1 score of this param on testset :', f1)
        outer_scores.append(mean_acc)
        f1_scores.append(f1)
        print("take time : {}".format(datetime.now() - start_time))
        # print('outer_scores: ', outer_scores)
        #, sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples]))))  # 이중 리스트로, 첫번째는 r^2 두번째는 rmse
    '''
    for training_samples, test_samples in outer_cv.split(X, y):
        # 최적의 매개변수를 찾습니다
        # grid_search = GridSearchCV(SVR(), parameter_grid, cv=inner_cv, n_jobs=-1)
        # print(training_samples.tolist())
        # scores = cross_val_score(SVR(**parameter_grid), X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()], cv=inner_cv)
        # grid_search.fit(X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()])
        # 바깥쪽 훈련 데이터 전체를 사용해 분류기를 만듭니다
        # print("교차 검증 점수: ", scores)
        reg = SVR(kernel='rbf', gamma='auto')
        reg.fit(X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()])
        # 테스트 세트를 사용해 평가합니다
        # outer_scores.append(reg.score(X[test_samples], y[test_samples]))
        print('R^2 of this param : ', reg.score(X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()]))
        # 테스트 세트를 사용해 평가합니다
        outer_scores.append(sqrt(mean_squared_error(reg.predict(X.todok()[test_samples.tolist()]), y.todok()[test_samples.tolist()])))
    print('outer_scores: ', outer_scores)
    '''
    return np.mean(outer_scores), np.mean(f1_scores)  # 전체 데이터 셋 대상으로 한 test의 예측값.


def svm_with_foot_note(X, y, try_cnt):  # https://data-newbie.tistory.com/32 이쪽도 참고바람.
    # param_grid = [{'kernel': ['rbf'],  # rbf면 c와 gamma 쓴다.
    #                'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale']},
    #               {'kernel': ['linear'],  # c만 사용.
    #                'C': [0.001, 0.01, 0.1, 1, 10]},
    #               {'kernel': ['poly'],
    #                'C': [0.001, 0.01, 0.1, 1, 10],
    #                'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale'],
    #                'degree': [2, 3, 4],
    #                'epsilon': [0.01, 0.1, 0.2, 0.5, 1]}
    #               ]
    param_grid = [{'kernel': ['rbf'],
                   'gamma': ['auto']}]
    over_random_state_try = []
    over_random_state_try_f1 = []
    for seed in range(try_cnt):
        kf = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
        average_kfold_train_test_score_with_highest_hyperparam_of_train_val, f1_score_mean = \
            nested_cv_multiprocess(X, y, kf, kf, param_grid, seed)
        over_random_state_try.append(average_kfold_train_test_score_with_highest_hyperparam_of_train_val)
        over_random_state_try_f1.append(f1_score_mean)
    return over_random_state_try, over_random_state_try_f1


def identity_tokenizer(text):
    return text


def filter_pos(df6, pos_tag_list):
    for index, row in df6.iterrows():
        filtered_list = [word for word in row['foot_note'] if word.split('/')[-1] in pos_tag_list]
        if len(filtered_list) < 3:
            print(index)
            break
        df6.at[index, 'foot_note'] = " ".join(filtered_list)
    return df6


def filter_number(df):  # 한나눔의 경우 숫자는 별도로 태깅하지 않는다. 이 숫자를 달리 의미 있게 할 방법이 tf-idf에는 없으므로 제거
    for index, row in df.iterrows():
        new_words = []
        for word in row['foot_note']:  # list형을 전제로 함.
            try:
                float(word.split('/')[0].replace(",", ""))
            except ValueError:
                new_words.append(word)
                pass
        print(new_words)
        df.at[index, 'foot_note'] = new_words
    return df


def add_one_hot(df, col_name):
    df = pd.concat([df, pd.get_dummies(df[col_name], dummy_na=False, prefix=col_name)], axis=1)
    # df.drop([col_name], axis=1, inplace=True)
    return df


def add_one_hot_with_ind_cd(df):
    sector_detailed = pd.read_excel('한국표준산업분류(10차)_표.xlsx', sheet_name='Sheet2', dtype=object)
    print(sector_detailed.info())
    for index, row in sector_detailed.iterrows():
        tmp = df[df.ind_cd.str.contains('^'+str(row['range']))]
        for idx, r in tmp.iterrows():
            # df.loc[idx, 'ind'] = row['sector']  # 대분류
            df.loc[idx, 'ind'] = row['sector2']  # 중분류
    df = add_one_hot(df, 'ind')
    df.drop(['ind_cd'], axis=1, inplace=True)
    return df


def change_list_to_string_footnote_(df):
    for index, row in df.iterrows():
        df.at[index, 'foot_note'] = " ".join(row['foot_note'])
    return df


def tf_idf_prerocess(matched_quanti_and_qual_data, save_dir):
    # save_dir ='C:/Users/lab515/PycharmProjects/crawlDartFootNote/merged_FnGuide/for_per_qual_tf_idf_komoran.npz'
    """ #if use komoran
    """

    path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote'
    for_filter_pos_tag = ['NNG;', 'NNP;', 'NNB;', 'NP;', 'VV;', 'VA;', 'VX;', 'VCP;', 'VCN;', 'MM;', 'MAG;', 'MAJ;',
                          'XPN;', 'XSN;', 'XSV;', 'XSA', 'XR;', 'NF;', 'NV', "NA;"]
    # matched_quanti_and_qual_data = jpd.join_pickle_data('C:/Users/lab515/PycharmProjects/crawlDartFootNote/divide_by_sector', 'komoran')
    # matched_quanti_and_qual_data = filter_pos(matched_quanti_and_qual_data, for_filter_pos_tag)
    tf = TfidfVectorizer(max_df=0.95, min_df=0)
    # for index, row in matched_quanti_and_qual_data.iterrows():
    #     matched_quanti_and_qual_data.at[index, 'foot_note'] = " ".join(row['foot_note'])
    tfidf_matrix = tf.fit_transform(matched_quanti_and_qual_data['foot_note'])
    #
    # dep_var = '수정PER3분할'
    # cols = list(matched_quanti_and_qual_data.columns)
    # cols.remove('foot_note')
    # cols.remove(dep_var)
    # cols.insert(0, "foot_note")
    # cols.append(dep_var)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data[cols]
    quanti_data_predict = matched_quanti_and_qual_data.loc[:, matched_quanti_and_qual_data.columns != 'foot_note']

    B = csr_matrix(quanti_data_predict.values[:,:-1])
    tfidf_matrix_and_quanti = hstack([tfidf_matrix, B])
    print(tfidf_matrix_and_quanti.toarray())

    scipy.sparse.save_npz(save_dir, tfidf_matrix_and_quanti)
    # tfidf_matrix = sparse.load_npz(save_dir)


    tfidf_matrix_komoran = sparse.load_npz('C:/Users/lab515/PycharmProjects/crawlDartFootNote/merged_FnGuide/dataset.npz')