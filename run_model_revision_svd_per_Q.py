from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import math
from sklearn.metrics import mean_squared_error
import time
import pandas as pd
import numpy as np
import os, sys
import preprocess_footnotes_data as pfd
import random as rn
from sklearn.preprocessing import StandardScaler
import keras.backend.tensorflow_backend as K
import keras
from keras import optimizers
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import decomposition
import pickle
from scipy.sparse import csr_matrix, hstack
from scipy import sparse
import glob
from multiprocessing import Pool
import static_test as st

global n_comp
n_comp = 8
bound = 100  # 200(100,8), 100(100, 8), 50(100,8), 1(100,8)
# memo n_comp:  8 loss: mean_squared_error, bound: 100  성공! mape와 mspe가 하락
# memo n_comp:  100 loss: mean_squared_error, bound: 200  #rmse만 유의하게 감소


class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))


def adjusted_rmse(y_true, y_pred):  # 사실 2009년엔 없는 단어여서 그런지 모르겠지만 이걸 Mean Square Percentage Error 라고 함. mape보다 인기는 없어보인다.
    # return np.sqrt((((targets - predictions)/targets) ** 2).mean())
    return np.square(((y_true - y_pred)/y_true)).mean() * 100


def Mape(y_true, y_pred):  # 0에 가까운 값에 약해(규모에 민감)서 사실 그닥 성능은 좋지 않아보인다.
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def Mpe(y_true, y_pred):  # underperformance 인지 overperformance 인지 판단 할 수 있다는 것입니다.
    return np.mean((y_true - y_pred) / y_true) * 100


def per_crp_rolling(tmp):
    identifier1 = ['Symbol',
                    '회계년',
                    '주기',
                    't4_M000901012_재고자산(천원)',
                    't4_M000901006_매출채권(천원)',
                    't4_M_CAPEX',
                    't4_M000904007_매출총이익(천원)',
                    't4_M000904017_판매비와관리비(천원)',
                    't4_ETR',
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
                    't3_ETR',
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
                    't2_ETR',
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
                    't1_ETR',
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
                   'ETR',
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
                                'ETR',
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


def boxplot(result1, result2):
    fig, ax = plt.subplots()
    ax.boxplot([result1, result2], sym="b*")
    # plt.title('Multiple box plots of tips on one Axes')
    plt.xticks([1, 2], ['1', '2'])
    plt.show()


def make_new_model(input_dim, sample_num):
    model = Sequential()
    # hidden = int((input_dim + 1) / 2)
    # hidden = int(sample_num/(5*(input_dim + 1)))
    # hidden = 122
    hidden = int(1/2*(input_dim + 1) + math.sqrt(input_dim + 1))  # 2번째 항은 https://deepai.org/machine-learning-glossary-and-terms/training-pattern
    # model.add(Dense(hidden, input_dim=input_dim, activation='elu'))
    model.add(Dense(hidden, input_dim=input_dim, activation='relu'))
    # model.add(Dense(hidden, input_dim=input_dim, activation='elu'))
    # model.add(Dense(25, input_dim=input_dim, activation='elu'))
    # while 1:
    #     hidden = int(hidden/2)
    #     if hidden <= 8:
    #         break
    #     model.add(Dense(int(hidden), activation='elu'))
    model.add(Dense(1))
    opt = optimizers.adam(lr=0.01)
    # opt = optimizers.SGD(lr=0.20, momentum=0.15, decay=0.0, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=opt)
    # model.compile(loss='mean_squared_logarithmic_error', optimizer=opt)
    # model.compile(loss='mean_absolute_percentage_error', optimizer=opt)
    return model


def mlp_adjust(df, dep, main_ind_var, ind_var_at_paper, adjust, qual):
    tfidf_matrix = sparse.load_npz('./merged_FnGuide/df8_tfidf.npz')  # (21570, 32 + 70133=70165)
    MODEL_DIR = os.getcwd()+'/model '+dep+'-'+main_ind_var+'/'
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
        os.mkdir(MODEL_DIR)
    else:
        os.mkdir(MODEL_DIR)
    num_epochs = 50000
    testRmseList = []
    testmaeList = []
    testmapeList = []
    testadjrmseList = []
    predict_df_li = []
    print(df.shape)
    for seed in range(30):  # seed 값 설정
        print('seed: ', seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        rn.seed(seed)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        # config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        tf.random.set_seed(seed)

        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=config)
        tf.compat.v1.keras.backend.set_session(sess)

        time_series = [(2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1), (2018, 2), (2018, 3)]
        p_li = []
        for t in range(len(time_series) - 1):  # 일종의 10-fold가 된다. 튜플들은 validation에 쓰는걸 기준. (2018, 3)은 test
            year_q = time_series[t]
            # year_q = (2016,1)  # for test
            print(main_ind_var, '-------------\nseed - year: ', seed, '-', year_q)

            val = df[(df['회계년'] == year_q[0]) & (df['주기'] == year_q[1])]
            train = df[df.index < val.index[0]]
            test = df[(df['회계년'] == time_series[t+1][0]) & (df['주기'] == time_series[t+1][1])]
            if qual == 2:
                # print(df[df.index < test.index[-1]].shape)
                df_attach = tfidf_svd(tfidf_matrix, df[df.index <= test.index[-1]].copy(), ind_var_at_paper, dep)
                # print(df_attach)
                val = df_attach[(df_attach['회계년'] == year_q[0]) & (df_attach['주기'] == year_q[1])]
                train = df_attach[df_attach.index < val.index[0]]
                test = df_attach[(df_attach['회계년'] == time_series[t+1][0]) & (df_attach['주기'] == time_series[t+1][1])]

            # print('train ', train.shape)
            # print('val ', val.shape)
            print('test ', test.shape)
            dir_id = str(seed)+'_'+str(year_q[0])+str(year_q[1])
            # 학습 자동 중단 설정
            if not os.path.exists(MODEL_DIR+'/'+dir_id+'/'):
                os.mkdir(MODEL_DIR+'/'+dir_id+'/')
            modelpath = MODEL_DIR+'/'+dir_id+"/{val_loss:11.3f}.hdf5"
            checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=100, mode='min')
            custom_hist = CustomHistory()
            number_of_var = train.shape[1]-4
            model = make_new_model(number_of_var, train.shape[0])
            custom_hist.init()
            model.fit(train.iloc[:, 3:-1], train.iloc[:, -1], epochs=num_epochs, verbose=0,  # 0,1,2번째 데이터는 식별용
                      batch_size=train.shape[0],
                      validation_data=(val.iloc[:, 3:-1], val.iloc[:, -1]),
                      callbacks=[custom_hist, checkpointer, early_stopping_callback])
            file_list = glob.glob(MODEL_DIR+'/'+dir_id+"/*.hdf5")
            file_list.sort(key=os.path.getmtime, reverse=True)
            # print(file_list[0].split('\\')[-1])
            del model       # 테스트를 위해 메모리 내의 모델을 삭제
            model = load_model(file_list[0])
            print('val_rmse_score: ', math.sqrt(float(file_list[0].split('\\')[-1][:-5])))
            # print('val_mape_score: ', file_list[0].split('\\')[-1][:-5])
            testScore = model.evaluate(test.iloc[:, 3:-1], test.iloc[:, -1], verbose=2)
            print('seed - year: ', seed, '-', year_q)
            # print('MSE: %s ' % testScore)
            predictY = model.predict(test.iloc[:, 3:-1])
            tmp = pd.merge(left=test, right=adjust, how='left')
            predictY = np.multiply(predictY.flatten(), tmp['주식수_수정_가중치'].values)

            testScore = math.sqrt(mean_squared_error(test.iloc[:, -1], predictY))
            print('RMSE:%.4f' % testScore)
            r_square = r2_score(test.iloc[:, -1], predictY)
            print('r_square:{0:.4f}'.format(round(r_square, 3)))
            adj_r_square = 1-(1-r_square)*((test.shape[0] - 1)/(test.shape[0]-number_of_var-1))
            print('adj r_square:{0:.4f}'.format(round(adj_r_square, 3)))

            adjrmse = adjusted_rmse(np.asarray(test.iloc[:, -1]), np.asarray(predictY.flatten()))
            print('mspe:', adjrmse)
            mae = mean_absolute_error(test.iloc[:, -1], predictY)
            print('mae:{0:.4f}'.format(round(mae, 3)))
            mape = Mape(np.asarray(test.iloc[:, -1]), np.asarray(predictY.flatten()))
            print('mape:{0:.4f}'.format(round(mape, 3)))

            mpe = Mpe(test.iloc[:, -1], predictY.flatten())
            if mpe < 0:
                print('over perform MPE:', mpe)
            else:
                print('under perform MPE:', mpe)

            predict_df = test.iloc[:, :3].copy()
            # print(list(test.iloc[:, -1]))
            # print(list(predictY.flatten()))
            predict_df['true'] = pd.Series(test.iloc[:, -1])
            predict_df['pred'] = predictY

            testRmseList.append(testScore)
            testadjrmseList.append(adjrmse)
            testmaeList.append(mae)
            testmapeList.append(mape)
            p_li.append(predict_df)
            for file in file_list[1:]:
                if os.path.isfile(file):
                    while 1:
                        try:
                            os.remove(file)
                        except Exception as e:
                            print(e)
                            time.sleep(0.5)
                        if not os.path.isfile(file):
                            break
        predict_df_li.append(pd.concat(p_li, ignore_index=True))
    return testRmseList, testadjrmseList, testmaeList, testmapeList, predict_df_li


def tfidf_svd(tfidf_matrix, df, ind_var_at_paper, dep):  # index 포함된 데이터 프레임
    # print('tfidf_matrix:', tfidf_matrix.shape)
    tsvd = TruncatedSVD(n_components=n_comp)
    tfidf_matrix_reduced = tsvd.fit_transform(tfidf_matrix[list(df['index']), :])
    # print('tfidf_matrix_reduced: ', tfidf_matrix_reduced.shape[0])
    tfidf_matrix_reduced_df = pd.DataFrame(data=tfidf_matrix_reduced, columns=list(range(tfidf_matrix_reduced.shape[1])))
    # tfidf_matrix_reduced_df.reset_index(drop=False, inplace=True)
    # print(tfidf_matrix_reduced_df.shape[0])
    # print(df.shape[0])
    df.reset_index(drop=True, inplace=True)
    tfidf_matrix_reduced_withtag = pd.concat([df.copy(), tfidf_matrix_reduced_df.copy()], axis=1)
    # print(tfidf_matrix_reduced_withtag.isna().sum())
    # tfidf_matrix_reduced_withtag = pd.merge(left=df9, right=tfidf_matrix_reduced_df,
    #                                         how='left', on=['index'], sort=False)
    # tfidf_matrix_reduced_withtag = tfidf_matrix_reduced_withtag[ind_var_at_paper].copy()
    tfidf_matrix_reduced_withtag.drop(['index'], inplace=True, axis=1)
    cols = list(tfidf_matrix_reduced_withtag.columns)
    cols.remove(dep)
    cols.append(dep)
    tfidf_matrix_reduced_withtag = tfidf_matrix_reduced_withtag[cols]
    # print(tfidf_matrix_reduced_withtag.shape)
    # print(list(tfidf_matrix_reduced_withtag.isna().sum()))
    # print(list(tfidf_matrix_reduced_withtag.columns))
    return tfidf_matrix_reduced_withtag


def chk_row_cnt(df):
    for year in range(2013, 2019):
        print(year, '년', df[df['회계년'] == year].shape)  # 738
        for q in range(1,5):
            print(year, '년', q, '분기', df[(df['회계년'] == year) & (df['주기'] == q)].shape)  # 972  # 2013년 14년은 뭔가 이상한 데이터가 많았다.


def select_n_components(var_ratio, goal_var: float) -> int:
    # 출처: https://chrisalbon.com/machine_learning/feature_engineering/select_best_number_of_components_in_tsvd/
    # Set initial variance explained so far
    total_variance = 0.0
    # Set initial number of features
    n_components = 0
    # For the explained variance of each feature:
    for explained_variance in var_ratio:
        # Add the explained variance to the total
        total_variance += explained_variance
        # Add one to the number of components
        n_components += 1
        # If we reach our goal level of explained variance
        if total_variance >= goal_var:
            # End the loop
            break
    # Return the number of components
    return n_components


def identity_tokenizer(text):
    txt_li = text.split()
    return txt_li


def tfidf_to_svd(df, indVar_withFootnote, ind_var_at_paper):
    df8 = pd.read_pickle(os.getcwd() + '/divide_by_sector/filter8 komoran_for_cosine_distance.pkl')  # 독립변수중 산업이나 코사인거리
    df8.reset_index(drop=True, inplace=True)
    df8.reset_index(drop=False, inplace=True)
    df9 = match_quanti_and_qual_data(df8[['index', 'crp_cd', 'rpt_nm', 't_1q_cos_dist']].copy(), df, 'tfidf_match_trimmed.pkl')
    df9.drop(['crp_cd', 'rpt_nm'], inplace=True, axis=1)
    df9.dropna(inplace=True)
    df9['index'] = df9['index'].astype('int')
    df9.to_pickle('./merged_FnGuide/df9_with_index.pkl')
    # df9 = pd.read_pickle('./merged_FnGuide/df9_with_index.pkl')
    return df9
    """    
    # tf = TfidfVectorizer(max_df=0.95, min_df=0, tokenizer=identity_tokenizer)
    # tfidf_matrix = tf.fit_transform(df8['foot_note'])
    # tfidf_matrix = sparse.load_npz('./merged_FnGuide/df_sparse.npz')  # (21570, 32 + 70133=70165)
    tfidf_matrix = sparse.load_npz('./merged_FnGuide/df8_tfidf.npz')  # (21570, 32 + 70133=70165)
    print('tfidf_matrix:', tfidf_matrix.shape)
    tsvd = TruncatedSVD(n_components=100)
    tfidf_matrix_reduced = tsvd.fit_transform(tfidf_matrix[list(df9['index']), :])
    # tfidf_matrix_reduced = tsvd.fit_transform(tfidf_matrix)
    print('tfidf_matrix_reduced: ', tfidf_matrix_reduced.shape[0])
    tfidf_matrix_reduced_df = pd.DataFrame(data=tfidf_matrix_reduced, columns=list(range(tfidf_matrix_reduced.shape[1])))
    # tsvd = TruncatedSVD(n_components=tfidf_matrix.shape[1] - 1)
    # tfidf_tsvd = tsvd.fit(tfidf_matrix)
    # tsvd = TruncatedSVD(n_components=1000)
    # tfidf_tsvd = tsvd.fit(tfidf_matrix)  # 쓸모 있어보이는건 앞의 8개정도. 나머진 자잘하다.

    # tsvd_var_ratios = tsvd.explained_variance_ratio_
    # n_components = select_n_components(tsvd_var_ratios, 0.95)
    # print(n_components)

    # tsvd = TruncatedSVD(n_components=n_components)
    # tfidf_matrix_reduced = tsvd.fit_transform(tfidf_matrix)

    # indVar_withFootnote.append('index')
    # df9 = df9[indVar_withFootnote].copy()
    print(df9.isna().sum())
    # indVar_withFootnote.remove('index')

    # df9 = pd.read_pickle('./merged_FnGuide/data1.pkl')
    tfidf_matrix_reduced_df.reset_index(drop=False, inplace=True)
    df9.reset_index(drop=True, inplace=True)
    tfidf_matrix_reduced_withtag = pd.concat([df9.copy(), tfidf_matrix_reduced_df.copy()], axis=1)
    print(tfidf_matrix_reduced_withtag.isna().sum())
    # tfidf_matrix_reduced_withtag = pd.merge(left=df9, right=tfidf_matrix_reduced_df,
    #                                         how='left', on=['index'], sort=False)
    print(tfidf_matrix_reduced_withtag.shape)
    tfidf_matrix_reduced_withtag.to_pickle('./merged_FnGuide/tfidf_matrix_reduced_withtag.pkl')
    # dd = pd.read_pickle('./merged_FnGuide/tfidf_matrix_reduced_withtag.pkl')

    return tfidf_matrix_reduced_withtag
    """


def placebo_svd(data1):
    tfidf_matrix = sparse.load_npz('./merged_FnGuide/placebo.npz')  # (21570, 32 + 70133=70165)
    tfidf_matrix = tfidf_matrix[:data1.shape[0], :70133]
    tsvd = TruncatedSVD(n_components=n_comp)
    tfidf_matrix_reduced = tsvd.fit_transform(tfidf_matrix)
    data1.reset_index(drop=True, inplace=True)
    tfidf_matrix_reduced_df = pd.DataFrame(data=tfidf_matrix_reduced, columns=list(range(tfidf_matrix_reduced.shape[1])))
    tfidf_matrix_reduced_withtag = pd.concat([data1, tfidf_matrix_reduced_df], axis=1)
    tfidf_matrix_reduced_withtag.to_pickle('./merged_FnGuide/placebo_tfidf_matrix_reduced_withtag.pkl')
    return tfidf_matrix_reduced_withtag


def q_replace_int(df):
    df.loc[df['주기'] == '1Q', '주기'] = 1
    df.loc[df['주기'] == '2Q', '주기'] = 2
    df.loc[df['주기'] == '3Q', '주기'] = 3
    df.loc[df['주기'] == '4Q', '주기'] = 4
    return df


def consensus_performance_check_per_q(consensus_attach):
    time_series = [(2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1), (2018, 2), (2018, 3)]
    testRmseList = []
    testmaeList = []
    testmapeList = []
    testadjrmseList = []

    for t in range(len(time_series) - 1):  # 일종의 10-fold가 된다. 튜플들은 validation에 쓰는걸 기준. (2018, 3)은 test
        print('-----------------------', time_series[t+1])
        # year_q = (2016,1)  # for test
        test = consensus_attach[(consensus_attach['회계년'] == time_series[t + 1][0])
                                & (consensus_attach['주기'] == time_series[t + 1][1])]
        print('test ', test.shape)
        predictY = test['F730011500_EPS (E3)(원)'].values
        true = test['M000601002_EPS(원)'].values
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
    return testRmseList, testadjrmseList, testmaeList, testmapeList


def df_list_to_excel(df_list, file_name):
    with pd.ExcelWriter(file_name) as writer:  # doctest: +SKIP
        for i in range(len(df_list)):
            df_list[i].to_excel(writer, sheet_name='try'+str(i))


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    print('memo n_comp: ', n_comp, 'loss: mean_squared_error, bound:', bound)
    path_dir = os.getcwd()
    ind_var_list = ['M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)',
                    # 'M000904017_판매비와관리비(천원)', 'ETR', 'LF_salesDivEmp'  # 기존 EPS 계산시 사용된 수치들.
                    'M000904017_판매비와관리비(천원)', 'M000911020_유효세율(%)', 'LF_salesDivEmp'  # 기존 EPS 계산시 사용된 수치들.
                    ]  # 여기에 사용할 eps를 더 추가
    version_number = 1
    # constant = True
    constant = False  # 모예린 있을 때만
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
    quanti_qual_matched_file_name = 'revisionAll_quanti_qaul_komoran_dnn.pkl'  # 정성 데이터 포함시킨 데이터.
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

        matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/diffPerInd_'+ quanti_qual_matched_file_name)
        """
    """  
    # matched_quanti_and_qual_data.drop(['t_1y_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data.drop(['t_1q_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/diffPerInd_'+quanti_qual_matched_file_name)
    # print(matched_quanti_and_qual_data.dtypes)
    # matched_quanti_and_qual_data.sort_values(['Symbol', '회계년', '주기'], ascending=['True', 'True', 'True'], inplace=True)
    matched_quanti_and_qual_data['LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['LF_salesDivEmp'])

    etr_calc = pd.read_excel(path_dir+'/financial ratio for independent variable/ETR_calc_table.xlsx', sheet_name='Sheet1')  # 3번 조건. 2번은 이미 충족됨.
    matched_quanti_and_qual_data = pd.merge(left=matched_quanti_and_qual_data, right=etr_calc[['Symbol', '회계년', '주기', 'ETR', 'ETR_treat_minus']],
                                            how='left', on=['Symbol', '회계년', '주기'], sort=False)
    matched_quanti_and_qual_data.replace(to_replace='N/A(IFRS)', value=np.nan, inplace=True)
    matched_quanti_and_qual_data_t4Flatten = rolling_t4(matched_quanti_and_qual_data, ind_var_list_for_rolling)
    print(matched_quanti_and_qual_data.shape[0])
    print(matched_quanti_and_qual_data_t4Flatten.shape[0])
    # matched_quanti_and_qual_data_t4Flatten.reset_index(drop=True, inplace=True)
    # result_df.dropna(inplace=True)
    """
    """    
    columns = ['Symbol', '회계년', '주기', '주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q',
               # 'M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)',
               # 'M000904017_판매비와관리비(천원)', 'M000911020_유효세율(%)', 'LF_salesDivEmp'  # 기존 EPS 계산시 사용된 수치들.
               #  , 'M000909001_영업활동으로인한현금흐름(천원)'  # 황선희 2006 연구
               #  , 'tminus_inv_M000901001_총자산(천원)', 'deltaREV_minus_deltaREC', 'div_A_M000901017_유형자산(천원)'
               #  , 'BIG', 'SIZE', 'LEV', 'ROA', 'LOSS', 'MTB', 'LARGE', 'FOREIGN',
                'dep_M000601002_EPS(원)', 'dep_M000601005_EPS(보통주)(원)',
               'dep_M000601003_수정EPS(원)', 'dep_M000601006_수정EPS(보통주)(원)',
               'dep_M000908001_당기순이익(천원)',
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
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['주기'] == '1Q', '주기'] = 1
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['주기'] == '2Q', '주기'] = 2
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['주기'] == '3Q', '주기'] = 3
    matched_quanti_and_qual_data.loc[matched_quanti_and_qual_data['주기'] == '4Q', '주기'] = 4

    matched_quanti_and_qual_data_t4Flatten.loc[matched_quanti_and_qual_data_t4Flatten['주기'] == '1Q', '주기'] = 1
    matched_quanti_and_qual_data_t4Flatten.loc[matched_quanti_and_qual_data_t4Flatten['주기'] == '2Q', '주기'] = 2
    matched_quanti_and_qual_data_t4Flatten.loc[matched_quanti_and_qual_data_t4Flatten['주기'] == '3Q', '주기'] = 3
    matched_quanti_and_qual_data_t4Flatten.loc[matched_quanti_and_qual_data_t4Flatten['주기'] == '4Q', '주기'] = 4
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

    matched_quanti_and_qual_data['t1_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t1_LF_salesDivEmp'])
    matched_quanti_and_qual_data['t2_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t2_LF_salesDivEmp'])
    matched_quanti_and_qual_data['t3_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t3_LF_salesDivEmp'])
    matched_quanti_and_qual_data['t4_LF_salesDivEmp'] = np.log(matched_quanti_and_qual_data['t4_LF_salesDivEmp'])

    common_share_cnt = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\eps_predict\\financial ratio for independent variable\\for_EPS_independant_var.xlsx'
                                   , sheet_name='Sheet1')
    common_share_cnt['상장주식수_보통주'] = common_share_cnt['상장주식수_보통주'].astype('float')
    common_share_cnt.loc[common_share_cnt['주기'] == '1Q', '주기'] = 1
    common_share_cnt.loc[common_share_cnt['주기'] == '2Q', '주기'] = 2
    common_share_cnt.loc[common_share_cnt['주기'] == '3Q', '주기'] = 3
    common_share_cnt.loc[common_share_cnt['주기'] == '4Q', '주기'] = 4

    matched_quanti_and_qual_data = pd.merge(left=matched_quanti_and_qual_data, right=common_share_cnt[['Symbol', '회계년', '주기', '상장주식수_보통주']],
                                            how='left', on=['Symbol', '회계년', '주기'], sort=False)
    for i in range(1, 5):
        for ind_var in ['M000901012_재고자산(천원)', 'M000901006_매출채권(천원)', 'M_CAPEX', 'M000904007_매출총이익(천원)', 'M000904017_판매비와관리비(천원)']:
            div_var_name = 't' + str(i) + '_' + ind_var
            matched_quanti_and_qual_data[div_var_name] = matched_quanti_and_qual_data[div_var_name] / matched_quanti_and_qual_data['상장주식수_보통주']

    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/t4_'+quanti_qual_matched_file_name)
    
    
    """
    """        
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/t4_'+quanti_qual_matched_file_name)
    matched_quanti_and_qual_data.replace(to_replace='N/A(IFRS)', value=np.nan, inplace=True)

    ## 13~18년 합병 이벤트 있는 기업 제외. 시작27163134322 - 27801470820
    merge_log = pd.read_excel(path_dir+'/합병_내역.xlsx', dtype=object, sheet_name='crp_cd_merge_happend')  # 3번 조건. 2번은 이미 충족됨.
    merge_crp = ['A'+str(x) for x in list(merge_log['거래소코드'].unique())]
    matched_quanti_and_qual_data = matched_quanti_and_qual_data.loc[~matched_quanti_and_qual_data['Symbol'].isin(merge_crp)]
    del merge_log
    ## 13~18년 합병 이벤트 있는 기업 제외. 끝

    
    matched_quanti_and_qual_data.loc[:, ~matched_quanti_and_qual_data.columns.isin(['Symbol', '주기', 'ind'])] = \
        matched_quanti_and_qual_data.loc[:, ~matched_quanti_and_qual_data.columns.isin(['Symbol', '주기', 'ind'])].apply(pd.to_numeric)
    matched_quanti_and_qual_data['t1_M000911020_유효세율(%)'] = matched_quanti_and_qual_data[
                                                                't1_M000911020_유효세율(%)'] / 100.0
    matched_quanti_and_qual_data['t2_M000911020_유효세율(%)'] = matched_quanti_and_qual_data[
                                                                't2_M000911020_유효세율(%)'] / 100.0
    matched_quanti_and_qual_data['t3_M000911020_유효세율(%)'] = matched_quanti_and_qual_data[
                                                                't3_M000911020_유효세율(%)'] / 100.0
    matched_quanti_and_qual_data['t4_M000911020_유효세율(%)'] = matched_quanti_and_qual_data[
                                                                't4_M000911020_유효세율(%)'] / 100.0
    print(matched_quanti_and_qual_data.shape)
    # d = matched_quanti_and_qual_data.describe()
    matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data[
        # (matched_quanti_and_qual_data['t1_M000904007_매출총이익(천원)'] > 0)  # 이론상 매출총이익이 음수가 될 수 있다. 실제로 체크해보니 정말 삼성 엔지니어링이 그런 사례.
        (matched_quanti_and_qual_data['t1_M000904017_판매비와관리비(천원)'] > 0)  # 이게 비용이라 음수로 적는 회사도 있고 이유는 모르지만 계산해보면 억단위로 차이가 나는 경우도 존재. (샘표 15년4분기)
        & (matched_quanti_and_qual_data['t1_LF_salesDivEmp'] > 0)  # 매출이 음수란 의미다. 이건 확실히 문제.
        & (matched_quanti_and_qual_data['t1_M000901006_매출채권(천원)'] > 0)  # 신라섬유가 음수로 보고한게 많은데 실제론 그렇지 않다

        # & (matched_quanti_and_qual_data['t2_M000904007_매출총이익(천원)'] > 0)
        & (matched_quanti_and_qual_data['t2_M000904017_판매비와관리비(천원)'] > 0)
        & (matched_quanti_and_qual_data['t2_LF_salesDivEmp'] > 0)
        & (matched_quanti_and_qual_data['t2_M000901006_매출채권(천원)'] > 0)

        # & (matched_quanti_and_qual_data['t3_M000904007_매출총이익(천원)'] > 0)
        & (matched_quanti_and_qual_data['t3_M000904017_판매비와관리비(천원)'] > 0)
        & (matched_quanti_and_qual_data['t3_LF_salesDivEmp'] > 0)
        & (matched_quanti_and_qual_data['t3_M000901006_매출채권(천원)'] > 0)

        # & (matched_quanti_and_qual_data['t4_M000904007_매출총이익(천원)'] > 0)
        & (matched_quanti_and_qual_data['t4_M000904017_판매비와관리비(천원)'] > 0)
        & (matched_quanti_and_qual_data['t4_LF_salesDivEmp'] > 0)
        & (matched_quanti_and_qual_data['t4_M000901006_매출채권(천원)'] > 0)

        & (matched_quanti_and_qual_data[dep] != 0)
        ].copy()
    print(matched_quanti_and_qual_data_fin.shape)
    # matched_quanti_and_qual_data_fin.dropna(subset=[main_ind_var], inplace=True)
    matched_quanti_and_qual_data_fin.sort_values(['회계년', '주기'], ascending=['True', 'True'], inplace=True)
    """

    add_ind_var_list = ind_var_list.copy()
    add_ind_var_list.append(eps_version)
    ind_var_at_paper = []  # 1~4분기전 데이터
    for i in [1, 2, 3, 4]:  # for i in range(1, 5):
        ind_var_at_paper.extend(['t'+str(i)+'_'+ind_var for ind_var in add_ind_var_list])

    ind_var_at_paper_with_dummy = ind_var_at_paper.copy()  # 순수하게 독립변수

    # matched_quanti_and_qual_data_fin = matched_quanti_and_qual_data_fin[(matched_quanti_and_qual_data_fin['회계년'] > 2012)]
    # matched_quanti_and_qual_data_fin.reset_index(drop=True, inplace=True)  # csr matrix의 index 찾기 좀 번거롭기 때문.

    # print(matched_quanti_and_qual_data_fin.shape)

    identifier = ['Symbol', '회계년', '주기']
    indVar_withoutFootnote = identifier.copy()
    indVar_withoutFootnote.extend(ind_var_at_paper)
    # indVar_withoutFootnote.append('index')
    indVar_withoutFootnote.append(dep)  # 식별 데이터 포함

    indVar_withFootnote = identifier.copy()
    indVar_withFootnote.extend(ind_var_at_paper)
    # indVar_withFootnote.append(main_ind_var)  # footnotes
    # indVar_withFootnote.extend(range(100))  # footnotes
    indVar_withFootnote.append('index')
    indVar_withFootnote.append(dep)
    # df = matched_quanti_and_qual_data_fin.copy()
    # df = df[indVar_withoutFootnote].copy()
    # df.dropna(inplace=True)
    """            
    matched_quanti_and_qual_data_fin.to_pickle('./merged_FnGuide/matched_quanti_and_qual_data_fin.pkl')
    df = matched_quanti_and_qual_data_fin.copy()
    df = df[indVar_withoutFootnote].copy()
    df.sort_values(['회계년', '주기'], ascending=['True', 'True'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    tmp = df[(df['회계년'] == 2018) & (df['주기'] == 4)]
    df = df[df.index < tmp.index[0]]
    del tmp
    df.dropna(subset=ind_var_at_paper, inplace=True)
    df.dropna(subset=[dep], inplace=True)
    # print(df.shape)
    d = df.describe()
    # print(d)
    df = tfidf_to_svd(df, indVar_withFootnote, ind_var_at_paper)
    # df.to_pickle('./merged_FnGuide/before_divide1and2.pkl')
    # df.to_excel('./merged_FnGuide/before_divide1and2.xlsx', encoding='cp949')
    """
    """    """
    # df = pd.read_pickle('./merged_FnGuide/before_divide1and2.pkl')
    df = pd.read_pickle('./merged_FnGuide/df9_with_index.pkl')
    # df.dropna(inplace=True)
    # df['index'] = df['index'].astype('int')

    d = df.describe()
    # print(d)
    # bound = 100  # 5000(no), 3000(no), 1000, 500(no), 100, 1
    df = df[
        (df['t1_M000911020_유효세율(%)'] > -bound)  # 유효세율은 최대 50,000 (여기서 100 나눔)보다 작다.
        & (df['t1_M000911020_유효세율(%)'] < bound)
        & (df['t2_M000911020_유효세율(%)'] > -bound)
        & (df['t2_M000911020_유효세율(%)'] < bound)
        & (df['t3_M000911020_유효세율(%)'] > -bound)
        & (df['t3_M000911020_유효세율(%)'] < bound)
        & (df['t4_M000911020_유효세율(%)'] > -bound)
        & (df['t4_M000911020_유효세율(%)'] < bound)
        ].copy()

    data1 = df[indVar_withoutFootnote].copy()
    data2 = df[indVar_withFootnote].copy()
    # data2 = df[indVar_withFootnote].copy()
    # print('data2: ', data2.shape)
    # data2.dropna(inplace=True)
    # print(data2.shape)

    # data1 = data2[indVar_withoutFootnote].copy()
    # print('data1: ', data1.shape)
    # data1.dropna(inplace=True)
    # print(data1.shape)

    chk_row_cnt(data1)

    common_share_cnt = pd.read_excel(path_dir+'\\financial ratio for independent variable\\common_share_cnt.xlsx'
                                     , sheet_name='Sheet1')
    common_share_cnt['주식수_수정_가중치'] = common_share_cnt['주식수_수정_가중치'].astype('float')
    common_share_cnt = common_share_cnt[['Symbol', '회계년', '주기', '주식수_수정_가중치']].copy()
    common_share_cnt = q_replace_int(common_share_cnt)

    start_time = datetime.now()
    print("total start_time : ", start_time)

    rmse2, adjrmse2, mae2, mape2, predict_df2 = mlp_adjust(data2, dep, main_ind_var, indVar_withFootnote, common_share_cnt, 2)
    print("take time : {}".format(datetime.now() - start_time))
    rmse1, adjrmse1, mae1, mape1, predict_df1 = mlp_adjust(data1, dep, '', indVar_withoutFootnote, common_share_cnt, 1)
    print("take time : {}".format(datetime.now() - start_time))


    k = 10
    rmse1_avg = mean_per_n(rmse1, k)
    mspe1_avg = mean_per_n(adjrmse1, k)
    mae1_avg = mean_per_n(mae1, k)
    mape1_avg = mean_per_n(mape1, k)

    rmse2_avg = mean_per_n(rmse2, k)
    mspe2_avg = mean_per_n(adjrmse2, k)
    mae2_avg = mean_per_n(mae2, k)
    mape2_avg = mean_per_n(mape2, k)

    alpha = 0.05
    print('---------rmse check')
    st.shpiro_dist_test(mape1)

    result1 = pfd.equ_var_test_and_unpaired_t_test(rmse1_avg, rmse2_avg)  # 독립 t-test 단방향 검정 # 유의
    boxplot(rmse1_avg, rmse2_avg)
    result1_1 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse2)  # 독립 t-test 단방향 검정 # 유의
    print('---------mspe check')
    result2 = pfd.equ_var_test_and_unpaired_t_test(mspe1_avg, mspe2_avg)  # 독립 t-test 단방향 검정
    boxplot(mspe1_avg, mspe2_avg)
    result2_1 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse2)  # 독립 t-test 단방향 검정 # 유의
    print('---------mae check')
    result3 = pfd.equ_var_test_and_unpaired_t_test(mae1_avg, mae2_avg)  # 독립 t-test 단방향 검정  # 유의
    boxplot(mae1_avg, mae2_avg)
    result3_1 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse2)  # 독립 t-test 단방향 검정 # 유의
    print('---------mape check')
    result4 = pfd.equ_var_test_and_unpaired_t_test(mape1_avg, mape2_avg)  # 독립 t-test 단방향 검정
    boxplot(mape1_avg, mape2_avg)
    result4_1 = pfd.equ_var_test_and_unpaired_t_test(rmse1, rmse2)  # 독립 t-test 단방향 검정 # 유의

    """        """
    # np.random.seed(42)  # for placebo
    # df_sparse_placebo = placebo_attach(data1)
    # data3 = placebo_svd(data1)
    # data3.to_pickle('./merged_FnGuide/data3.pkl')
    data3 = pd.read_pickle('./merged_FnGuide/placebo_tfidf_matrix_reduced_withtag.pkl')
    indVar_placebo = identifier.copy()
    indVar_placebo.extend(ind_var_at_paper)
    # indVar_withFootnote.append(main_ind_var)  # footnotes
    indVar_placebo.extend(range(n_comp))  # footnotes
    # indVar_placebo.append('index')
    indVar_placebo.append(dep)

    data3 = data3[indVar_placebo]
    rmse3, adjrmse3, mae3, mape3, predict_df3 = mlp_adjust(data3, dep, 'placebo'+main_ind_var, indVar_placebo, common_share_cnt, 1)
    # rmse3, adjrmse3, mae3, mape3, predict_df3 = mlp(data3, dep, 'placebo'+main_ind_var, ind_var_at_paper)
    print("take time : {}".format(datetime.now() - start_time))

    rmse3_avg = mean_per_n(rmse3, k)
    mspe3_avg = mean_per_n(adjrmse3, k)
    mae3_avg = mean_per_n(mae3, k)
    mape3_avg = mean_per_n(mape3, k)

    zippedList = list(zip(rmse1, rmse2, rmse3, adjrmse1, adjrmse2, adjrmse3, mae1, mae2, mae3, mape1, mape2, mape3))
    print("zippedList = ", zippedList)
    # Create a dataframe from zipped list
    col_name = []
    for metric in ['rmse', 'mspe', 'mae', 'mape']:
        for i in range(1, 4):
            col_name.append(metric+str(i))
    idx_li = [(2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1), (2018, 2)]
    idx_list = []
    for j in range(1, 31):
        for idx in idx_li:
            idx_list.append(str(j)+"_"+str(idx[0])+"Q"+str(idx[1]))
    metric_collection = pd.DataFrame(zippedList, columns=col_name, index=idx_list)
    zippedList = list(zip(rmse1_avg, rmse2_avg, rmse3_avg,
                          mspe1_avg, mspe2_avg, mspe3_avg,
                          mae1_avg, mae2_avg, mae3_avg,
                          mape1_avg, mape2_avg, mape3_avg))
    col_name = []
    for metric in ['rmse', 'mspe', 'mae', 'mape']:
        for i in range(1, 4):
            col_name.append(metric + str(i)+'_avg')
    idx_list = []
    for seed in range(30):
        idx_list.append('try'+str(seed))
    metric_collection_10 = pd.DataFrame(zippedList, columns=col_name, index = idx_list)

    with pd.ExcelWriter('metric_result.xlsx') as writer:  # doctest: +SKIP
        metric_collection.to_excel(writer, sheet_name='raw')
        metric_collection_10.to_excel(writer, sheet_name='avg')

    print('placebo check----------------------------------------------------')  # placebo 효과 확인.
    result5 = pfd.equ_var_test_and_unpaired_t_test(rmse1_avg, rmse3_avg)  # 유의
    result6 = pfd.equ_var_test_and_unpaired_t_test(mspe1_avg, mspe3_avg)  # 독립 t-test 단방향 검정
    result7 = pfd.equ_var_test_and_unpaired_t_test(mae1_avg, mae3_avg)  # 독립 t-test 단방향 검정
    result8 = pfd.equ_var_test_and_unpaired_t_test(mape1_avg, mape3_avg)  # 독립 t-test 단방향 검정


    print('check industry classification check------------------------------')  # placebo 효과 확인.
    inv_var_for_ctrl = ['주기_1Q', '주기_2Q', '주기_3Q', '주기_4Q',
                        'crp_cls_K', 'crp_cls_Y',
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
    cols = identifier.copy()
    cols.extend(inv_var_for_ctrl)
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/t4_'+quanti_qual_matched_file_name)
    matched_quanti_and_qual_data = matched_quanti_and_qual_data[cols]
    matched_quanti_and_qual_data[inv_var_for_ctrl] = matched_quanti_and_qual_data[inv_var_for_ctrl].astype('int')
    data4 = pd.merge(left=data1.copy(), right=matched_quanti_and_qual_data.copy(),
                     how='left', on=['Symbol', '회계년', '주기'], sort=False)
    for ctrl in inv_var_for_ctrl:
        if data4[(data4[ctrl] == 1)].shape[0] == 0:
            print(ctrl)
            data4.drop([ctrl], inplace=True, axis=1)

    cols = list(data4.columns)
    cols.remove(dep)
    cols.append(dep)
    data4 = data4[cols]

    rmse4, adjrmse4, mae4, mape4, predict_df4 = mlp_adjust(data4, dep, 'fixed'+main_ind_var, indVar_placebo, common_share_cnt, 1)

    rmse4_avg = mean_per_n(rmse4, k)
    mspe4_avg = mean_per_n(adjrmse4, k)
    mae4_avg = mean_per_n(mae4, k)
    mape4_avg = mean_per_n(mape4, k)
    print('---------rmse check')
    result1 = pfd.equ_var_test_and_unpaired_t_test(rmse1_avg, rmse4_avg)  # 독립 t-test 단방향 검정 # 유의
    print('---------mspe check')
    result2 = pfd.equ_var_test_and_unpaired_t_test(mspe1_avg, mspe4_avg)  # 독립 t-test 단방향 검정
    print('---------mae check')
    result3 = pfd.equ_var_test_and_unpaired_t_test(mae1_avg, mae4_avg)  # 독립 t-test 단방향 검정  # 유의
    print('---------mape check')
    result4 = pfd.equ_var_test_and_unpaired_t_test(mape1_avg, mape4_avg)  # 독립 t-test 단방향 검정

    print('who better check----------------------------------------------------')  # 귀무가설은 산업 데이터 모델보다 주석이 못하다는 것 H0 model2 > model4
    result9 = pfd.equ_var_test_and_unpaired_t_test(rmse2_avg, rmse4_avg)  # t값이 양수인데다 p값이 0.05보다 작으면 사실 model 2의
    result10 = pfd.equ_var_test_and_unpaired_t_test(mspe2_avg, mspe4_avg)  #
    result11 = pfd.equ_var_test_and_unpaired_t_test(mae2_avg, mae4_avg)  #
    result12 = pfd.equ_var_test_and_unpaired_t_test(mape2_avg, mape4_avg)  #

    zippedList = list(zip(rmse3, rmse4, adjrmse3, adjrmse4, mae3, mae4, mape3, mape4))
    # Create a dataframe from zipped list
    col_name = []
    for metric in ['rmse', 'mspe', 'mae', 'mape']:
        for i in range(3, 5):
            col_name.append(metric + str(i))
    idx_li = [(2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1),
              (2018, 2)]
    idx_list = []
    for j in range(1, 31):
        for idx in idx_li:
            idx_list.append(str(j) + "_" + str(idx[0]) + "Q" + str(idx[1]))
    p_f_metric_collection = pd.DataFrame(zippedList, columns=col_name, index=idx_list)

    zippedList = list(zip(rmse3_avg, rmse4_avg,
                          mspe3_avg, mspe4_avg,
                          mae3_avg, mae4_avg,
                          mape3_avg, mape4_avg))
    col_name = []
    for metric in ['rmse', 'mspe', 'mae', 'mape']:
        for i in range(3, 5):
            col_name.append(metric + str(i) + '_avg')
    idx_list = []
    for seed in range(30):
        idx_list.append('try' + str(seed))
    p_f_metric_collection_10 = pd.DataFrame(zippedList, columns=col_name, index=idx_list)

    with pd.ExcelWriter('p_f_metric_result.xlsx') as writer:  # doctest: +SKIP
        p_f_metric_collection.to_excel(writer, sheet_name='raw')
        p_f_metric_collection_10.to_excel(writer, sheet_name='avg')
    """        """
    #### 컨센서스 붙인뒤 dropna하고 성능 평가.
    consensus = pd.read_excel('consensus.xlsx', Sheet_name='Sheet1')
    df = pd.read_pickle('./merged_FnGuide/before_divide1and2.pkl')
    # data1 = df[indVar_withoutFootnote].copy()
    consensus = q_replace_int(consensus)
    consensus_attach = pd.merge(left=data1[['Symbol', '회계년', '주기']].copy(),
                                right=consensus[['Symbol', '회계년', '주기', 'F710011500_EPS (E1)(원)',
                                                 'F730011500_EPS (E3)(원)', 'F760011500_EPS (E6)(원)',
                                                 'M000601002_EPS(원)', 'FM10012910_추정기관수 (E1)']].copy(),
                                how='left', on=['Symbol', '회계년', '주기'], sort=False)
    consensus_attach = consensus_attach[['Symbol', '회계년', '주기', 'F730011500_EPS (E3)(원)', 'M000601002_EPS(원)']]
    consensus_attach.dropna(inplace=True)
    consensus_attach = consensus_attach[consensus_attach['M000601002_EPS(원)'] != 0.0]
    consensus_rmse, consensus_adjrmse, consensus_mae, consensus_mape = consensus_performance_check_per_q(consensus_attach)

    c_rmse_avg = mean_per_n(consensus_rmse, k)
    c_mspe_avg = mean_per_n(consensus_adjrmse, k)
    c_mae_avg = mean_per_n(consensus_mae, k)
    c_mape_avg = mean_per_n(consensus_mape, k)

    zippedList = list(zip(consensus_rmse, consensus_adjrmse, consensus_mae, consensus_mape))
    col_name = []
    for metric in ['rmse', 'mspe', 'mae', 'mape']:
        col_name.append('c_' + metric)
    idx_li = [(2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2), (2017, 3), (2017, 4), (2018, 1), (2018, 2)]
    c_metric_collection = pd.DataFrame(zippedList, columns=col_name, index=idx_li)
    # Create a dataframe from zipped list
    zippedList = list(zip(c_rmse_avg, c_mspe_avg, c_mae_avg, c_mape_avg))
    col_name = []
    for metric in ['rmse', 'mspe', 'mae', 'mape']:
        col_name.append('c_'+metric+'_avg')
    c_metric_collection_10 = pd.DataFrame(zippedList, columns=col_name)

    with pd.ExcelWriter('c_metric_result.xlsx') as writer:  # doctest: +SKIP
        c_metric_collection.to_excel(writer, sheet_name='raw')
        c_metric_collection_10.to_excel(writer, sheet_name='avg')

    df_list_to_excel(predict_df1, 'predict_df1.xlsx')
    df_list_to_excel(predict_df2, 'predict_df2.xlsx')
    df_list_to_excel(predict_df3, 'predict_df3.xlsx')
    df_list_to_excel(predict_df4, 'predict_df4.xlsx')