from selenium import webdriver
import time
import pandas as pd
from urllib.parse import urljoin
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from datetime import datetime, timedelta
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.support import expected_conditions
from dateutil.relativedelta import relativedelta
import os
import html5lib
import requests
import urllib
import math
import copy
import numpy as np
from bs4 import BeautifulSoup

crawl_start_time = datetime.now()
print("crawl_start_time : ", crawl_start_time)
# begin_date = '20110101'
# end_date = '20111231'
# begin_date = '20120101'
# end_date = '20121231'
# begin_date = '20130101'  #
# end_date = '20131231'  #
# begin_date = '20140101'  #
# end_date = '20141231'  #
# begin_date = '20150101'  #
# end_date = '20151231'  #
# begin_date = '20160101'  #
# end_date = '20161231'  #
# begin_date = '20170101'  #
# end_date = '20171231'  #
# begin_date = '20180101'  #
# end_date = '20150817'  #
# end_date = '20181231'  #
# end_date = '20181231'  #
# end_date = '20181231'  #
begin_date = '20190101'  #
end_date = '20190630'  #


def init():
    global driver
    driver = webdriver.Chrome()


def goto_next_page():
    page_list = driver.find_element_by_class_name('page_list')
    current_page_num = int(page_list.find_elements_by_class_name('on')[0].text)
    max_page_num = int(page_list.find_elements_by_class_name('page_info')[0].text.split('/')[1].split(']')[0])
    if current_page_num != max_page_num:  # 마지막 페이지인지 확인
        current_page_num += 1
        driver.execute_script("search(" + str(current_page_num) + ")")  # 자바스크립트 함수 호출하는 방식 사용
        return wait_until_result_appear('table_list', 7)
    else:  # 마지막 페이지라면
        print('any more page!')
        return False


def wait_until_result_appear(class_name, delay):  # 로딩이 끝났는지 확인. 100건 기준으로 일반적으로 7초면 로딩 완료.
    # delay = 7  # 로딩만 확인할 뿐 table_list가 나타난다고 검색결과가 0건이 아니란 보장은 없음.
    while True:
        try:
            element_present = EC.presence_of_element_located((By.CLASS_NAME, class_name))
            WebDriverWait(driver, delay).until(element_present)
            # print("Page is ready!")
            return True
        except TimeoutException:
            print("Loading took too much time! ", delay, 'sec')


def check_page_existence():  # 검색결과가 없는지 확인. 있으면 True 없으면 False.
    page_list = driver.find_element_by_class_name('page_list')
    if len(page_list.find_elements_by_class_name('on')) != 0:  # 페이지가 하나라도 존재하면 검색결과가 존재.
        return True
    else:
        return False


def get_attached_footnote(col_or_not, directory_name, rcp_no):  # 재무제표 본문의 주석을 스크래핑 위한 코드.
    doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_tag_name(
        'li')  # 구조상 ul이 하위에 포함된 li까지 검출된다.
    for doc_content in doc_contents:  # 분기재무제표 ul 아래 있는 li에 주석이 있다면, ul, li 텍스트 둘다 나오게 된다. 그러나 최종적으로는 말단에 위치한 텍스트를 붙여 오게 된다.
        # 비효율적인 구간이므로 추후 개량을 요함.  # 해당 li 하위에 추가적인 li가 없다면 안 나오는 식으로 개량 완료(설마 주석이 하위에 뭔가를 두는 일은 없을것)
        if "주석" in doc_content.text and len(
                doc_content.find_element_by_class_name('x-tree-node-ct').find_elements_by_tag_name('li')) == 0:
            # print('at ', doc_content.text)
            doc_content.click()
            driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
            foot_note = driver.find_element_by_xpath('/html/body').text
            get_table_at_footnotes(col_or_not, directory_name, rcp_no)  # 별도로 테이블 추출해서 저장.
            driver.switch_to.default_content()
    return foot_note


def get_table_at_footnotes(col_or_not, directory_name, rcp_no):
    # tbl_list = driver.find_element_by_xpath('/html/body').find_elements_by_tag_name('table')
    tbl_list = list(BeautifulSoup(driver.page_source).find_all('table'))
    tbl_list = [str(i) for i in tbl_list]
    tbl_list = pd.DataFrame(tbl_list)

    if not os.path.exists(
            directory_name + '/' + rcp_no):  # 페이지별 크롤링 결과를 pickel로 저장하는 디렉토리에 rcp_no 이름의 디렉토리 작성. 거기에 연결, 개별 주석의 테이블 데이터 넣을 예정.
        os.mkdir(directory_name + '/' + rcp_no)
    if col_or_not:
        file_name = 'col_table_list ' + str(rcp_no)  # 개별, 연결 구분은 앞의 col의 유무에 따라.
        tbl_list.to_pickle(directory_name + '/' + rcp_no + "/" + file_name + ".pkl")
    else:
        file_name = 'table_list ' + str(rcp_no)
        tbl_list.to_pickle(directory_name + '/' + rcp_no + "/" + file_name + ".pkl")


def get_dcmNo_and_col_dcmNo(option_list, rcp_dt, dcm_no, col_dcm_no):
    # if attach_doc_text_strip.split()[0] == rcp_dt:  # 날짜가 일치한다고 가정.  # 일치하지 않는 경우도 존재하므로 무의미.
    standard_date = datetime.strptime(rcp_dt, "%Y.%m.%d").date()
    min_delta = timedelta.max  # 그냥 무조건 큰 수면 된다.
    col_min_delta = timedelta.max  # 그냥 무조건 큰 수면 된다.
    for option in option_list:
        if option.text.strip() != "+본문선택+" and option.text.strip() != "+첨부선택+":
            # print(option.text.strip().split()[-1])
            # print(rpt_nm.split()[0])
            option_date = datetime.strptime(option.text.strip().split()[0], "%Y.%m.%d").date()
            delta = abs(standard_date - option_date)
            # print('delta : ', delta)
            # print('min_delta  : ', min_delta)
            # if delta < min_delta and (rpt_nm.split()[0] == option.text.strip().split()[-1]):
            #     min_delta = delta
            #     index.append(idx)
            # print("option ", ' '.join(option.text.strip().split()[1:]))
            if delta < min_delta and ' '.join(option.text.strip().split()[1:]) in (  # 가까운 날짜여야 하고.
                    '[정정] 감사보고서', '[정정] 반기검토보고서', '[정정] 분기검토보고서'):  # [정정]이 있는 공시를 우선
                min_delta = delta
                dcm_no = option.get_attribute('value')
            elif delta < col_min_delta and ' '.join(option.text.strip().split()[1:]) in (
                    '[정정] 연결감사보고서', '[정정] 반기연결검토보고서', '[정정] 분기연결검토보고서'):  # 정정을 우선시한다.
                col_min_delta = delta
                col_dcm_no = option.get_attribute('value')
            elif delta < min_delta and dcm_no == "NA" and ' '.join(option.text.strip().split()[1:]) in (
                    '감사보고서', '반기검토보고서', '분기검토보고서'):
                min_delta = delta
                dcm_no = option.get_attribute('value')
            elif delta < col_min_delta and col_dcm_no == "NA" and ' '.join(option.text.strip().split()[1:]) in (
                    '연결감사보고서', '반기연결검토보고서', '분기연결검토보고서'):
                col_min_delta = delta
                col_dcm_no = option.get_attribute('value')
    # link_to_attach = attach_doc.get_attribute('value')
    return dcm_no, col_dcm_no
    # 실제 본문에 종속되지 않은 링크가 존재하는 경우도 있다. 이 때 별도의 rcp_no가 필요해서 실제로는 rcpNo=~&dcmNo=~ 식으로 리턴


def get_index_of_rpt_nm_at_closest_date(option_list, rcp_dt, rpt_nm):  # optionlist는 본문 select 태그나 첨부 select 태그를 의미.
    standard_date = datetime.strptime(rcp_dt, "%Y.%m.%d").date()
    min_delta = timedelta.max  # 그냥 무조건 큰 수면 된다.
    # min_delta = timedelta(days=365*100)  # 그냥 무조건 큰 수면 된다.
    idx = 0
    index = 0
    # print(rpt_nm)
    for option in option_list:
        if option.text.strip() != "+본문선택+" and option.text.strip() != "+첨부선택+":
            # print(option.text.strip().split()[-1])
            # print(rpt_nm.split()[0])
            option_date = datetime.strptime(option.text.strip().split()[0], "%Y.%m.%d").date()
            delta = standard_date - option_date
            # print('delta : ', delta)
            # print('min_delta  : ', min_delta)
            if delta < min_delta and (rpt_nm.split()[0] == option.text.strip().split()[-1]):
                min_delta = delta
                index = idx
        idx += 1
    return index


def make_sheet(crp_cls, directory_name):  # 디렉토리명은 추후 추가적인 테이블 추출 후 사용 예정.
    df = pd.DataFrame()
    tbody = driver.find_element_by_xpath('//*[@id="listContents"]/div[1]/table/tbody')
    i = 0
    # row = tbody.find_elements_by_tag_name('tr')[0] # for test
    for row in tbody.find_elements_by_tag_name('tr'):
        i += 1
        # j = 2
        print('i', i)
        for j in (1, 4, 2):  # 먼저 날짜 얻어두는게 유리.
            cell = row.find_elements_by_tag_name('td')[j]
            # cell = row.find_elements_by_tag_name('td')[2]  # for test
            # print('j', j)
            if j == 1:
                crp_nm = cell.text.strip()  # 공시대상회사(종목명)
                '''
                cell.click()
                wait_until_result_appear('pop_table_B', 0.5)
                crp_cd = driver.find_element_by_xpath('//*[@id="pop_body"]/div/table/tbody/tr[4]/td').text
                driver.find_element_by_xpath('//*[@id="closePop"]/img').click()
                '''
                current_window = driver.current_window_handle
                driver.execute_script("window.open('" + cell.find_element_by_tag_name('a').get_attribute('href') + "')")  # 이 코드에서 자꾸 사용하는 윈도우 창이 바뀌어 사용에 불편을 줌.
                new_window = [window for window in driver.window_handles if window != current_window][0]
                driver.switch_to.window(new_window)

                crp_cd = driver.find_element_by_xpath('//*[@id="pop_body"]/div/table/tbody/tr[4]/td').text  # 종목코드.
                #  6자리 숫자가 아니면 에러라 봐도 좋다.

                # driver.execute_script("window.history.go(-1)")  # 뒤로 가기
                # driver.execute_script("closeCorpInfo(); return false;")  # 스크립트 실행
                driver.close()  # 현재 탭을 닫고 기존 검색결과 창으로 돌아감.
                driver.switch_to.window(current_window)

            elif j == 4:
                rcp_dt = cell.text  # 접수일자. 형식은 'yyyy.mm.dd' 파싱에 주의.
            elif j == 2:
                dic_cls = 'NA'
                if '[' in cell.text:  # 만약 별도로 공시구분이 될만한 사항이 존재한다면.
                    dic_cls = cell.text.split(']')[0].split('[')[1]  # 공시구분. 아마 정기공시라면 기재정정과 그렇지 않은 두 종류 뿐일것.
                    rpt_nm = cell.text.split(']')[1]  # 보고서명. '공시구분+보고서명+기타정보'로 구성
                else:
                    rpt_nm = cell.text  # 보고서명. '공시구분+보고서명+기타정보'로 구성
                rcp_no = cell.find_element_by_tag_name('a').get_attribute('href').split('=')[1]  # 보고서 번호
                # cell.click()
                current_window = driver.current_window_handle
                driver.execute_script("window.open('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + rcp_no + "')")
                new_window = [window for window in driver.window_handles if window != current_window][0]
                driver.switch_to.window(new_window)

                # trigger_point = 0
                # if '본 문서는 최종문서가 아니므로 투자판단시 유의하시기 바랍니다' in driver.find_element_by_xpath('//*[@id="center"]/div[1]').text:
                #     trigger_point = 1
                #     print('본 문서는 최종문서가 아니므로 투자판단시 유의하시기 바랍니다')

                # 사업보고서를 눌렀는데 바로 본문이 안나오고 [정정]감사보고서가 나오는 경우가 존재한다.
                # rcp_no가 따로따로 논다(사실 따로 놀아도 알아서 조정되어 출력되므로 실제 페이지 보는데는 문제 없음).
                # 이 경우 강제로 본문으로 바꿔야 할 필요가 존재.
                primary_doc_list = driver.find_element_by_id('family').find_elements_by_tag_name('option')
                select = Select(driver.find_element_by_id('family'))
                if '+본문선택+' == select.first_selected_option.text.strip():
                    # print('첫장부터 관련 본문이 안 나오고 검토보고서가 나오는 예외상황')
                    prim_index = get_index_of_rpt_nm_at_closest_date(primary_doc_list, rcp_dt, rpt_nm)
                    # print(prim_index)
                    rcp_no = primary_doc_list[prim_index].get_attribute('value').split("=")[1]
                    driver.get('http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no)
                wait_until_result_appear('x-tree-root-node', 1)
                doc_contents = driver.find_element_by_class_name('x-tree-root-node').find_elements_by_tag_name('li')

                foot_note = ''  # 재무제표 주석
                consolidated_foot_note = ''  # 연결재무제표 주석
                acc_crp = ''  # 감사 혹은 검토한 회계법인명

                # 재무제표 본문에서 있으면 주석 스크래핑. 없으면 감사인(법인명)이라도.
                for doc_content in doc_contents:
                    # print(doc_content.text)
                    if "연결재무제표 주석" in doc_content.text and len(
                            doc_content.find_element_by_class_name('x-tree-node-ct').find_elements_by_tag_name(
                                'li')) == 0:  # 연결 재무제표가 그냥 계열사가 없다고만 서술 될 가능성도 존재.
                        # print("연결재무제표 주석")
                        doc_content.click()
                        driver.switch_to.frame(
                            driver.find_element_by_tag_name("iframe"))  # iframe은 selenium으로 읽을 때 주의 사항. 버그 예상 지점.
                        consolidated_foot_note = driver.find_element_by_xpath('/html/body').text
                        get_table_at_footnotes(True, directory_name, rcp_no)
                        driver.switch_to.default_content()
                        continue  # 연결재무제표 주석은 재무제표 주석이라는 문자열을 공통으로 포함하기 때문.
                    if "재무제표 주석" in doc_content.text and len(
                            doc_content.find_element_by_class_name('x-tree-node-ct').find_elements_by_tag_name(
                                'li')) == 0:
                        # print('재무제표 주석')
                        doc_content.click()
                        driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
                        foot_note = driver.find_element_by_xpath('/html/body').text
                        get_table_at_footnotes(False, directory_name, rcp_no)
                        driver.switch_to.default_content()
                    if "감사인" in doc_content.text:  # IV. 감사인의 감사의견 등
                        # print(doc_content.text)
                        doc_content.click()
                        driver.switch_to.frame(driver.find_element_by_tag_name("iframe"))
                        # texts = driver.find_element_by_xpath('/html/body').text.split('\n')
                        tables = driver.find_element_by_xpath('/html/body').find_elements_by_tag_name('table')
                        # 테이블 중에 법인명이 가장 빨리나오는 법인이 해당 재무제표의 감사법인이라고 간주.
                        texts = ''  # 법인이란 키워드는 커녕 아무 내용도 없는 경우도 존재할 경우를 대비
                        for table in tables:
                            if '법인' in table.text:
                                texts = table.text
                                break
                            else:
                                texts = table.text
                        # 공시하는 회사명 법인명을 적었을 수 있다. 열 이름이 '법인', '회계법인'일 가능성도 존재.
                        is_1strow = 0
                        texts_list = texts.split('\n')
                        for line in texts_list:
                            # print(line)
                            if acc_crp == '' and is_1strow == 0:
                                acc_crp = 'http://dart.fss.or.kr/dsaf001/main.do?rcpNo=' + rcp_no
                                is_1strow = 1
                                continue  # 첫줄은 건너뛴다. (주의)
                            # print(line)
                            if '법인' in line:  # 추후에 교차확인 요함.
                                for corporation in line.split():  # '한영 회계법인' 처럼 입력되면 아예 아무것도 입력이 안되는 사태가 벌어짐.
                                    if '회계법인' in corporation and '법인' != corporation and '회계법인' != corporation:  # (주의)
                                        acc_crp = corporation  # 유난히 글자수가 적다면 확인이 요구됨.
                                        break  # 가장 처음 법인이란 단어가 나오면 루프 깨고 나간다.
                                break
                        # print("acc_crp ", acc_crp)
                        driver.switch_to.default_content()
                # flag1, flag2, flag3, flag4 = 0  # for test
                dcm_no = 'NA'  # 감사보고서 번호
                col_dcm_no = 'NA'  # 연결 감사 보고서 번호

                if foot_note == '' or len(
                        foot_note.split('\n')) < 5:  # 만약 재무제표 상에서 주석을 긁어오지 못했다면 또는 주석 란에 몇줄 적지 않았다면 첨부파일을 찾을 것.
                    attach_doc_list = driver.find_element_by_id('att').find_elements_by_tag_name('option')
                    # select = Select(driver.find_element_by_id('att'))
                    '''
                    for attach_doc in attach_doc_list:  
                        # print(attach_doc.text)
                        option_text = attach_doc.text.strip().split()
                        if len(option_text) > 1:  # 기타 다른 첨부 보고서명. # 형식이 '날짜 보고서명'을 따르기 때문에 띄워쓰기로 분할하면 2조각 이상 나와야 한다.
                            option_text = " ".join(option_text[1:])
                        else:  # +첨부선택+ 옵션만 여기서 걸러질 것.
                            if len(attach_doc_list) == 1:  # 첨부서류가 없는 경우
                                print('no attachment!')
                                break
                            continue
                        '''
                    if len(attach_doc_list) == 1:  # 첨부서류가 없는 경우
                        # print('no attachment!')
                        pass
                    else:  # 모종의 이유로 로딩이 안되어서 select태그를 이용한 선택 자체가 안되는 오류가 생김. 기다려도 안되는 것으로 보아 빠른 시간내에 해치워야 가망이 있다고 판단. dcm_no, col_dcm_no만 빨리 얻을 수있도록 변경.
                        dcm_no, col_dcm_no = get_dcmNo_and_col_dcmNo(attach_doc_list, rcp_dt, dcm_no, col_dcm_no)

                    if dcm_no != "NA":
                        driver.get('http://dart.fss.or.kr/dsaf001/main.do?' + dcm_no)
                        foot_note = get_attached_footnote(False, directory_name, rcp_no)
                    else:
                        dcm_no == "http://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + rcp_no  # 만약 본문에도 주석을 못 얻었는데 이렇다면 문제가 있는것. 눈으로 확인 요함.
                    if col_dcm_no != "NA":  # 각각 실행해야하므로 elif로 묶지 말 것.
                        driver.get('http://dart.fss.or.kr/dsaf001/main.do?' + col_dcm_no)
                        consolidated_foot_note = get_attached_footnote(True, directory_name, rcp_no)
                    else:
                        col_dcm_no == "http://dart.fss.or.kr/dsaf001/main.do?rcpNo=" + rcp_no  # 만약 본문에도 주석을 못 얻었는데 이렇다면 문제가 있는것. 눈으로 확인 요함.

                else:  # 이미 주석을 얻었다면 감사/검토 문서 번호 만 있다면 추출
                    attach_doc_list = driver.find_element_by_id('att').find_elements_by_tag_name('option')
                    for attach_doc in attach_doc_list:  # 모종의 이유로 로딩이 안되어서 선택 자체가 안되는 오류가 생김. 기다려도 안되는 것으로 보아 빠른 시간내에 해치우면 가망이 있다고 판단.
                        # print(attach_doc.text)
                        option_text = attach_doc.text.strip().split()
                        if len(option_text) == 1:  # 기타 다른 첨부 보고서명. # 형식이 '날짜 보고서명'을 따르기 때문에 띄워쓰기로 분할하면 2조각 이상 나와야 한다.
                            # option_text = " ".join(option_text[1:])
                            pass
                        else:  # +첨부선택+ 옵션만 여기서 걸러질 것.
                            if len(attach_doc_list) == 1:  # 첨부서류가 없는 경우
                                print('no attachment!')
                                break
                            continue

                        # print(option_text)
                        dcm_no, col_dcm_no = get_dcmNo_and_col_dcmNo(attach_doc_list, rcp_dt, dcm_no, col_dcm_no)
                if len(consolidated_foot_note.split(
                        '\n')) < 5:  # and '없' in consolidated_foot_note:  # 5줄 미만이면 연결재무제표가 없는데도 항목이 존재한것.
                    print('연결재무제표 주석 해당사항 없음')
                    col_dcm_no = 'NA'
                    consolidated_foot_note = 'NA'  # 일단 내용이 없다는 것만이라도 표시를 해두는 것이 좋을 것.
                if len(foot_note.split(
                        '\n')) < 5:  # and '없' in consolidated_foot_note:  # 5줄 미만이면 연결재무제표가 없는데도 항목이 존재한것.
                    print('error 개별 재무제표 주석도 몇 줄 없음')
                    dcm_no = 'NA'
                    foot_note = 'NA'  # 일단 내용이 없다는 것만이라도 표시를 해두는 것이 좋을 것.

                driver.close()  # 탭을 닫고 기존 검색결과 창으로 돌아감.
                driver.switch_to.window(current_window)

        # 보고서명에 있는 날짜(e.g. 2018년 3월). 현실적으로는 별로 중요치 않아 제외.
        # 법인등록번호 # 종목 코드가 있으므로 제외.
        # 공시가 올라온 당일부터 5 영업일. 계산으로 때우기 바람.
        df.loc[i - 1, 'crp_cls'] = crp_cls  # 법인유형(유가증권Y, 코스닥K)
        df.loc[i - 1, 'crp_nm'] = crp_nm  # 공시대상회사(종목명)
        df.loc[i - 1, 'crp_cd'] = crp_cd  # 종목코드
        df.loc[i - 1, 'rpt_nm'] = rpt_nm  # 보고서명
        df.loc[i - 1, 'rcp_no'] = rcp_no  # 보고서 번호. 그러나 첨부파일 번호가 항상 해당 rcp_no에 종속되지 않음.
        df.loc[i - 1, 'dic_cls'] = dic_cls  # 공시구분
        df.loc[i - 1, 'dcm_no'] = dcm_no  # 만약 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
        df.loc[i - 1, 'col_dcm_no'] = col_dcm_no  # 만약 연결 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
        df.loc[i - 1, 'foot_note'] = foot_note  # 주석
        df.loc[
            i - 1, 'consolidated_foot_note'] = consolidated_foot_note  # 연결재무제표 주석. 주석과 연결재무제표주석이 있다면 일단 연결재무제표 주석을 우선시해서 크롤링.
        df.loc[i - 1, 'acc_crp'] = acc_crp  # 감사보고서를 작성한 회사.
        df.loc[i - 1, 'rcp_dt'] = rcp_dt  # 접수일자

        # print('i', i)
    return df


def main():
    init()
    # crp_cls = input('법인유형(e.g. 유가증권시장이면 Y, 코스닥이면 K, 코넥스면 N, 기타는 E)를 대문자로 입력하세요.')
    crp_cls = 'Y'  # 테스트용
    # crp_cls = 'K'  # 테스트용
    now = datetime.now().strftime('%Y-%m-%d %H-%M-%S')  # 파일이름 지을때 사용할 문자열(날짜형이 아닌 문자형)
    driver.get('http://dart.fss.or.kr/dsab002/main.do')

    # 기간지정(코드상에 미리 지정)
    driver.find_element_by_id('startDate').clear()
    driver.find_element_by_id('ext-gen81').send_keys(Keys.RETURN)  # alert 제거
    driver.find_element_by_name('startDate').send_keys(begin_date)
    driver.find_element_by_id('endDate').clear()
    driver.find_element_by_id('endDate').send_keys(end_date)

    # 최종보고서만 보는 것을 해제(실제로는 정정되지 않은 정보가 영향을 미쳤을 것이므로)
    driver.find_element_by_id('finalReport').click()

    # 업종지정 : 일단 업종 코드만 얻는 것이 목적이므로 생략. 추후 고려 요함.

    # 법인유형
    select = Select(driver.find_element_by_id('corporationType'))
    if crp_cls == 'Y':
        select.select_by_visible_text('유가증권시장')  # 참고로 value로 하려면 유가증권시장은 P. API와 다름.
    elif crp_cls == 'K':
        select.select_by_visible_text('코스닥시장')  # 참고로 value로 하려면 코스닥은 A. API와 다름.
    elif crp_cls == 'N':
        select.select_by_visible_text('코넥스시장')  # 참고로 value로 하려면 코넥스는 N. API와 동일.
    else:
        select.select_by_visible_text('기타법인')  # 참고로 value로 하려면 기타는 E. API와 동일.

    # 보고서 지정
    driver.find_element_by_id('publicTypeButton_01').click()  # 정기 공시
    driver.find_element_by_id('publicType1').click()  #
    # driver.find_element_by_id('publicType2').click()
    # driver.find_element_by_id('publicType3').click()

    # 최대 조회 수 지정(최대 100까지 볼 수 있다)
    select = Select(driver.find_element_by_id('maxResultsCb'))
    select.select_by_visible_text('100')  # select by visible text
    # select.select_by_visible_text('15')  # for test

    driver.find_element_by_id('searchpng').click()
    wait_until_result_appear('table_list', 7)
    check_more = check_page_existence()
    page_num = 0

    directory_name = './' + crp_cls + '_' + str(page_num) + '_from' + begin_date + 'to' + end_date
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)

    while check_more:
        # searchDateEnd = searchDateStart + timedelta(days=1)
        # searchDateStartStr = searchDateStart.strftime('%Y-%m-%d')
        # searchDateEndStr = searchDateEnd.strftime('%Y-%m-%d')
        page_num += 1
        print('page : ', page_num)
        df = make_sheet(crp_cls, directory_name)  # 로드된 웹 요소들 긁어오기.
        df.to_pickle(directory_name + '/' + crp_cls + '_' + str(
            page_num) + '_from' + begin_date + 'to' + end_date + '_crawlDate_' + now + '.pkl')
        # break  # for test
        check_more = goto_next_page()  # 다음 게시물 번호로 이동 일단 한페이지당 약 10분정도 소요되는 것으로 추정.
        # 코스피 관련 결과를 모두 긁는다면 199*10=33.1시간, 코스닥 관련 결과를 모두 긁는다면 299*10=49.8시간. 총 약 83시간=약 3일 5시간.


# if __name__ == '__main__':
#     main()
main()
print("take time : {}".format(datetime.now() - crawl_start_time))
# driver.quit()
