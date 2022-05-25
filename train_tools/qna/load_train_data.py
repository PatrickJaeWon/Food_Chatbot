import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import pymysql
import openpyxl
from config.DatabaseConfig import *

def all_clear_train_data(db):
    # 기존 학습 데이터 삭제
    sql = '''
            delete from chatbot_train_data
        '''
    with db.cursor() as cursor:
        cursor.execute(sql)

    # auto increment 초기화 (db에서 사용한 index번호를 1로 초기화)
    sql = '''
    ALTER TABLE chatbot_train_data AUTO_INCREMENT=1
    '''
    with db.cursor() as cursor:
        cursor.execute(sql)

#db에 데이터를 저장
def insert_data(db, xls_row):
    intent, ner, query, answer, answer_img_url=xls_row

    sql = '''
        INSERT chatbot_train_data(intent, ner, query, answer, answer_image)
        values(
         '%s', '%s', '%s', '%s', '%s'
        )
    ''' % (intent.value, ner.value, query.value, answer.value, answer_img_url.value)

    # 엑셀에서 불러온 cell에 데이터가 없을 경우, null로 치환
    sql=sql.replace("None","null")

    with db.cursor() as cursor:
        cursor.execute(sql)
        print('{} 저장'.format(query.value))
        db.commit()

train_file='./train_data.xlsx'
db=None
try:
    db = pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        passwd=DB_PASSWORD,
        db=DB_NAME,
        charset='utf8'
    )

    # 기존의 학습 데이터를 초기화 all_clear_train_data()호출
    all_clear_train_data(db)

    # 학습 엑셀 파일 불러오기(openpyxl을 이용)
    wb=openpyxl.load_workbook(train_file)
    sheet=wb['Sheet1']
    for row in sheet.iter_rows(min_row=2): #헤더는 불러오지 않음
        # db에 저장
        insert_data(db, row)
    wb.close()

except Exception as e:
    print(e)

finally:
    if db is not None:
        db.close()
