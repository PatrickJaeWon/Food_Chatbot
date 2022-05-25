import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import pymysql
import pymysql.cursors
import logging

# DataBase 제어
class Database:

    def __init__(self, host, user, password, db_name, charset='utf8'):
        self.host = host
        self.user = user
        self.password = password
        self.charset = charset
        self.db_name = db_name
        self.conn = None

    # DB연결
    def connect(self):
        #이미 연결이 되었으면 통과
        if self.conn != None:
            return

        self.conn = pymysql.connect(
            host=self.host,
            user=self.user,
            password=self.password,
            db=self.db_name,
            charset=self.charset
        )

    def close(self):
        if self.conn is None:
            return

        if not self.conn.open:
            self.conn = None
            return
        self.conn.close()
        self.conn = None

    # SQL 구문
    def execute(self, sql):
        last_row_id=-1
        try:
            with self.conn.cursor() as cursors:
                cursor.execute(sql)
            self.conn.commit()
            last_row_id=cursor.lastrowid
            logging.debug("execute last_row_id : %d", last_row_id)
        except Exception as ex:
            logging.error(ex)

        finally:
            return last_row_id

    # select 구문 실행 후 단 1개의 데이터 row만 불러오는 메소드
    def select_one(self, sql):
        result=None

        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                result=cursor.fetchone()
        except Exception as ex:
            logging.error(ex)

        finally:
            return result

    # select 구문 실행 후 단 1개의 데이터 row만 불러오는 메소드
    def select_all(self, sql):
        result=None

        try:
            with self.conn.cursor(pymysql.cursors.DictCursor) as cursor:
                cursor.execute(sql)
                result=cursor.fetchall()
        except Exception as ex:
            logging.error(ex)

        finally:
            return result
