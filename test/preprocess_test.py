import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing

sent='오늘 낮 12시에 탕짜면 먹고 싶어'
#전처리 과정 생성
p=Preprocess(userdic='../utils/user_dic.tsv') #Preprocess 클래스 객체 생성

#형태소 분석기 실행
pos=p.pos(sent)

#품사 태그와 같이 키워드 출력
ret=p.get_keywords(pos, without_tag=False)
print(ret)
