import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

# 챗봇에서 사용하는 사전파일 생성
from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
    with open(filename, 'r', encoding='UTF8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

# 말뭉치 데이터 가져오기
corpus_data=read_corpus_data('corpus.txt')

# 말뭉치 데이터에서 키워드만 추출해서 사전 리스트에 생성
p=Preprocess(word2index_dic='chatbot_dict.bin', userdic='../../utils/user_dic.tsv')
dict = [] # 단어 리스트
for c in corpus_data:
    pos = p.pos(c[1]) # 0000	헬로우		0 ->형식이므로 1번째 정보만 저장
    for k in pos:
        dict.append(k[0])

#토크나이저 처리
#사전에 사용될 word2index 생성
#사전에 첫번째 인덱스에는 oov 사용
tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index #인덱스 딕셔너리 데이터 만들어짐

f=open('chatbot_dict.bin','wb')
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()
