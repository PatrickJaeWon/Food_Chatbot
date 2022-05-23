import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import pandas as pd
import tensorflow as tf
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPool1D, concatenate
from utils.Preprocess import Preprocess

# 데이터 읽어오기
train_file = "total_train_data.csv"
data = pd.read_csv(train_file, delimiter=',')
queries = data['query'].tolist()
intents = data['intent'].tolist()

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')

# 단어 시퀀스 생성
sequences = []
for sentence in queries:
    pos = p.pos(sentence) #형태소 분석
    keywords = p.get_keywords(pos, without_tag=True) #품사 태깅 없이 키워드 분류
    seq = p.get_wordidx_sequence(keywords) # 시퀀스 생성
    sequences.append(seq)

# 단어 시퀀스 벡터 크기
from config.GlobalParams import MAX_SEQ_LEN
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')

print(padded_seqs.shape)
print(len(intents))
