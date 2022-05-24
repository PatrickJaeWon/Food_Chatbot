import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np
from utils.Preprocess import Preprocess

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')

new_sentence = '오늘 오후 13시 2분에 탕수육 주문 하고 싶어요'
pos = p.pos(new_sentence)
keywords = p.get_keywords(pos, without_tag=True)
new_seq = p.get_wordidx_sequence(keywords)
max_len = 40
new_padded_seqs = preprocessing.sequence.pad_sequences([new_seq], padding="post", value=0, maxlen=max_len)
print("새로운 유형의 시퀀스 : ", new_seq)
print("새로운 유형의 시퀀스 : ", new_padded_seqs)

# NER 예측
model = load_model('ner_model.h5')
p = model.predict(np.array([new_padded_seqs[0]]))
p = np.argmax(p, axis=-1)

print("{:10} {:5}".format("단어", "예측된 NER"))
print("-" * 50)
index_to_ner = {1: 'O', 2: 'B_DT', 3: 'B_FOOD', 4: 'I', 5: 'B_OG', 6: 'B_PS', 7: 'B_LC', 8: 'NNP', 9: 'B_TI', 0: 'PAD'}
for w, pred in zip(keywords, p[0]):
    print("{:10} {:5}".format(w, index_to_ner[pred]))
