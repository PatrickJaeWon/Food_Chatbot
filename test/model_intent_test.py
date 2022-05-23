import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.Preprocess import Preprocess
from models.intent.IntentModel import IntentModel

p=Preprocess(word2index_dic='../train_tools/dict/chatbot_dict.bin', userdic='../utils/user_dic.tsv')

intent=IntentModel(model_name='../models/intent/intent_model.h5',preprocess=p)

query='내일 점심 뭐먹을까?'
predict=intent.predict_class(query)
predict_label=intent.labels[predict]

print(query)
print('의도 예측 클래스 : ',predict)
print('의도 예측 레이블 : ',predict_label)
