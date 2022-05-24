import sys, os
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing
import numpy as np
from utils.Preprocess import Preprocess
from models.ner.NerModel import NerModel

p = Preprocess(word2index_dic='../../train_tools/dict/chatbot_dict.bin',
               userdic='../../utils/user_dic.tsv')

ner=NerModel(model_name='../ner/ner_model.h5', preprocess=p)
query = '저녁 7시에 맛있는 식사 준비해주세요.'
predicts=ner.predict(query)
tags=ner.predict_tags(query)
print(predicts)
print(tags)
