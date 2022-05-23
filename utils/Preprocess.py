from konlpy.tag import Komoran
import pickle
import jpype

# 전처리는 챗봇 엔젠 내부에서 사용하므로 클래스로 정의 한다.

class Preprocess :
    #userdic : 파일 경로 지정
    def __init__(self, word2index_dic='' , userdic=None):

        if(word2index_dic != ''):
            f = open(word2index_dic, "rb")
            self.word_index = pickle.load(f)
            f.close()
        else:
            self.word_index = None

        #형태소 분석기 초기화(코모란 사용)
        self.komoran=Komoran(userdic=userdic)

        self.exclusion_tags=[
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
            ]

    # 형태소 분석기 pos태거
    # 클래스 외부에서 형태분석기 객체를 호출하는 일이 없게 하기 위함(유지보수가 용이함)
    def pos(self, sentence):
        jpype.attachThreadToJVM()
        return self.komoran.pos(sentence)

    # 불용어 제거 후, 필요한 품사 정보만 가져오기
    # without_tag : 태그 출력 유무
    def get_keywords(self, pos, without_tag=False):
        # exclusion_tags 리스트 정보 저장
        f = lambda x: x in self.exclusion_tags
        word_list = []

        for p in pos:
            if f(p[1]) is False:
                #품사 정보만 추가, 그렇지 않으면 키워드만 추가
                word_list.append(p if without_tag is False else p[0])
        return word_list;

    def get_wordidx_sequence(self, keywords):
        if self.word_index is None:
            return[]

        w2i=[] #형태소 부석 -> 불요어 -> 시퀀스 생성 -> oov를 제외한 나머지가
        for word in keywords:
            try:
                w2i.append(self.word_index[word])
            except KeyError:
                #OOV (out-of-Vocabulary) 사전에 해당 단어에 대한 태깅이 없음 => <UNK> unknown
                w2i.append(self.word_index['OOV'])
        return w2i
