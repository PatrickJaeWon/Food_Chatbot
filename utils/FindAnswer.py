class FindAnswer:
    def __init__(self,db):
        self.db=db
    # 검색 쿼리 생성
    def _make_query(self, intent_name, ner_tags):
        sql="select * from chatbot_train_data"
        # 데이터에 의도분석만 존재할 경우 의도 분류를 조건절로 사용
        if intent_name != None and ner_tags == None:
            sql=sql+ " where intent='{}'".format(intent_name)

        elif intent_name != None and ner_tags != None:
            where = ' where intent="%s" ' % intent_name
            if (len(ner_tags) > 0):
                where += 'and ('
                for ne in ner_tags:
                    where += " ner like '%{}%' or ".format(ne)
                where = where[:-3] + ')'
            sql = sql + where

        # 동일한 답변이 2개 이상 존재할 경우 -> 랜덤으로 답변을 가져온다
        sql = sql + " order by rand() limit 1"
        return sql

    # 답변 검색
    def search(self, intent_name, ner_tags):
        # 의도명, 개체명으로 답변 검색
        # select * from chatbot_train_data where=intent_name and ner_tags=ner_tags
        sql=self._make_query(intent_name, ner_tags)
        answer= self.db.select_one(sql)

        # select * from chatbot_train_data where=intent_name
        if answer is None:
            sql=self._make_query(intent_name, None)
            answer= self.db.select_one(sql)

        return (answer['answer'], answer['answer_image'])

    # NER 태그를 실제 입력된 단어로 변환
    def tag_to_word(self, ner_predicts, answer):
        for word, tag in ner_predicts:

            # 변환해야하는 태그가 있는 경우 추가
            if tag == 'B_FOOD' or tag == 'B_DT' or tag == 'B_TI':
                answer = answer.replace(tag, word)

        answer = answer.replace('{', '')
        answer = answer.replace('}', '')
        return answer
