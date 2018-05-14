import tensorflow as tf
import numpy as np
import json
import sys
import argparse
import requests
#import PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import nltk as nt
from config import FLAGS
from collections import Counter


MY_KEY = '91084af0198be3d0b93ea1d15cb7f989'

with open('dataset.json',encoding='utf8') as f:
    d = json.load(f)
    #If for dump,       https://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file


def key_generation(image_url):
    headers = {'Authorization':'KakaoAK {}'.format(MY_KEY)}



class Twit():
    MAIN_KEY = ['hashtags','text','hashtags']
    def __init__(self):
        self.data_path = FLAGS.twit_path
        self.twits = self.load_data() # 80,10,10 for train,dev,test
        self.voca = self.build_voca()

        self.UNK_KEY = len(self.voca.keys())    #단어장에 없는 단어
        self.BEG_KEY = len(self.voca.keys())+1  #ENCODING 시작
        self.EOS_KEY = len(self.voca.keys())+2  #ENCODING 종료

        start = self._idx_in_epoch = 0


    def load_data(self):
        """
        raw data에서 MAIN_KEY만을 추출한 뒤 가공한다.
        text tokenize, VISION API
        """
        with open(self.data_path,encoding='utf8') as f:
            raw_data = json.load(f)
        data = []
        for i,t in enumerate(raw_data):
            data.append({'raw_text':t['text'],'hashtags':t['hashtags'],'image':t['media']})
            pos = nt.pos_tag(nt.word_tokenize(t['text']))
            data[-1]['tokens'] = [i[0] for i in pos]

            while 'https' in data[-1]['tokens']: #rejoin the tokenized url
                idx = data[-1]['tokens'].index('https')
                data[-1]['tokens'][idx] = data[-1]['tokens'][idx] + data[-1]['tokens'][idx+1] + data[-1]['tokens'][idx+2]
                data[-1]['tokens'] = data[-1]['tokens'][:idx+1] + data[-1]['tokens'][idx+3:]
        return data

    def build_voca(self):
        '''
        단어 저장, idx 부여, 나중에 embedding한거도 여기에 해야하나?
        '''
        voca = Counter()
        for d in self.twits:
            for w in d['tokens']:
                voca[w]+=1

        pairs = sorted(voca.items(), key=lambda x: (-x[1],x[0]))
        words, _ = list(zip(*pairs))
        word_id = dict(zip(words, range(len(words))))
        return word_id

        # TODO
        # 1. dialog.py 와 비슷한 기능을 하는 Twit 클래스 완성
        # 2. 이에 맞게 train.py model_.py 수정
        #.3. chat.py는 필요할까??? 저건 그냥 실시간으로 테스트하려고 model.predict()를 이용해 만들어 둔 클래스같다.
        #    저거랑 동일한 기능을 하는 클래스를 찾아서 하는게 어떨까?
        pass

    def next_batch(self, batch_size = 100):
        pass


Twit()





