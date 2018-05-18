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
import pickle
import os

"""
트윗 데이터 전처리 파일
"""

#카카오 API : Not yet image tag used...
#MY_KEY = '91084af0198be3d0b93ea1d15cb7f989'

#def key_generation(image_url):
#    headers = {'Authorization':'KakaoAK {}'.format(MY_KEY)}


class Twit():
    MAIN_KEY = ['hashtags','text','hashtags']
    def __init__(self):
        self.hashtag_max = 0
        self.max_len_input = 0

        self.data_path = FLAGS.twit_path
        self.twits, self.test = self.load_data() # 80,10,10 for train,dev,test
        self.voca = self.build_voca()
        self.vocab_size = len(self.voca.keys())

        self.vec_generation() # 각 트윗의 tokens을 ids로 변환하여 저장(data embedding)

        self.curr_tag = 0 # 1twit - 1tag matching을 위해, 현재 읽는 twit의 tag idx를 저장
        print('end')


        self._idx_in_epoch = 0

    def load_data(self):
        """
        raw data에서 MAIN_KEY만을 추출한 뒤 가공한다.
        text tokenize, VISION API(api not yet)
        """
        if os.path.exists('data.pickle'):
            with open('data.pickle','rb') as f:
                print("이미 존재하는 정형 트윗 데이터 이용")
                data,tests = pickle.load(f)
                for i in data:
                    if len(i['hashtags'])>self.hashtag_max:
                        self.hashtag_max = len(i['hashtags'])
                    if len(i['tokens'])>self.max_len_input:
                        self.max_len_input = len(i['tokens'])
                return data,tests

        with open(self.data_path,encoding='utf8') as f:
            raw_data = json.load(f)
        data = []
        for i,t in enumerate(raw_data):
            t['text'] = t['text'].replace('#','').lower()
            lowered_tag = []
            for j in t['hashtags']:
                lowered_tag.append(j.lower())
            data.append({'raw_text':t['text'],'hashtags':lowered_tag,'image':t['media']})
            pos = nt.pos_tag(nt.word_tokenize(t['text']))
            data[-1]['tokens'] = [i[0] for i in pos]

            while 'https' in data[-1]['tokens']: #rejoin the tokenized url
                idx = data[-1]['tokens'].index('https')
                data[-1]['tokens'][idx] = data[-1]['tokens'][idx] + data[-1]['tokens'][idx+1] + data[-1]['tokens'][idx+2]
                data[-1]['tokens'] = data[-1]['tokens'][:idx+1] + data[-1]['tokens'][idx+3:]

        tests = data[0:len(data)//10]
        #data = data[len(data)//10:]

        for i in data:
            if len(i['hashtags']) > self.hashtag_max:
                print(i['hashtags'])
                self.hashtag_max = len(i['hashtags'])
            if len(i['tokens']) > self.max_len_input:
                self.max_len_input = len(i['tokens'])

        with open('data.pickle','wb') as f:
            pickle.dump((data,tests),f)
        return data,tests

    def build_voca(self):
        # TODO
        # BOW 외의 word embedding 구현 시, 이곳에 추가할 것
        voca = Counter()
        for d in self.twits:
            for w in d['tokens']:
                voca[w]+=1

        pairs = sorted(voca.items(), key=lambda x: (-x[1],x[0]))
        words, _ = list(zip(*pairs))
        self.voca_list = list(words) + ['_UNK_','_BEG_','_EOS_','_PAD_']
        word_id = dict(zip(words, range(len(words))))
        word_id['_UNK_'] = len(words)
        word_id['_BEG_'] = len(words)+1
        word_id['_EOS_'] = len(words)+2
        word_id['_PAD_'] = len(words)+3

        self.UNK_KEY = word_id['_UNK_']  # 단어장에 없는 단어
        self.BEG_KEY = word_id['_BEG_']  # ENCODING 시작
        self.EOS_KEY = word_id['_EOS_']  # ENCODING 종료
        self.PAD_KEY = word_id['_PAD_']  # PADDING
        return word_id

    def vec_generation(self):
        for idx,t in enumerate(self.twits):
            self.twits[idx]['vec'] = self.tokens_to_id(t['tokens'])
            self.twits[idx]['tag_vec'] = self.tokens_to_id(t['hashtags'])
        for idx,t in enumerate(self.test):
            self.test[idx]['vec'] = self.tokens_to_id(t['tokens'])
            self.test[idx]['tag_vec'] = self.tokens_to_id(t['hashtags'])

    def next_batch(self, batch_size = 100, test=False):
        start = self._idx_in_epoch
        enc_input = []
        dec_input = []
        target = []

        if not test:
            if self._idx_in_epoch + batch_size < len(self.twits) -1:
                self._idx_in_epoch = self._idx_in_epoch + batch_size
            else:
                self._idx_in_epoch = 0
            batch_set = self.twits[start:start+batch_size]

        else:
            if self._idx_in_epoch + batch_size < len(self.test) -1:
                self._idx_in_epoch = self._idx_in_epoch + batch_size
            else:
                self._idx_in_epoch = 0
            batch_set = self.test[start:start+batch_size]

        self.hashtag_max = 0
        max_len_input,self.hashtag_max = self._max_len(batch_set)

        for t in batch_set:
            a,b,c = self.transform(t['vec'],t['tag_vec'],max_len_input)
            enc_input.append(a)
            dec_input.append(b)
            target.append(c)

        return enc_input, dec_input, target

    def _max_len(self, batch_set):
        max_len_input = 0
        max_len_output = 0

        for i in range(0, len(batch_set)):
            len_input = len(batch_set[i]['vec'])
            len_output = len(batch_set[i]['tag_vec'])
            if len_input > max_len_input:
                max_len_input = len_input
            if len_output > max_len_output:
                max_len_output = len_output

        return max_len_input, max_len_output + 1

    def transform(self,input,output,input_max):
        enc_input = input + [self.PAD_KEY]*max(0,input_max - len(input))
        dec_input = [self.BEG_KEY] + output + [self.PAD_KEY]*max(0, self.hashtag_max- len(output))
        target = output + [self.EOS_KEY] + [self.PAD_KEY]*max(0, self.hashtag_max - len(output))

        enc_input = np.eye(self.vocab_size)[enc_input]
        dec_input = np.eye(self.vocab_size)[dec_input]

        return enc_input, dec_input, target

    def tokens_to_id(self,tokens):
        '''
        String to vector
        '''
        ids = [self.voca[i] if i in self.voca else self.UNK_KEY for i in tokens]
        return ids

    def decode(self,indices,string=False):
        '''
        vector to string
        '''
        tok = [[self.voca_list[i] for i in dec]for dec in indices]

        return tok
