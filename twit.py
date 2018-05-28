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
import random
import re
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
        self.data_path = FLAGS.twit_path
        self.twits, self.test = self.load_data() # 80,10,10 for train,dev,test
        self.voca = self.build_voca()
        self.vocab_size = len(self.voca.keys())

        self.vec_generation() # 각 트윗의 tokens을 ids로 변환하여 저장(data embedding)
        self.curr_tag = 0 # 1twit - 1tag matching을 위해, 현재 읽는 twit의 tag idx를 저장
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
                return data,tests

        with open(self.data_path,encoding='utf8') as f:
            raw_data = json.load(f)
        data = []
        print("트위터 새로~")
        for i,t in enumerate(raw_data):
            t['text'],tmp_url=self.extract_url(t['text'].lower())
            t['text'] = re.sub(r'[^\w]',' ',t['text']).lower()
            lowered_tag = [j.lower() for j in t['hashtags']]
            if len(lowered_tag)==0: continue

            data.append({'raw_text':t['text'],'hashtags':lowered_tag,'image':t['media']})
            pos = nt.pos_tag(nt.word_tokenize(t['text']))

            data[-1]['tokens'] = [_pos[0] for _pos in pos]
            data[-1]['tokens'] += tmp_url
            for tok in tmp_url: data[-1]['raw_text'] += ' '+tok

        for i in range(10):
            print(data[i]['tokens'])
        tests = data[int(len(data)*9/10):]
        data = data[:int(len(data)*9/10)]
        print('훈련 {}개, 테스트 {}개'.format(len(data),len(tests)))
        print('섞섞')
        random.shuffle(data)
        random.shuffle(tests)

        with open('data.pickle','wb') as f:
            pickle.dump((data,tests),f)
        return data,tests

    def extract_url(self,strs):
        url = []
        text = ''
        for i,tok in enumerate(strs.split()):
            if 'https' in tok:
                url.append(tok)
            else:
                text += ' '+tok
        if text[:3] ==' rt': #retweet mark remove
            text = ' '.join(text.strip().split()[2:])
        return text.strip(), url

    def build_voca(self):
        # TODO
        # BOW 외의 word embedding 구현 시, 이곳에 추가할 것
        voca = Counter()
        for d in self.twits:
            for w in d['tokens']:
                voca[w]+=1

        #작게 나온 단어들은 단어장에서 제외
        pairs = sorted(voca.items(), key=lambda x: (-x[1],x[0]))
        for i,qwer in enumerate(pairs):
            if qwer[1]== FLAGS.minimum_cnt: break
        pairs=pairs[:i-1]

        words, _ = list(zip(*pairs))
        word_id = dict(zip(words, range(len(words))))

        self.voca_list = list(words) + ['_UNK_', '_BEG_', '_EOS_', '_PAD_']
        self.UNK_KEY = word_id['_UNK_'] = len(words)
        self.BEG_KEY = word_id['_BEG_'] = len(words)+1
        self.EOS_KEY = word_id['_EOS_'] = len(words)+2
        self.PAD_KEY = word_id['_PAD_'] = len(words)+3

        self.DEFINED = [self.UNK_KEY,self.BEG_KEY,self.EOS_KEY,self.PAD_KEY]
        return word_id

    def vec_generation(self):
        for idx,t in enumerate(self.twits):
            self.twits[idx]['vec'] = self.tokens_to_id(t['tokens'])
            self.twits[idx]['vec'].reverse()
            self.twits[idx]['tag_vec'] = self.tokens_to_id(t['hashtags'])
        for idx,t in enumerate(self.test):
            self.test[idx]['vec'] = self.tokens_to_id(t['tokens'])
            self.test[idx]['vec'].reverse()
            self.test[idx]['tag_vec'] = self.tokens_to_id(t['hashtags'])

    def next_batch(self, batch_size = 100, test=False):
        start = self._idx_in_epoch
        enc_input = []
        dec_input = []
        target = []

        batch_set = []

        if not test:
            while len(batch_set)<batch_size:
                if self._idx_in_epoch == len(self.twits)-1:
                    self._idx_in_epoch = 0
                    self.curr_tag = 0
                if self.curr_tag == len(self.twits[self._idx_in_epoch]['tag_vec']):
                    self.curr_tag = 0
                    self._idx_in_epoch += 1

                t=self.twits[self._idx_in_epoch].copy()
                t['tag_vec'] = [t['tag_vec'][self.curr_tag]]

                batch_set.append(t)
                self.curr_tag+=1
        if test:
            batch_set = self.test

        max_len_input,self.hashtag_max = self._max_len(batch_set)

        if not test:
            for t in batch_set:
                for tag in t['tag_vec']:
                    a,b,c = self.transform(t['vec'],[tag],max_len_input)
                    enc_input.append(a)
                    dec_input.append(b)
                    target.append(c)
        if test:
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
        dec_input = [self.BEG_KEY] + output + [self.PAD_KEY]*max(0, self.hashtag_max- len(output)-1)
        target = output + [self.EOS_KEY] + [self.PAD_KEY]*max(0, self.hashtag_max - len(output)-1)

        enc_input = np.eye(self.vocab_size)[enc_input]
        dec_input = np.eye(self.vocab_size)[dec_input]

        return enc_input, dec_input, target

    def tokens_to_id(self,tokens):
        #String to vector
        ids = [self.voca[i] if i in self.voca else self.UNK_KEY for i in tokens]
        return ids

    def decode(self,indices,string=False):
        #vector to string
        tok = [[self.voca_list[i] for i in dec]for dec in indices]
        return tok

if __name__ == '__main__':
    t = Twit()
    print(len(t.voca_list))
    for i in t.twits:
        print(i)
