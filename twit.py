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
        self.twits = self.load_data()
        self.voca = self.build_voca()

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
            while 'https' in data[-1]['tokens']: #rejoin the url
                idx = data[-1]['tokens'].index('https')
                data[-1]['tokens'][idx] = data[-1]['tokens'][idx] + data[-1]['tokens'][idx+1] + data[-1]['tokens'][idx+2]
                data[-1]['tokens'] = data[-1]['tokens'][:idx+1] + data[-1]['tokens'][idx+3:]
        return data

    def build_voca(self):
        pass


Twit()





