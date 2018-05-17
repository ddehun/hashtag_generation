import tensorflow as tf
import numpy as np
import math
import sys

from config import FLAGS
from model_ import Seq2Seq as model
from twit import Twit

'''
훈련한 모델을 통해 해시태그 생성을 직접 해볼 수 있는 파일 (Not developed yet)
참고한 예제파일의 chat.py와 비슷한 파일
'''

class Tagger():
    def __init__(self,data_path,train_dir):
        self.twit = Twit()
        self.model = model(self.twit.vocab_size)
        ckpt = tf.train.get_checkpoint_mtimes(train_dir)
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)

if __name__ =='__main__':
    print(123)