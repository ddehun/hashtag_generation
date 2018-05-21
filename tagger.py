import tensorflow as tf
import numpy as np
import math
import sys

from config import FLAGS
from model_ import Seq2Seq as model
from twit import Twit
import nltk as nt

'''
훈련한 모델을 통해 해시태그 생성을 직접 해볼 수 있는 파일
'''

class Tagger():
    def __init__(self,data_path,train_dir):
        self.twit = Twit()
        print("Model upload....")
        self.model = model(self.twit.vocab_size)
        self.sess = tf.Session()
        print('checkpoint upload...')
        ckpt = tf.train.get_checkpoint_state(train_dir)
        print('model restore,,,')
        self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print('initialize end!')

    def run(self):
        sys.stdout.write('> ')
        sys.stdout.flush()
        line = sys.stdin.readline()

        while line:
            tags = self.recommend({'text':line.strip()})
            for i in tags:
                print('#'+i,end=' ')
            print('\n')
            sys.stdout.write('\n> ')
            sys.stdout.flush()
            line = sys.stdin.readline()

    def recommend(self, post):
        #언젠가 유저가 그림도 줄 수 있으니, string이 아닌 dict으로 처리
        print("유저 POST : {}".format(post['text']))
        msg = post['text'].replace('#','').lower()
        tok = nt.pos_tag(nt.word_tokenize(msg))
        tok = [i[0] for i in tok]
        enc_input = self.twit.tokens_to_id(tok)
        dec_input = []

        curr_seq = 0
        for i in range(1):#FLAGS.max_decode_len):
            top_k, outputs = self._decode(enc_input, dec_input)
            candis = []
            top_k_indices = top_k.indices[0][0]
            final_recommend = []
            for candi in top_k_indices:
                if candi not in [self.twit.EOS_KEY,self.twit.PAD_KEY,self.twit.UNK_KEY]:
                    candis.append(self.twit.voca_list[candi])
                    final_recommend.append(candi)
        final_recommend = final_recommend[:FLAGS.map_k]
        reply = self.twit.decode([final_recommend],True)
        return reply[0]

    def _decode(self,enc_input,dec_input):
        if type(dec_input) == np.ndarray:
            dec_input = dec_input.tolist()
        input_len = int(math.ceil((len(enc_input)+1)*1.5))
        enc_input, dec_input, _ = self.twit.transform(enc_input, dec_input, input_len)

        return self.model.predict(self.sess, [enc_input], [dec_input])


def main(_):
    print("삐빅....")
    re = Tagger(FLAGS.voc_path, FLAGS.train_dir)
    print("GOGO")
    re.run()

if __name__ =='__main__':
    tf.app.run()