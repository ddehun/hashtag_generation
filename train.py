import tensorflow as tf
import random
import math
import os
import numpy as np

from config import FLAGS
from model_ import Seq2Seq
from twit import Twit
import time
def train(twit, batch_size=100, epoch=100):
    print('훈련 시작~')
    model = Seq2Seq(twit.vocab_size)

    with tf.Session() as sess:
        # TODO: 세션을 로드하고 로그를 위한 summary 저장등의 로직을 Seq2Seq 모델로 넣을 필요가 있음
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("새로운 모델을 생성하는 중 입니다.")
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        total_batch = int(math.ceil(len(twit.twits) / float(batch_size))) #배치 개수

        print("Real training start!")
        print('{}번 해야 완료됨'.format(total_batch*epoch))
        t=time.time()
        for step in range(total_batch * epoch):

            enc_input, dec_input, targets = twit.next_batch(batch_size,False)
            _, loss = model.train(sess, enc_input, dec_input, targets)

            if (step + 1) % 10 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets)

                print('Step:', '%06d' % model.global_step.eval(),
                      'cost =', '{:.6f}'.format(loss),
                      'time  = {}min'.format(int(t-time.time())/60))
                t=time.time()
        print('training epoch end')
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print('최적화 완료!')


def test(twit, batch_size=100):
    print("\n=== 예측 테스트 ===")

    model = Seq2Seq(twit.vocab_size)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print("다음 파일에서 모델을 읽는 중 입니다..", ckpt.model_checkpoint_path)
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, dec_input, targets = twit.next_batch(batch_size,True)

        #DEC : beg, 단어
        #taget : 단어, EOS

        여기에 MAP metric 추가하기
        expect, outputs, accuracy, top_k = model.test(sess, enc_input, dec_input, targets)
        print('zz')
        print('타겟',targets)
        print(len(targets))
        print('top10',top_k.indices[:,0,:].shape)
        print(top_k.indices[:,0,:])
        len(set(123))

        expect = twit.decode(expect)

        outputs = twit.decode(outputs)

        #For human test of result
        for i in range(10):
            print('입력값 ',twit.test[i]['raw_text'])
            if '_PAD_' in expect[i]:
                ia = expect[i].index('_PAD_')
                expect[i] = expect[i][:ia]
            print('실제값 ', expect[i])
            if '_PAD_' in outputs[i]:
                ia = outputs[i].index('_PAD_')
                outputs[i] = outputs[i][:ia]
            print('예측값 ', outputs[i])


def main(_):
    print('시작~')
    twit = Twit()
    if FLAGS.train:
        print('model train start')
        train(twit, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)

    elif FLAGS.test:
        print('model test start')
        test(twit, batch_size=FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()