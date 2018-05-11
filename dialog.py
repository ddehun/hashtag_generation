import tensorflow as tf
import numpy as np
import re
from config import FLAGS


class Symbol():
    _EMP_ = '_EMP_'  # 빈칸
    _BEG_ = '_BEG_'  # 디코드 입력의 시작
    _EOS_ = '_EOS_'  # 디코드 입력의 종료
    _OOV_ = '_OOV_'  # VOCA에 없는 단어

    PRE_DEFINED_ = [_EMP_, _BEG_, _EOS_, _OOV_]

    _EMP_ID = 0
    _BEG_ID = 1
    _EOS_ID = 2
    _OOV_ID = 3


class Dialog():
    '''
    voca 및 embedded word를 생성하고, dialog data를 읽는 Utility Class
    '''
    def __init__(self):
        self.vocab_list = {}
        self.vocab_dict = {}
        self.vocab_size = 0
        self.examples = []
        self.symbol = Symbol()
        self._index_in_epoch = 0

    def decode(self, indices, string=False):
        tokens = [[self.vocab_list[i] for i in dec] for dec in indices]

        if string:
            return self._decode_to_string(tokens[0])
        else:
            return tokens

    def _decode_to_string(self, tokens):
        text = ' '.join(tokens)
        return text.strip()

    def cut_eos(self, indices):
        eos_idx = indices.index(self.symbol._EOS_)
        return indices[:eos_idx]

    def is_eos(self, voc_id_):
        return voc_id_ == self.symbol._EOS_ID

    def is_defined(self, voc_id):
        return voc_id in self.symbol.PRE_DEFINED

    def _max_len(self, batch_set):
        max_len_input = 0
        max_len_output = 0

        for i in range(0,len(batch_set),2):
            len_input = len(batch_set[i])
            len_output = len(batch_set[i + 1])
            if len_input > max_len_input:
                max_len_input = len_input
            if len_output > max_len_output:
                max_len_output = len_output

        return max_len_input, max_len_output + 1

    def _pad(self, seq, max_len, start=None, eos=None):
        if start:
            padded_seq = [self.symbol._BEG_ID] + seq
        elif eos:
            padded_seq = seq + [self.symbol._EOS_ID]
        else:
            padded_seq = seq

        if len(padded_seq) < max_len:
            return padded_seq + ([self.symbol._EMP_ID] * (max_len - len(padded_seq)))
        else:
            return padded_seq

    def _pad_left(self, seq, max_len):
        if len(seq) < max_len:
            return ([self.symbol._EMP_ID]*(max_len - len(seq))) + seq
        return seq

    def transform(self, input, output, input_max, output_max):
        enc_input = self._pad(input, input_max)
        dec_input = self._pad(output, output_max, start=True)
        target = self._pad(output, output_max, eos=True)

        enc_input.reverse()

        enc_input = np.eye(self.vocab_size)[enc_input]
        dec_input = np.eye(self.vocab_size)[dec_input]

        return enc_input, dec_input, target

    def next_batch(self, batch_size):
        enc_input = []
        dec_input = []
        target = []

        start = self._index_in_epoch

        if self._index_in_epoch + batch_size < len(self.examples) - 1:
            self._index_in_epoch = self._index_in_epoch + batch_size
        else:
            self._index_in_epoch = 0

        batch_set = self.examples[start:start+batch_size]

        # 작은 데이터셋을 실험하기 위한 꼼수
        # 현재의 답변을 다음 질문의 질문으로 하고, 다음 질문을 답변으로 하여 데이터를 늘린다.
        if FLAGS.data_loop is True:
            batch_set = batch_set + batch_set[1:] + batch_set[0:1]

        # TODO: 구글처럼 버킷을 이용한 방식으로 변경
        # 간단하게 만들기 위해 구글처럼 버킷을 쓰지 않고 같은 배치는 같은 사이즈를 사용하도록 만듬
        max_len_input, max_len_output = self._max_len(batch_set)

        for i in range(0, len(batch_set) - 1, 2):
            enc, dec, tar = self.transform(batch_set[i], batch_set[i+1],
                                           max_len_input, max_len_output)

            enc_input.append(enc)
            dec_input.append(dec)
            target.append(tar)

        return enc_input, dec_input, target

    def tokens_to_ids(self, tokens):
        ids = [self.vocab_dict[t] if t in self.vocab_dict else self.symbol._OOV_ID for t in tokens]
        return ids

    def ids_to_tokens(self,ids):
        tokens=[self.vocab_list[i] for i in ids]
        return tokens

    def load_examples(self,data_path):
        self.examples = []

        with open(data_path, 'r') as content_file:
            for line in content_file.readlines():
                tokens = self.tokenizer(line.strip())
                ids = self.tokens_to_ids(tokens)
                self.examples.append(ids)


    def tokenizer(self,sent):
        words = []
        _TOKEN_RE_ = re.compile("([.,!?\"':;)(])")

        for fragment in sent.strip().split():
            words.extend(_TOKEN_RE_.split(fragment))

        return [w for w in words if w]

    def build_vocab(self, data_path, vocab_path):
        with open(data_path, 'r') as content_file:
            content = content_file.readlines()
            content = ''.join(content)
            words = self.tokenizer(content)
            words = list(set(words))

        with open(vocab_path, 'w') as vocab_file:
            for w in words:
                vocab_file.write(w + '\n')

    def load_vocab(self, vocab_path):
        self.vocab_list = self.symbol.PRE_DEFINED_ + []

        with open(vocab_path, 'r') as vocab_file:
            for line in vocab_file.readlines():
                self.vocab_list.append(line.strip())

        self.vocab_dict = {n: i for i, n in enumerate(self.vocab_list)}
        self.vocab_size = len(self.vocab_list)

def main(_):
    dialog = Dialog()

    if FLAGS.data_path and FLAGS.voc_test:
        print("다음 데이터로 어휘 사전을 테스트합니다.", FLAGS.data_path)
        dialog.load_vocab(FLAGS.voc_path)
        dialog.load_examples(FLAGS.data_path)

        enc, dec, target = dialog.next_batch(10)
        print(target)
        enc, dec, target = dialog.next_batch(10)
        print(target)

    elif FLAGS.data_path and FLAGS.voc_build:
        print("다음 데이터에서 어휘 사전을 생성합니다.", FLAGS.data_path)
        dialog.build_vocab(FLAGS.data_path, FLAGS.voc_path)

    elif FLAGS.voc_test:
        dialog.load_vocab(FLAGS.voc_path)
        print(dialog.vocab_dict)


if __name__ == "__main__":
    tf.app.run()
