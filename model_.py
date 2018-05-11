import tensorflow as tf


class Seq2Seq():
    logits = None
    outputs = None
    cost = None
    train_op = None

    def __init__(self,voca_size, n_hidden=128, n_layers=3):
        self.learning_rate = 0.001

        self.voca_size = voca_size
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.enc_input = tf.placeholder(tf.float32, [None, None, self.voca_size])
        self.dec_input = tf.placeholder(tf.float32, [None, None, self.voca_size])
        self.targets = tf.placeholder(tf.int64,[None,None])

        #softmax variables
        self.weights = tf.Variable(tf.ones([self.n_hidden,self.voca_size]),name='weights')
        self.bias = tf.Variable(tf.zeros([self.voca_size]),name='bias')
        self.global_step = tf.Variable(0,trainable=False,name='global_step')

        self._build_model()#encoding,decode cells generation

        self.saver = tf.train.Saver(tf.global_variables())

    def _build_model(self):
        encoder,decoder = self._build_cells()

        with tf.variable_scope('encode'):
            outputs, encode_states = tf.nn.dynamic_rnn(encoder, self.enc_input,dtype=tf.float32)
        with tf.variable_scope('decode'):
            outputs, decode_states = tf.nn.dynamic_rnn(decoder,self.dec_input, dtype=tf.float32,
                                                       initial_state=encode_states)
            #outputs = [batch_size, max_time, self.n_layers] size

        #softmax, cost, optimizer를 생성
        self.logits, self.cost, self.train_op = self._build_ops(outputs, self.targets)
        self.outputs = tf.argmax(self.logits, 2)

    def _cell(self, dropout_prob):
        cell = tf.nn.rnn_cell.BasicLSTMCell(self.n_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=dropout_prob)
        return cell

    def _build_cells(self,dropout_prob = 0.5):
        encoder = tf.nn.rnn_cell.MultiRNNCell([self._cell(dropout_prob) for _ in range(self.n_layers)])
        decoder = tf.nn.rnn_cell.MultiRNNCell([self._cell(dropout_prob) for _ in range(self.n_layers)])
        return encoder,decoder

    def _build_ops(self,outputs,targets):
        '''
        :param outputs: output of encoder(or decoder)
        :param targets: answer sheeet?
        :return: softmax result, cost, optimizer
        '''
        timesteps = tf.shape(outputs)[1]#무엇?
        outputs =tf.reshape(outputs, [-1,self.n_hidden])

        logits = tf.matmul(outputs, self.weights) + self.bias
        logits = tf.reshape(logits,[-1,timesteps,self.voca_size])

        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
        train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost,global_step=self.global_step)

        # for tensorboard visualization
        tf.summary.scalar('cost',cost)

        return logits, cost, train_op

    def train(self,session,enc_input,dec_input,targets):
        return session.run([self.train_op,self.cost],feed_dict = {
            self.enc_input : enc_input,
            self.dec_input : dec_input,
            self.targets : targets})

    def test(self, session, enc_input, dec_input, targets):
        prediction_check = tf.equal(self.outputs, self.targets)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        return session.run([self.targets, self.outputs, accuracy],
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input,
                                      self.targets: targets})

    def predict(self, session, enc_input, dec_input):
        return session.run(self.outputs,
                           feed_dict={self.enc_input: enc_input,
                                      self.dec_input: dec_input})

    def write_logs(self, session, writer, enc_input, dec_input, targets):
        merged = tf.summary.merge_all()

        summary = session.run(merged, feed_dict={self.enc_input: enc_input,
                                                 self.dec_input: dec_input,
                                                 self.targets: targets})

        writer.add_summary(summary, self.global_step.eval())



