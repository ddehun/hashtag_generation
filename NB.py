from sklearn.naive_bayes import  GaussianNB,MultinomialNB
import numpy as np
from twit import Twit
import random

"""
Baseline model. Naive Bayesian
"""
def NB():
    #clf = GaussianNB()
    clf = MultinomialNB()
    twit = Twit()

    X,y,X_test,y_test = [],[],[],[]

    for t in twit.twits:
        for i,h in enumerate(t['tag_vec']):
            if h in [twit.PAD_KEY,twit.UNK_KEY]: continue
            X.append(t['vec'])
            y.append([h])

    for t in twit.test:
        X_test.append(t['vec'])
        y_test.append([])
        for i,h in enumerate(t['tag_vec']):
            if h in [twit.PAD_KEY,twit.UNK_KEY]: continue
            y_test[-1].append(h)

    print("CHECK COPLETE")
    X = np.array(X)
    y = np.array(y)

    clf.fit(X,y)

    res = clf.predict(np.array(X_test))
    prob = clf.predict_log_proba(np.array(X_test))

    samples = random.sample(range(len(X_test)),10)
    for i in samples:
        t=' '.join(twit.decode([X_test[i]])[0]) #raw text
        idx = t.index('_PAD_')
        t=t[:idx] #text padding 제거

        top_n = prob[i].argsort()[-len(y_test[i]):][::-1] #정답과 같은 개수의 추천 받는다
        print('원문 : ',t)
        print('예측 : ',twit.voca_list[res[i]])
        print('예측 후보 : ',[twit.voca_list[w] for w in top_n])
        print('정답 : ',[twit.voca_list[j] for j in y_test[i]])
        print()

if __name__ == '__main__':
    NB()




