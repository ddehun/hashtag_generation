'''
Evaluation 관련 util file
'''
from twit import Twit
from config import FLAGS


def mAP(answer,top_n,twit):
    total_precision = 0
    assert len(answer)==len(top_n)
    for ans,top in zip(answer,top_n):
        ans, top = remove_predefined(list(ans),list(top),twit)
        top = top[:FLAGS.map_k]

        hit, precision_sum = 0, 0
        for i in range(len(top)):
            if top[i] in ans:
                hit += 1
                precision_sum += hit/(i+1)
            if hit == len(ans): break
        total_precision += precision_sum/len(ans)#  FLAGS.map_k

    return total_precision/len(top_n)


def remove_predefined(ans,top,twit):
    while len(set(top) & set(twit.DEFINED)) != 0 or len(set(ans) & set(twit.DEFINED)) != 0:
        if twit.UNK_KEY in top: top.remove(twit.UNK_KEY)
        if twit.BEG_KEY in top: top.remove(twit.BEG_KEY)
        if twit.EOS_KEY in top: top.remove(twit.EOS_KEY)
        if twit.PAD_KEY in top: top.remove(twit.PAD_KEY)
        if twit.UNK_KEY in ans: ans.remove(twit.UNK_KEY)
        if twit.BEG_KEY in ans: ans.remove(twit.BEG_KEY)
        if twit.EOS_KEY in ans: ans.remove(twit.EOS_KEY)
        if twit.PAD_KEY in ans: ans.remove(twit.PAD_KEY)
    if len(top) < FLAGS.map_k:
        raise ValueError("{} is not enough to recommend tag! UNK or EOS is too many!".format(FLAGS.recommend_count))
    return ans,top


def precision(a,b):
    # a : USER's answer
    # b : real answer
    right = list(set(a)&set(b))
    return len(right)/len(a)

if __name__ == '__main__':
    a=[[1,2,4]]
    b=[[1,4,8,2,9,1,4,8,2,9]]
    t=Twit()
    print(mAP(a,b,t))