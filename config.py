import tensorflow as tf

tf.app.flags.DEFINE_string("train_dir", "./model", "학습한 신경망을 저장할 폴더")
tf.app.flags.DEFINE_string("log_dir", "./logs", "로그를 저장할 폴더")
tf.app.flags.DEFINE_string("ckpt_name", "conversation.ckpt", "체크포인트 파일명")

tf.app.flags.DEFINE_boolean("train", False, "학습을 진행합니다.")
tf.app.flags.DEFINE_boolean("test", True, "테스트를 합니다.")
tf.app.flags.DEFINE_boolean("data_loop", True, "작은 데이터셋을 실험해보기 위해 사용합니다.")
tf.app.flags.DEFINE_integer("batch_size", 100, "미니 배치 크기")
tf.app.flags.DEFINE_integer("epoch", 600, "총 학습 반복 횟수")#800

tf.app.flags.DEFINE_string("data_path", "./data/dialog.txt", "대화 파일 위치")
tf.app.flags.DEFINE_string("voc_path", "./data/voca.txt", "어휘 사전 파일 위치")
tf.app.flags.DEFINE_boolean("voc_test", False, "어휘 사전을 테스트합니다.")
tf.app.flags.DEFINE_boolean("voc_build", False, "주어진 대화 파일을 이용해 어휘 사전을 작성합니다.")

tf.app.flags.DEFINE_integer("max_decode_len", 1, "최대 디코더 셀 크기 = 최대 답변 크기.")
tf.app.flags.DEFINE_string('twit_path','dataset.json','트위터 데이터 파일')
tf.app.flags.DEFINE_string('twit_img_path','dataset_with_img_tag.json','트위터 데이터 파일 with ImgTags')
tf.app.flags.DEFINE_integer('minimum_cnt',5,'단어장 만들 떄 최소 출현 횟수')
tf.app.flags.DEFINE_integer('recommend_count',15,'추천하는 해시태그 개수') #UNK, EOS등이 추천되는 경우가 빈번하므로, 크게 잡는다
tf.app.flags.DEFINE_integer('map_k',10,'mAP에 사용하는 k')
FLAGS = tf.app.flags.FLAGS