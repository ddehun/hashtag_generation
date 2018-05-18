# 해시태그 생성 

LSTM Seq2Seq을 기본으로 한 해시태그 생성 모델

참고 링크 : [LSTM을 이용한 chatbot](https://github.com/golbin/TensorFlow-Tutorials/tree/master/10%20-%20RNN/ChatBot)


## Guide
* training : `python train.py --train` 명령어를 통해 실행 가능. epoch 조정은 `Config.py`파일의 `epoch` 변수를 통해 가능합니다
* test : `python train.py --test` 명령어를 통해 가능
* 직접 트윗을 적어서 해보기 : training을 완료한 이후, `tagger.py`를 실행

## Package Structure
- `model.py` : LSTM 모델
- `train.py` : 모델의 훈련 및 테스트를 위한 스크립트
- `Config.py` : hyperparameter를 조절하기 위한 파일
` 'twit.py' : 트윗 데이터를 읽고 전처리하는 파일
- `tagger.py` : 훈련된 모델을 통해 직접 해시태그 추천을 실행해볼 수 있는 파일
- `model`,`logs` : 훈련된 모델 및 로그를 저장하는 폴더


## Etc...
- repository를 다운받은 후, 트위터 데이터인 `dataset.json`파일을 파이썬 코드들과 동일한 폴더에 위치
- 현재는 트위터 전처리 과정을 반복하지 않기 위해, 한번 처리된 트윗 데이터는 `data.pickle` 이름으로 저장되고 이후 재사용됩니다. 데이터가 추가되거나 데이터를 바꿀 필요가 있는 경우 유의해야 합니다
