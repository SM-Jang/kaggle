1. [KAGGLE](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)
캐글의 빙산 분류 챌린지를 CNN을 이용하여 도전하였습니다.

2. [참조](https://www.kaggle.com/code/devm2024/keras-model-for-beginners-0-210-on-lb-eda-r-d/notebook)
위의 Kaggler분의 노트북을 참고하여 pytorch로 모댈링하였습니다.

3. 데이터
'../input/iceberg'
- train.josn
- tesst.json
- sample_submission.csv

4. 실행방법
- mkdir weights
- python main.py

5. 세부사항
- model구조는 alexnet과 유사하게 설계
- 하이퍼 파라미터는 main file 안에서 조정가능
- 5 epoch 단위로 model을 저장
- train: 1604장, test: 8424장으로 훈련용 샘플이 매우 부족
- trainset을 3:1로 나누어 validset을 만듬(overfitting 관찰)
- CNN.png는 훈련시 모델의 성능 변화를 기록한 것
- pretrained model을 이용한 transfer learning을 적용해보면 좋을 것으로 예상
