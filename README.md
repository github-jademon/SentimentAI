# SentimentAI (감정분석 AI)

![SentimentAI](https://github.com/github-jademon/SentimentAI/assets/79764169/c097c010-22b3-40b1-afa7-bc63cd3d981c)
이 프로젝트는 텍스트 감정분석을 수행하는 감정분석 AI를 개발한 것입니다. 해당 AI는 주어진 텍스트 문장을 입력으로 받아 '긍정', '중립', '부정'으로 감정을 예측합니다. 

## 목차

1. [프로젝트 설명](#프로젝트-설명)
2. [데이터셋](#데이터셋)
3. [실행환경](#실행환경)
4. [프로젝트 설치 및 실행 방법](#프로젝트-설치-및-실행-방법)
5. [프로젝트 사용 방법](#프로젝트-사용-방법)
6. [참고 자료](#참고-자료)
7. [라이센스](#라이센스)

## 프로젝트 설명

이 프로젝트는 Python과 TensorFlow를 사용하여 텍스트 감정분석 AI를 개발한 것입니다. 감정분석 AI 모델은 LSTM(Long Short-Term Memory)과 임베딩 층을 활용하여 텍스트 시퀀스를 분석하고 긍정, 중립, 또는 부정으로 분류합니다.

## 데이터셋
- [AIhub](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=263)
- [Tistory(sig413)](https://sig413.tistory.com/5)

## 실행환경
- [Cuda (release 11.8, V11.8.89)](https://developer.nvidia.com/)
- [Anaconda (conda 23.7.1)](https://www.anaconda.com/)
- [Python (Python 3.11.4)](https://www.python.org/)
- [Tensorflow (2.13.0)](https://www.tensorflow.org/?hl=ko)

## 프로젝트 설치 및 실행 방법

1. 먼저, Python 및 필요한 라이브러리를 설치합니다.
   ```bash
   pip install pandas scikit-learn tensorflow
   ```
2. `data.csv` 파일을 준비하여 훈련 및 테스트 데이터를 저장합니다.
   - `data.csv` 파일은 '발화문'과 해당 발화문의 감정을 나타내는 '상황' 열로 구성되어야 합니다.
   - '상황' 열의 감정은 'happiness', 'angry', 'anger', 'disgust', 'fear', 'neutral', 'sadness', 'sad', 'surprise' 등의 문자열로 표현됩니다.

3. `main.py` 스크립트를 실행하여 모델을 훈련합니다.

## 프로젝트 사용 방법
`main.py` 스크립트를 실행하여 감정 예측을 진행합니다.

텍스트 문장을 입력하면 AI 모델이 해당 문장의 감정을 예측하여 출력합니다.
   - 예측된 감정은 '긍정', '중립', '부정'으로 분류됩니다.

![SentimentAI](https://github.com/github-jademon/SentimentAI/assets/79764169/445a28e4-f421-4d33-94f7-3966f48e182f)

## 참고 자료

1. https://wikidocs.net/22894
2. https://sig413.tistory.com/5

이 프로젝트에 관한 문의 사항이나 버그 리포트는 [j2python@gmail.com]로 보내주세요.
