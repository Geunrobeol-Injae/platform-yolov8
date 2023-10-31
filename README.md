# SageMaker Ver.
- 기존의 ```predict.py```를 분리하여 ```utils.py``` ```inference.py``` ```test.py``` 생성
  
    - ```utils.py``` : 함수 모음zip
    - ```inference.py``` : DetectionPredictor 클래스
    - ```test.py``` : 웹캠 실행
 
&nbsp;

### 실행 방법
```
python test.py
```

### 달라진 점
- 코드 분리
- 인자 없이 코드 실행
