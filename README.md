# BoostCamp AI Tech level 2 데이터 제작 프로젝트-CV14조


## Member🔥
| [김지훈](https://github.com/kzh3010) | [원준식](https://github.com/JSJSWON) | [송영섭](https://github.com/gih0109) | [허건혁](https://github.com/GeonHyeock) | [홍주영](https://github.com/archemist-hong) |
| :-: | :-: | :-: | :-: | :-: |
| <img src="https://avatars.githubusercontent.com/kzh3010" width="100"> | <img src="https://avatars.githubusercontent.com/JSJSWON" width="100"> | <img src="https://avatars.githubusercontent.com/gih0109" width="100"> | <img src="https://avatars.githubusercontent.com/GeonHyeock" width="100"> | <img src="https://avatars.githubusercontent.com/archemist-hong" width="100"> |

## Index
* [Project](#project)
* [Team role](#team-role)
* [Procedures](#procedures)
* [Command](#command)
* [Wrap UP Report](#wrap-up-report)  

## Project

- 배경: 학습 데이터 추가 및 수정을 통한 이미지 속 글자 검출 성능 개선
- 주제: OCR task 중 글자 검출(text detection) task의 성능 개선(모델 관련 부분 변경 불가)

<img width="50%" src="./images/프로젝트개요(흰 배경).png"/>

- Input: 글자가 포함된 전체 이미지
- Output: bbox 좌표가 포함된 UFO Format(Upstage Format for OCR)
- 평가 방법: DetEval

## Team role
- 김지훈: data 실험, albumentation 관련 실험
- 원준식: data 추가 실험, validation score 추가
- 송영섭: 대회 실험 관리 및 진행, data 및 augmentation 실험
- 허건혁: data Visual 개발, data annotation merge 개발, straug 실험
- 홍주영: Opimization, TTA, data 관련 실험


## Procedures
대회 기간: 2022.12.08. ~ 2022.12.15.

| 날짜 | 내용 |
| :---: | :---: |
| 12.05 ~ 12.09 | OCR 이론 학습, data 제작
| 12.10 ~ 12.14 | model 실험
| 12.14 ~ 12.15 | fine tuning을 이용한 성능 개선

## Command

- train
```
python train.py
```

- streamlit
```
streamlit run visual.py
```

## Wrap UP Report
- [Report](https://www.notion.so/Wrap-Up-Report-CV-14-cd0961e6516c45dd97cc6535a8cb9586)
