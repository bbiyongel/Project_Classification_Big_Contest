# '블레이드앤소울' 게임 유저 이탈 예측 문제

## 1. Introduction
- 주제 : 유저들의 게임 활동 데이터를 활용하여 향후 게임 서비스에서 이탈하는 시점 예측하기
- 이탈예측
  - 고객 관계 관리(CRM)분야에서 중요하게 다루는 문제
  - 비용 효율적
    - 신규 고객 유입을 위해 필요한 비용 > 고객 유지에 필요한 비용
  - 정확한 예측이 근본적으로 어려움
    - 저마다 다른 욕구 및 선호도
    - 데이터에서 확인 불가능한 외적인 문제로 인한 이탈
    - 빠른 변화
    - 이탈에 대한 기준 모호
- 분석 대상
  - Blade&Soul
    - 2012년 6월부터 엔씨소프트에서 서비스 중인 무협 MMORPG
- 데이터 분석 측면에서 게임이 갖는 매력
  - 현실과 매우 유사한 가상세계
  - 현실에서 접하기 힘든 고품질 데이터

### 1.1 문제 및 평가 기준 소개
- 문제: 게임 유저의 향후 이탈 여부 및 시점 예측
- ‘이탈’의 정의: **“4주 이상 접속하지 않으면 이탈로 판단한다.”**
  - 제공 데이터 시점 이후 12주 동안의 접속 이력으로 판단
- Target label: 이탈 여부, 이탈 시점에 따른 4개 클래스
  - 1주 이내 이탈: Week
  - 4주 이내 이탈: Month
  - 8주 이내 이탈: 2Month
  - 잔존: Retained
- 평가
  - 예측 성능 : F1 score
    - 각 클래스별 precision과 recall을 계산한 후 전체에 대한 조화 평균
### 1.2 데이터 소개

- 데이터 규모
  - 학습 데이터 : (계정 아이디 기준) 10만 명의 게임 활동 데이터
  - 평가 데이터 : (계정 아이디 기준) 4만 명의 게임 활동 데이터
- 데이터 종류
  - 주요활동정보 : 게임내에서 활동하는 주요 활동량을 유저별로 1주일 단계로 집계한 정보
  - 결제정보 : 사용자가 게임 활동을 위해 결제한 정보를 1주일 단위로 집계
  - 사회관계정보 : 유저 간에 상호 작용 및 사회 관계에 대한 정보
  **단, 사회관계정보에는 이탈 예측 대상자가 아닌 유저들도 포함되어 있음**

## 2. EDA & Feature Engineering

### 2.1 [EDA](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/2_EDA.ipynb) & Feature Enginerring Overview

  - Overview
    - 각각 다른 schema를 가진 data에서 예측 대상인 유저 id를 기준으로 하여 feature varible 생성

1) activity, payment data의 경우
      - 한 유저가 week 별로 여러 개의 관측치를 가지고 있음
      -> 이를 column마다 week별 변수로 확장(w1~w8, groupby하거 이들의 비율을 구해 변수 생성
2) party, guild, trade data의 경우
    - party와 guild의 경우 개인 유저 레벨이 아닌 그룹(party, guild) 레벨의 data
    - 전체 사회관계를 담기 위해 train id에 대해 sampling되어 있지 않음
-> party 멤버 id와 guild 멤버id에서 개별id를 추출하여 참여 횟수 등의 변수 생성
-> trade의 경우 전체 trade 리스트 중 train id가 구매/판매한 데이터만 이용해 변수 생성
-> trade와 party 전체 데이터에서 network를 구해 중심성 변수 생성

- 현재 feature variables 총 536개
  - Modeling에서는 feature간 상관관계 등을 고려하여 선택 사용

- Key from EDA
  1. Retained 유저는 두드러진 특징을 나타낸다
  2. Week 유저 또한 두드러진 특징을 나타낸다
  3. Month와 2Month 유저는 구분하기 쉽지 않다

### 2.2 주제별 Feature Engineering

- [activity data](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/1_1_FE_activity.ipynb) : 유저의 인게임 활동 정보를 일주일 정보로 집계
- [payment data](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/1_2_FE_payment.ipynb) : 유저별 주간 결제 금액을 집계
- [party data](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/1_3_FE_party.ipynb) : 유저간 파티 구성 관계를 집계한 정보
- [party network](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/1_3_FE_party_network.ipynb) : party 구성원들의 연관성등을 파악하기 위해 새로 생성한 변수
- [guild data](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/1_4_FE_guild.ipynb) : 문파별 문파원 목록을 집계
- [trade data](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/1_5_FE_trade.ipynb) : 유저간 1:1 거래 내역을 집계

## 3. Modeling

### 3.1 Modeling Overview : Ensemble Technique

- [Stacking Model](https://github.com/benestump/Project_Classification_Big_Contest/blob/master/stacking%20model.ipynb) : 여러 개의 예측 모델에서 나온 예측치 값을 새로운 변수로 하여 최종적으로 결과를 예측하는 모델을 만드는 앙상블 기법
-  장점
• 단일모델이아닌다양한모델을활용하여하나의모델을구성
• 1단계모델들의에측성능이좋지않은부분에집중하여좋은성능을발휘
• 1단계 모델의 결과가 현저히 다를수록 효율적임
• 각모델별로다른독립변수를사용가능함
- 단점
• 작업량 및 연산량이 많음
• 예측 성능의 향상이 많지않아 효율성이 떨어짐
• Train data에 과적합될 가능성이 큼
- 선정 이유
• 각클래스별로분류성능이좋은모델을활용하기위해
• month, 2month에 대해서만 분류 성능이 저하되는 현상을 meta model에서 보완하기 위해

- 최종 모델 성능
  - F1 score : 0.73

## 4. Conclusion

### 4.1 시사점 및 개선방안
1. 예측시점별로예측성능에차이발생
• 1주 이내 이탈자와 잔류자는 상대적으로 쉽게 분류된다
– week (0.87), retained(0.81), 2month(0.64), month(0.59) • 4주 이내 이탈자와 8주 이내 이탈자가 유사하게 나타난다.
• Month의 recall과 2Month의 precision이 떨어지는 경향은 Month가 2Month로 분류되고 있음을 시사한다.
### 5.2 분석 한계점
- 임의로 나눈 month와 2month를 구분하기 위한 변수를 찾아내는 것이 쉽지 않다
- Stacking을 사용하면 각 모델별로 하이퍼 파라미터를 조절하고 모델을 학습시키는 과정이 복잡하고 물리적 한계가 있다
