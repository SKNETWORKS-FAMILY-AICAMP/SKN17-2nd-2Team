# 🔖2nd_PROJECT_2TEAM 2팀

<br>

# **프로젝트 명 : 도서관 이용자 이탈률 예측 모델 개발** 📚

<br>

## 💡**팀명 : Libraritics**

<br>

✔️ Library + Analytics의 합성어로, **도서관**과 **이탈률 분석**이라는 프로젝트의 핵심 주제를 직관적으로 표현하였습니다.

✔️ 데이터 분석을 통해 도서관 이용자 이탈 현상을 이해하고, 이를 기반으로 효과적인 운영 전략을 제시하겠다는 **프로젝트의 목표**를 담고 있습니다.


<br>

---

## 🌟 **팀원 소개**  

| 💻👩 양송이 | 💻👩 박지수 | 💻🧑 조세희| 💻🧑 김세한 | 💻🧑 양정민 |
|---|---|---|---|---|
| <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-2nd-2Team/blob/main/img/common2.png" width="150" height="170"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-2nd-2Team/blob/main/img/common3.png" width="150" height="170"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-2nd-2Team/blob/main/img/common.png" width="150" height="170"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-2nd-2Team/blob/main/img/common5.png" width="150" height="170"/> | <img src="https://github.com/SKNETWORKS-FAMILY-AICAMP/SKN17-2nd-2Team/blob/main/img/common4.jpg" width="150" height="170"/> | 
|[@songeeeey](https://github.com/songeeeey)| [@0lipa](https://github.com/0lipa) | [@SEHEE-8546](https://github.com/SEHEE-8546) | [@kimsehan11](https://github.com/kimsehan11) | [Yangmin3](https://github.com/Yangmin3)|

<br>

---

## 1. 프로젝트 개요

### 1.1 프로젝트 주제 선정 배경

<br>

<img src="./img/리드미배경2.png" width="600"/>

<br>

>_디스커버리 뉴스: [문체부, '2025년 전국 공공도서관 통계조사' 결과 발표](https://www.discoverynews.kr/news/articleView.html?idxno=1064416&utm_source=chatgpt.com)_

<img src="./img/리드미배경3.png" width="600"/>

>_농민 뉴스: [북캉스’ 최적지 공공도서관 …지난해 이용자 2억명 ‘훌쩍’](https://www.nongmin.com/article/20240806500181?utm_source=chatgpt.com)_

<br>
<br>

<img src="./img/도서관이용자수그래프.png"/>

>_출처: [통계청 지표누리](https://www.index.go.kr/unity/potal/main/EachDtlPageDetail.do?idx_cd=1639)_

<br>

최근 도서관 이용자 수 통계에 따르면, **코로나19 팬데믹 시기**(2019→2020)에 25만 명 이상이던 **이용자 수가 급격히 감소**했습니다. 이후 점진적으로 회복되고 있으나, 여전히 코로나 이전 수준에는 미치지 못하고 있습니다.

그 뿐만 아니라 다양한 온라인 콘텐츠의 확산과 전자책의 보급 그리고 코로나19 등과 같이 사회적 변화들로 인해 도서관 이용자 수는 하락하는 추세를 보이고 있습니다.


<br>

---

### 1.2 프로젝트 주제 필요성

<br>


이러한 상황에서 도서관 운영진들은 크게 **두 가지 과제**에 직면해 있습니다.

① 새로운 이용자를 **유입**하는 것

② 기존 이용자의 **이탈을 방지**하는 것

저희는 특히 **두 번째 과제**, **이탈 방지**에 주목했습니다. 신규 이용자 유입보다 기존 이용자를 유지하는 것이 장기적으로 더 안정적이고 비용 효율적이기 때문입니다.

> 조사결과, 도서관 밖 유동인구 수는 증가하고 있지만, 도서관으로 유입되는 방문자 수는 감소하는 것으로 나타났다.
> 
> 
> *(출처: 박성재, 2023.01.09,공공도서관 이용자는 도서관을 어떻게 이용하는가?, 한국문헌정보학회지)*

<br>

단순한 이용 통계만으로는 이탈 원인을 파악하기 어려우므로, **인구통계학적 특성**(성별, 연령, 거주지, 소득, 학력, 거리)과 **이용 경험 데이터**를 결합해 분석하는 예측 모델이 필요합니다.


<br>


---

### 1.3 프로젝트 목적

<br>

본 프로젝트의 목적은 다음과 같습니다:
- **주요 영향 요인 분석**: 이탈에 영향을 미치는 핵심 변수들을 식별하고 그 중요도를 정량화
- **이탈 예측 모델 개발**: 다양한 머신러닝 분류 모델을 활용하여 도서관 이용자의 이탈 가능성을 예측하는 최적 모델 도출
- **도서관 운영 개선**: 예측 결과를 바탕으로 도서관 서비스 개선 및 이용자 유지 전략 수립에 활용할 수 있는 인사이트 제공

<br>

본 프로젝트는 다음과 같은 가설을 전제로 시작되었습니다 :

<mark>**"도서관 이용자의 이탈은 인구통계학적 특성(성별, 나이, 거주지역, 소득, 학력, 도서관과의 거리)과 도서관 이용 경험 등 다양한 요인들의 복합적 작용으로 예측이 가능하다."** </mark>

이 가설을 바탕으로, 실제 도서관 이용자 데이터를 수집하고 정제한 뒤 머신러닝 기반 분류 모델을 통해 이탈에 영향을 주는 주요 요인을 분석하고자 합니다.

<br>

---



## 2.  **기술 스택** 🛠️

| **분류**         | **기술/도구**                                                                            |
|------------------|------------------------------------------------------------------------------------------|
| **언어**         | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python)     |
| **라이브러리**   | ![NumPy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy) ![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas) ![Matplotlib](https://img.shields.io/badge/Matplotlib-ffffff?style=for-the-badge&logo=Matplotlib) <br> ![Seaborn](https://img.shields.io/badge/seaborn-0C5A5A?style=for-the-badge&logo=Seaborn) ![scikitlearn](https://img.shields.io/badge/scikitlearn-green?style=for-the-badge&logo=scikitlearn) ![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm) ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost)|
| **협업 툴**      | ![GitHub](https://img.shields.io/badge/github-121011?style=for-the-badge&logo=github) ![Git](https://img.shields.io/badge/git-F05033?style=for-the-badge&logo=git)|


---
## 3. WBS 

<img src="./img/WBS.png" width=700/>




---

<br>

## 4. 데이터 선택 및 특징

### 4.1 데이터 선택

- 도서관 이용에 대한 설문지를 바탕으로 이용자의 인구통계학적 정보, 이용 패턴 등을 포함하고 있는 데이터 셋 활용

<br>

**서울특별시 서울시민의 도서관 이용실태 설문조사 데이터**

<br>

| 항목 | 설명 |
| --- | --- |
| **데이터 명** | 서울특별시 서울시민의 도서관 이용실태|
| **데이터 출처** | 공공데이터포털 |
| **데이터 제공 기관** | 서울특별시 데이터전략과 |
| **데이터 설명** | 서울특별시민 도서관 이용실태 조사에 대한 데이터로 서울시민 대상(3,000명) 도서관 이용, 인식 조사 설문자료와 설문결과 데이터 |
| **주요 컬럼명** | 거주지역, 성별, 출생년도, 최종학력, 직업, 월평균 소득, 1년간 도서관 이용 경험 여부, 현재 도서관 이용 의향, 미래에 도서관 이용 의향 등 |

<br>

### 4.2 데이터 특징

**① 사용자 개인 특성 반영**
- 성별, 연령, 학력, 소득, 거주지역, 직업 등 인구통계학적 변수와 도서관 이용 여부 등 행동 패턴 변수를 모두 포함

**② 이탈 여부 파생 변수 생성**
- 설문 문항(미래 이용 의향 없음 & 1년 안에 도서관을 간 적이 없는 사람)을 바탕으로 ‘이탈 여부(Churn)’ 컬럼 직접 생성 

**③ 분류 모델 학습에 적합**
- 수치형 및 범주형 변수가 혼합된 형태로, 다양한 분류 알고리즘 실험 가능

**④ 현실적 적용 가능성**
- 실제 도서관 이용 패턴을 반영한 설문 기반 데이터로, 운영 개선에 바로 적용 가능

<br>

<br>

---

## 5. 데이터 전처리 및 EDA (탐색적 데이터 분석)

### 5.1 데이터 전처리

#### 5.1.1 결측치 처리
- `gender`, `age`, `education`, `income`, `experience`, `job`, `living_area`, `distance`, `future_use ` 전부 결측치 없음.

  <img src="./img/결측치없음.png" width=300/> 


#### 5.1.2 이상치 처리
- `income` : 99 → 모름/무응답 제거

  <img src="./img/income.png" width=800/>

#### 5.1.3 지역 데이터 그룹화
서울시 25개 구를 경제·상권 특성으로 3개 군집화:
- **Group_A**: 강남, 서초, 송파, 종로, 중구, 영등포, 용산
- **Group_B**: 강동, 마포, 서대문, 성동, 광진, 동작, 양천
- **Group_C**: 그 외 구

  <img src="./img/living_area_grouped.png" width=400/>

#### 5.1.4 범주형 변수 재코딩

(1) 범주 병합

- education (학력) 

  - 고등학교 졸업 + 대학교 중퇴 -> "고등학교 졸업"으로 변환

  <img src="./img/학력전처리.png"/> --> 사진 수정!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

(2) 0과 1 등으로 이진화

- experience (1년간 방문 경험) 

  -  "0(없다), 1(있다)"로 변환

    <img src="./img/1년이용전처리.png"/> 
    -> 사진 ???????????????? 


### 5.2. 최종 데이터 구조

#### **주요 변수**
- `gender` : 성별
- `age` : 나이
- `education` : 학력 수준
- `income` : 소득 수준
- `experience` : 1년 이내 도서관 이용 여부
- `job` : 직업
- `living_area_grouped` : 거주 지역 (서울시 구별 코드)
- `distance` : 도서관과의 거리



#### **분석 타겟 컬럼**
- `churn` : 도서관 이용자 이탈 여부 (0: 비이탈, 1: 이탈)



### 5.3 상관관계 분석

<img src="./img/상관관계분석.png" width=800/>




### 5.4 이탈률 분포

이미지 추가 !!!!!!!!!!!!!!!!!!!





<!-- ### 3.5 Churn 연관 1차 관찰
??????????????????
- `income` 증가 구간에서 churn 비율 점진적 감소
- `distance` 증가 시 일부 구간에서 churn 소폭 상승 경향
- `experience=1` (실사용 경험 있음) 그룹이 낮은 churn 비율
- `education` 상위 단계(고/대졸 이상)에서 retention 경향 -->

<br>

---

<br>


## 6. 머신러닝 파이프라인 🔧

### 6.1 사용 피처
- **Target**: `churn`
- **Features**: `gender`, `age`, `education`, `income`, `experience`, `job`, `living_area_grouped`, `distance`

  <br>

### 6.2 전처리 전략
| **변수 유형** | **변수명** | **전처리 방법** |
|---|---|---|
| 수치형 | `age` | StandardScaler |
| 명목형(순서X) | `living_area_grouped`, `job`, `experience` | OneHotEncoder |
| 순서형 | `gender`, `income`, `distance`, `education` | Label Encoding |

- **파이프라인 구성**: ColumnTransformer + Pipeline

<br>

### 6.3 클래스 불균형 처리 ⚖️
- **원본 분포**: Churn 0 ≈ 89.4%, 1 ≈ 10.6%
- **클래스 불균형 사진**
- **적용 기법**: **SMOTE**

<br>

### 6.4 모델 구성 및 성능 향상 전략 

#### 6.4.1 Base Model
- **기본 features 구성**: 7개의 기본 features만 사용
  - `gender`, `living_area`, `job`, `age`, `education`, `income`, `child`
- **여기에 base model 사진**

<br>

#### 6.4.2 Updated Features
- **변수 제거**: 상관계수가 매우 낮아 상관관계가 없다고 판단된 변수 `child` 제거
- **새로운 파생변수 추가**: 
  - `distance`: 새로운 파생변수 distance 추가 
  - `living_area_grouped`: 거주지/소득과 도서관 이용률간의 상관관계가 존재한다는 [기사](https://www.sisain.co.kr/news/articleView.html?idxno=47046)를 바탕으로 지역 `living_area`를 경제·상권 특성으로 그룹화한 `living_area_grouped` 변수 추가
 
<br>

#### 6.4.3 Best Model: **Stacking Ensemble**

**Base Estimators:**
1. **RandomForestClassifier** 
2. **LightGBM** 
3. **XGBoost** 

**Final Estimator 후보:**
- logistic regression (기본)
- *RidgeClassifier
- SVC
- ElasticNet
- KNeighborsClassifier

**제일 잘 나온 성능 사진**

<br>

---

## 7. 모델 성능 결과 

### 7.1 테스트 성능 (Stacking + SMOTE)

** 평가 지표**: precision / recall / F1 / accuracy

** 하이퍼파라미터 조정**: hyperopt를 통해 진행

<br>

#### 7.1.1 성능 비교
<img width="800" height="250" alt="Image" src="./readme_data/before_webp" />

**하이퍼파라미터 적용 전**

<img width="800" height="250" alt="Image" src="./readme_data/after_webp" />

**하이퍼파라미터 적용 후**

<br>

---

## 8. 인사이트 및 정책 제언 

### 8.1 개별 이용자 맞춤형 이탈 방지 전략

본 예측 모델을 활용하여 **최소한의 인구통계학적 특성만으로** 개별 이용자의 이탈 확률을 사전에 예측 가능
- **선별적 고객 관리**: 이탈 확률이 높은 고객에게만 집중적으로 개인 맞춤형 서비스(책 추천, 프로그램 소개, 개별 상담 등)를 제공
- **인력 및 예산 최적화**: 모든 이용자가 아닌 고위험군에게만 자원을 집중 투입하여 운영 효율성 극대화

<br>

### 8.2 전체 이용자 대상 정책 수립 지원

예측 모델 분석 결과를 통해 이탈 위험이 높은 **인구통계학적 집단**을 식별하고, 해당 집단에 특화된 정책을 수립

- **고위험 집단 식별**: 이탈 확률이 높게 나타나는 특정 연령대, 직업군, 학력 수준, 거주 지역 등을 데이터 기반으로 파악
- **집단별 맞춤 프로그램**: 식별된 고위험 집단의 특성에 맞는 전용 프로그램 개발 및 운영
- **예방적 정책 수립**: 이탈 요인 분석을 바탕으로 근본적인 서비스 개선 방향 설정
- **예산 효율성**: 전체 이용자 대상이 아닌 특정 집단에 집중하여 정책 효과 극대화 및 예산 절약

<br>

### 8.3 운영 전략 제언

1. **단계별 적용**: 먼저 고위험 개별 고객 관리를 통해 즉시 효과를 확인한 후, 장기적으로 정책 수준의 개선 방안 적용
2. **지속적 모델 업데이트**: 새로운 이용자 데이터를 주기적으로 반영하여 예측 정확도 지속 개선
3. **성과 측정**: 이탈 방지 프로그램 적용 전후의 이탈률 변화를 정량적으로 측정하여 효과 검증

<br>

---

## 9. 한계점 

### 9.1 불균형 데이터 처리
- SMOTE를 통한 균형 조정에도 불구하고 여전히 존재하는 클래스 불균형 문제 → **낮은 precision**

<br>

### 9.2 개인특성 이외의 변수
- 현재 사용된 변수 외에 추가적인 파생 변수 생성 가능
  - `total reading time` : 총 독서 시간
  - `books per year` : 연간 독서량
- **제거 사유**: 총 독서시간과 연간 독서량은 처음 보는 사람의 개인 특성으로 보기에는 어려움이 있어서 제거 

<br>

 **팀원 한 줄 회고**
<table>
  <tr>
    <th style="width:80px;">이름</th>
    <th>회고 내용</th>
  </tr>
