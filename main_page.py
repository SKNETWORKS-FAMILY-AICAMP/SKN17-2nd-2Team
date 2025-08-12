import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import joblib
import time
from sklearn.preprocessing import Normalizer, OneHotEncoder

# 한글 폰트 사용을 위한 설정
import matplotlib.font_manager as fm
import matplotlib

font_path = 'C:/Windows/Fonts/gulim.ttc'
font = fm.FontProperties(fname=font_path).get_name()
matplotlib.rc('font', family=font)

# --- 모델 로딩 ---
# 모델은 한 번만 로딩하여 캐시에 저장해두고 사용합니다.
@st.cache_resource
def load_churn_model():
    """'final_model.joblib' 파일을 로드하는 함수"""
    try:
        model = joblib.load("./final_model.joblib")
        return model
    except FileNotFoundError:
        st.error("모델 파일을 찾을 수 없습니다. './final_model.joblib' 경로를 확인해주세요.")
        return None

model = load_churn_model()

# --- 입력값 -> 숫자 변환을 위한 맵(Map) 정의 ---
# UI에서 받은 텍스트 입력을 모델이 학습한 숫자 형태로 변환하기 위한 규칙입니다.

# 지역(living_area) -> 그룹(living_area_grouped)
area_to_group_map = {
    "강남구": 1, "서초구": 1, "송파구": 1, "종로구": 1, "중구": 1, "영등포구": 1, "용산구": 1,
    "강동구": 2, "마포구": 2, "서대문구": 2, "성동구": 2, "광진구": 2, "동작구": 2, "양천구": 2,
    "강북구": 3, "관악구": 3, "구로구": 3, "금천구": 3, "노원구": 3, "도봉구": 3, "동대문구": 3,
    "성북구": 3, "은평구": 3, "중랑구": 3
}

# 직업(job)
job_map = {
    "농업, 어업, 임업": 1, "자영업": 2, "판매/서비스직": 3, "기능/숙련공": 4,
    "일반작업직": 5, "사무/기술직": 6, "경영관리직": 7, "전문/자유직": 8,
    "가정주부": 9, "학생": 10, "구직중": 11, "은퇴자": 12, "기타": 13
}

# 학력(education)
education_map = {"초등학교 졸업 이하": 1, "중학교 졸업": 2, "고등학교 졸업": 3, "대학교 졸업": 4}

# 소득(income)
income_map = {
    "100만원 미만": 1, "100∼150만원 미만": 2, "150∼200만원 미만": 3,
    "200∼250만원 미만": 4, "250∼300만원 미만": 5, "300∼350만원 미만": 6,
    "350∼400만원 미만": 7, "400∼450만원 미만": 8, "450∼500만원 미만": 9,
    "500∼550만원 미만": 10, "550∼600만원 미만": 11, "600~650만원 미만": 12,
    "650~700만원 미만": 13, "700만원 이상": 14
}

# 너비 넓게 설정
st.set_page_config(layout="wide")


# 제목
st.title("🏢서울시민 도서관 이용자 이탈 예측📚")

# =================================================================================

tab1, tab2, tab3 = st.tabs(["1️⃣ 개요", "2️⃣ 이탈 예측", "3️⃣ 결과 분석"])

# 페이지 개요 탭
with tab1:
    st.subheader("")

    c1, c2 = st.columns(2)

    # 페이지 설명 
    with c1:    
        st.info("""
### ✔️ 페이지 소개

🗨️ 이용자의 정보, 이용 패턴 등을 이용하여 **이탈 여부 파악**이 가능합니다.

🗨️ 실제 도서관 이용 패턴을 반영하였기 때문에 예측 결과의 **신뢰도**를 보장합니다.

🗨️ 도서관 서비스 개선 및 이용자 유지 전략 수립에 활용 가능한 **인사이트**를 드립니다.
""")

    # 페이지 사용 방법
    with c2:    
        st.info("""
### 📍 사용 방법


1. 예측하고자 하는 **사용자 정보** 입력

2. **이탈 예측** 결과 확인

3. 이탈 예측에 대한 **분석과 인사이트** 수집
""")
        
    
        


# =================================================================================

# 사용자 특성 탭
with tab2:

    # 사용자 개인 특성 입력
    st.header("💁사용자 개인 특성")
    st.caption("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # gender
        gender = st.radio("성별을 선택해주세요", ["남자", "여자"], horizontal=True)

    with col2:
        # living_area
        living_area = st.selectbox("사는 지역을 선택해주세요", list(area_to_group_map.keys()))
    
    with col3:
        # age
        age = st.slider("나이를 입력해주세요", 10, 100)
    
    with col4:
        # job
        job = st.selectbox("직업을 선택해주세요", list(job_map.keys()))
    
    st.header("")

    col11, col12, col13, col14 = st.columns(4)

    with col11:
        education = st.radio("학력을 선택해주세요", list(education_map.keys()))    

    with col12:
        # income
        income = st.selectbox("소득을 선택해주세요", list(income_map.keys()))
    
    with col13:
        pass
    st.divider()
    st.caption("")

    # 사용자 도서관 이용 특성 이용
    st.header("🪪사용자 도서관 이용 특성")
    st.caption("")

    col21, col22, col23 = st.columns(3)

    with col21:
        # experience
        experience = st.radio("도서관 이용 경험", ["1년 안에 이용함", "과거에만 이용함"])
    
    with col22:
        # distance
        distance = st.radio("현재 거주지 근처에 있는 도서관을 이용할 때, 도보로 이용하나요?", ["도보로 이용한다", "도보로 이용하지 않는다"], horizontal=True)

    st.divider()

    # --- 4. 예측 버튼 및 결과 출력 ---
    if st.button("예측하기", use_container_width=True, type="primary"):
        if model:
            # --- 입력값을 모델이 학습한 데이터에 사용한 값과 동일하게 변환 ---
            gender_val = 1 if gender == "남자" else 0
            age_val = age
            education_val = education_map[education]
            income_val = income_map[income]
            job_val = job_map[job]
            living_area_grouped_val = area_to_group_map[living_area]
            experience_val = 1 if experience == "1년 안에 이용함" else 0
            distance_val = 1 if distance == "도보로 이용한다" else 0

            # --- 모델 입력을 위한 데이터프레임 생성 ---
            # (중요) 컬럼 순서와 이름은 모델 학습 때 사용한 것과 정확히 일치해야 합니다.
            feature_names = ['gender', 'age', 'education', 'income', 'job', 'living_area_grouped', 'experience', 'distance']
            
            input_data = pd.DataFrame([[
                gender_val,
                age_val,
                education_val,
                income_val,
                job_val,
                living_area_grouped_val,
                experience_val,
                distance_val
            ]], columns=feature_names)

            # --- 예측 실행 및 결과 표시 ---
            with st.spinner("AI 모델이 이탈 확률을 계산하고 있습니다..."):
                time.sleep(1) # 로딩하는 것처럼 보이게 잠시 대기
                
                try:
                    # 1. decision_function으로 확신 점수(score)를 얻음
                    decision_score = model.decision_function(input_data)

                    # 2. 시그모이드 함수를 적용하여 점수를 0~1 사이의 확률 같은 값으로 변환
                    def sigmoid(x):
                        return 1 / (1 + np.exp(-x))
                    
                    churn_probability = sigmoid(decision_score[0])

                except AttributeError:
                    st.error("모델에 decision_function이 없습니다. 모델이 잘못 저장되었을 수 있습니다.")
                    st.stop() # 오류 발생 시 중단


            st.header("이탈 가능성 예측 결과")
            
            # 확률에 따라 다른 메시지 표시
            if churn_probability > 0.5: # 임계값은 필요에 따라 조정 가능
                st.error(f"이탈 확률이 **{churn_probability:.2%}** 로, **'이탈 예상'** 회원입니다.")
            else:
                st.success(f"이탈 확률이 **{churn_probability:.2%}** 로, **'잔류 예상'** 회원입니다.")
            
            # 확률을 프로그레스 바로 시각화
            st.progress(churn_probability, text=f"이탈 확률: {churn_probability:.0%}")

        else:
            st.error("모델을 로드하지 못했습니다. 파일 경로를 다시 확인해주세요.")

    st.caption("")

# =================================================================================

# 결과 분석 탭
with tab3:
    st.header("🫧사용자 이탈 예측 분석")

    co1, co2 = st.columns(2)

    # 예측 결과 시각화 그래프
    with co1:
        x = np.linspace(0, 10, 100)
        y = np.sin(x)

        fig, ax = plt.subplots(figsize=(2, 2))  # fig: 전체 그림, ax: 그래프 영역
        ax.plot(x, y, label="Sine Wave", color="blue")
        ax.set_title("선 그래프 예시")
        ax.set_xlabel("X축")
        ax.set_ylabel("Y축")
        ax.legend()
        st.pyplot(fig)
    
    with co2:
        categories = ["A", "B", "C", "D"]
        values = [10, 20, 15, 8]

        fig, ax = plt.subplots(figsize=(2, 2))
        ax.bar(categories, values, color="orange")
        ax.set_title("막대 그래프 예시")
        ax.set_ylabel("값")
        st.pyplot(fig)