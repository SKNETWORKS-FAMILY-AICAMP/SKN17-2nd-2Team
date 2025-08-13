import streamlit as st
import pandas as pd
import joblib
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import matplotlib

# --- 페이지 설정 (Streamlit 앱의 가장 상단에 위치해야 합니다) ---
st.set_page_config(
    page_title="서울시민 도서관 이용자 이탈 예측",
    page_icon="📚", # 이모지 아이콘 추가
    layout="wide", # 레이아웃을 wide로 설정
    initial_sidebar_state="expanded"
)

# --- 한글 폰트 사용을 위한 설정 ---
font_path = 'C:/Windows/Fonts/gulim.ttc' # 윈도우즈 굴림체 경로
if 'font_loaded' not in st.session_state:
    try:
        font_name = fm.FontProperties(fname=font_path).get_name()
        matplotlib.rc('font', family=font_name)
        matplotlib.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
        st.session_state.font_loaded = True
    except Exception as e:
        st.warning(f"폰트 로딩 오류: {e}. 그래프에 한글이 깨질 수 있습니다.")
        st.session_state.font_loaded = False


# --- 모델 로딩 ---
@st.cache_resource
def load_churn_model():
    """'final_model.joblib' 파일을 로드하는 함수"""
    try:
        model = joblib.load("C:/skn_17/project2/final_model.joblib")
        return model
    except FileNotFoundError:
        st.error("모델 파일을 찾을 수 없습니다. 'C:/skn_17/project2/final_model.joblib' 경로를 확인해주세요.")
        return None

model = load_churn_model()


# --- 입력값 -> 숫자 변환을 위한 맵(Map) 정의 ---
area_to_group_map = {
    "강남구": 1, "서초구": 1, "송파구": 1, "종로구": 1, "중구": 1, "영등포구": 1, "용산구": 1,
    "강동구": 2, "마포구": 2, "서대문구": 2, "성동구": 2, "광진구": 2, "동작구": 2, "양천구": 2,
    "강북구": 3, "관악구": 3, "구로구": 3, "금천구": 3, "노원구": 3, "도봉구": 3, "동대문구": 3,
    "성북구": 3, "은평구": 3, "중랑구": 3
}
job_map = {
    "농업, 어업, 임업": 1, "자영업": 2, "판매/서비스직": 3, "기능/숙련공": 4,
    "일반작업직": 5, "사무/기술직": 6, "경영관리직": 7, "전문/자유직": 8,
    "가정주부": 9, "학생": 10, "구직중": 11, "은퇴자": 12, "기타": 13
}
education_map = {"초등학교 졸업 이하": 1, "중학교 졸업": 2, "고등학교 졸업": 3, "대학교 졸업": 4}
income_map = {
    "100만원 미만": 1, "100∼150만원 미만": 2, "150∼200만원 미만": 3,
    "200∼250만원 미만": 4, "250∼300만원 미만": 5, "300∼350만원 미만": 6,
    "350∼400만원 미만": 7, "400∼450만원 미만": 8, "450∼500만원 미만": 9,
    "500∼550만원 미만": 10, "550∼600만원 미만": 11, "600~650만원 미만": 12,
    "650~700만원 미만": 13, "700만원 이상": 14
}

# --- 피처 중요도 시각화 함수 및 관련 정의 ---
# 모델 학습 시 사용된 피처 리스트 (hptn4.ipynb에서 X 정의 부분과 동일해야 함)
numerical_features = ['age', 'income']
categorical_features = ['living_area_grouped', 'job']
ordinal_features = ['gender', 'education', 'experience', 'distance']

# 피처 한글명 매핑 (시각화용)
feature_korean_names = {
    'gender': '성별', 'age': '나이', 'education': '학력', 'income': '소득',
    'job': '직업', 'living_area_grouped': '거주 지역', 'experience': '이용 경험',
    'distance': '거리'
}

def plot_feature_importances(model, numerical_feats, categorical_feats, ordinal_feats, korean_names_map):
    preprocessor = model.named_steps['preprocessor']
    stacking_classifier = model.named_steps['classifier']

    # ColumnTransformer를 거친 후의 피처 이름 가져오기
    transformed_feature_names = preprocessor.get_feature_names_out()

    # 각 기본 모델의 중요도를 합산할 배열 초기화
    aggregated_importances = np.zeros(len(transformed_feature_names))
    num_estimators_contributed = 0 # Initialize a counter for averaging

    # StackingClassifier의 각 기본 모델에서 중요도 추출 및 합산
    for estimator_name, estimator_model in stacking_classifier.named_estimators_.items():
        if hasattr(estimator_model, 'feature_importances_'):
            importances = estimator_model.feature_importances_

            # 각 모델의 중요도 합이 1이 되도록 정규화하여 모든 모델이 동일한 가중치를 갖도록 합니다.
            total_importance = np.sum(importances)
            if total_importance > 0:
                normalized_importances = importances / total_importance
            else:
                normalized_importances = importances # 모든 중요도가 0인 경우 방지

            aggregated_importances += normalized_importances
            num_estimators_contributed += 1 # Increment counter
        else:
            st.write(f"Estimator {estimator_name} does not have feature_importances_ attribute.")

    # 모든 모델의 중요도를 합산한 후, 평균을 계산합니다.
    if num_estimators_contributed > 0:
        aggregated_importances /= num_estimators_contributed

    # 임시 DataFrame 생성
    temp_df = pd.DataFrame({
        'transformed_feature': transformed_feature_names,
        'importance': aggregated_importances
    })

    # 변환된 피처 이름을 원본 피처 이름으로 매핑하고 중요도 집계
    final_importances = {}
    for _, row in temp_df.iterrows():
        tf_name = row['transformed_feature']
        importance = row['importance']

        original_name = None
        # 'num__', 'cat__', 'ord__' 접두사 제거
        if '__' in tf_name:
            prefix, base_name = tf_name.split('__', 1)
            if prefix in ['num', 'ord']: # 숫자형, 순서형 피처
                original_name = base_name
            elif prefix == 'cat': # 원-핫 인코딩된 범주형 피처
                # 어떤 원본 범주형 피처에서 파생되었는지 찾기
                for cat_feat in categorical_feats:
                    if base_name.startswith(f'{cat_feat}_'):
                        original_name = cat_feat
                        break
        else: # ColumnTransformer에 의해 처리되지 않은 피처 (거의 없을 것)
            original_name = tf_name

        if original_name:
            final_importances[original_name] = final_importances.get(original_name, 0) + importance

    # DataFrame으로 변환 및 중요도 순으로 정렬 (오름차순으로 정렬하여 그래프에서 가장 중요한 것이 위에 오도록)
    feature_importance_df = pd.DataFrame(list(final_importances.items()), columns=['feature', 'importance'])
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=True).reset_index(drop=True) # 오름차순 정렬

    # 한글 이름으로 매핑
    feature_importance_df['feature_korean'] = feature_importance_df['feature'].map(korean_names_map)
    # 매핑되지 않은 피처는 원본 이름 사용
    feature_importance_df['feature_korean'].fillna(feature_importance_df['feature'], inplace=True)

    # 시각화
    fig, ax = plt.subplots(figsize=(7, max(2, len(feature_importance_df) * 0.4))) # 동적으로 크기 조절
    sns.barplot(x='importance', y='feature_korean', data=feature_importance_df, ax=ax, palette='viridis')
    ax.set_title('피처 중요도', fontsize=16)
    ax.set_xlabel('중요도', fontsize=12)
    ax.set_ylabel('피처', fontsize=12)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    return fig


# --- 메인 앱 시작 ---
st.title("🏢서울시민 도서관 이용자 이탈 예측📚")

tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ 개요", "2️⃣ 개인 이탈 예측", "3️⃣ 단체 이탈 예측", "4️⃣ 결과 분석"])

# 페이지 개요 탭
with tab1:
    st.subheader("")

    c1, c2 = st.columns(2)

    # 페이지 설명 
    with c1:    
        st.info("""
### ✔️ 페이지 소개

️🗨️이용자의 정보, 이용 패턴 등을 이용하여 **이탈 여부 파악**이 가능합니다.

️🗨️실제 도서관 이용 패턴을 반영하였기 때문에 예측 결과의 **신뢰도**를 보장합니다.

️🗨️도서관 서비스 개선 및 이용자 유지 전략 수립에 활용 가능한 **인사이트**를 드립니다.
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
        age = st.slider("나이를 입력해주세요", 10, 100, 35)
    
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

    st.header("🪪사용자 도서관 이용 특성")
    st.caption("")

    col21, col22, col23 = st.columns(3)

    with col21:
        # experience
        experience = st.radio("도서관 이용 경험", ["1년 안에 이용함", "과거에만 이용함"])
    
    with col22:
        # distance
        distance = st.radio("거주지와 도서관이 인접한가요?(도보 20분 기준)", ["인접하다", "아니다"], horizontal=True)

    st.divider()

    # --- 4. 예측 버튼 및 결과 출력 ---
    # 사용자 도서관 이용 특성 이용
    st.header("🚀이탈 가능성 예측")
    st.caption("")
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
            distance_val = 1 if distance == "인접하다" else 0 # UI 텍스트와 일치시킴

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
                    st.error("모델에 decision_function이 없습니다. 모델이 잘못 저장되었거나 RidgeClassifier가 아닙니다.")
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
with tab4:
    st.header("🫧사용자 이탈 예측 분석")

    coll1, coll2 = st.columns(2)

    with coll1:

        if model:
            st.subheader("⬇️ 피처 중요도 분석")
            st.write("⚖️ 모델이 이탈 예측에 사용한 각 피처의 상대적 중요도를 보여줍니다. ")
            
            # 피처 중요도 그래프 생성 및 표시
            fig_importance = plot_feature_importances(model, numerical_features, categorical_features, ordinal_features, feature_korean_names)
            st.pyplot(fig_importance)
        else:
            st.warning("모델이 로드되지 않아 피처 중요도를 분석할 수 없습니다.")

# 데이터셋 분석 탭
with tab3:
    st.header("📊 데이터셋 기반 이탈 분석")
    st.write("이탈 여부 예측에 사용된 피처들이 각 그룹별로 어느 정도의 이탈 위험도를 보이는지 시각적으로 분석합니다.")
    
    uploaded_file = st.file_uploader("분석할 CSV 파일을 업로드해주세요.", type="csv")

    if uploaded_file is not None:
        try:
            df_preview = pd.read_csv(uploaded_file)
            st.write("##### ✔️ 업로드된 데이터 미리보기 (상위 5개)")
            st.dataframe(df_preview.head())

            if st.button("데이터셋 분석 시작", use_container_width=True, type="primary"):
                if model:
                    with st.spinner("AI 모델이 데이터셋 전체의 이탈 확률을 계산 중입니다... 잠시만 기다려주세요."):
                        # --- 데이터 전처리 및 예측 ---
                        df_analysis = df_preview.copy()
                        
                        required_cols = ['gender', 'age', 'education', 'income', 'job', 'living_area_grouped', 'experience', 'distance']
                        if not all(col in df_analysis.columns for col in required_cols):
                            missing_cols = [col for col in required_cols if col not in df_analysis.columns]
                            st.error(f"오류: 업로드된 CSV 파일에 필수 컬럼이 없습니다: **{', '.join(missing_cols)}**")
                            st.stop()

                        df_model_input = df_analysis[required_cols]
                        
                        decision_scores = model.decision_function(df_model_input)
                        churn_probabilities = 1 / (1 + np.exp(-decision_scores))
                        df_analysis['churn_probability'] = churn_probabilities
                        
                        bins = [20, 30, 40, 50, 60, 70, 101]
                        labels = ['20대', '30대', '40대', '50대', '60대', '70대 이상']
                        df_analysis['age_group'] = pd.cut(df_analysis['age'], bins=bins, labels=labels, right=False)
                        
                    st.success("데이터셋 분석 완료!")

                    # --- 전체 평균 이탈 확률 ---
                    overall_churn_prob = df_analysis['churn_probability'].mean()
                    st.metric(label="📈 전체 데이터셋 평균 이탈 확률", value=f"{overall_churn_prob:.2%}")
                    st.divider()

                    # --- 피처별 시각화 ---
                    st.subheader("🔎 피처별 이탈 위험도 상세 분석")

                    # 분석할 피처와 역변환 맵 정의
                    features_to_plot = {
                        'age_group': '나이대', 'living_area_grouped': '거주 지역 그룹', 'income': '소득 수준',
                        'job': '직업', 'education': '학력', 'gender': '성별',
                        'experience': '최근 이용 경험', 'distance': '도서관 인접성'
                    }
                    reverse_maps = {
                        'gender': {0: "여자", 1: "남자"}, 'education': {v: k for k, v in education_map.items()},
                        'income': {v: k for k, v in income_map.items()}, 'job': {v: k for k, v in job_map.items()},
                        'experience': {0: "과거에만 이용", 1: "1년 내 이용"}, 'distance': {0: "인접 안 함", 1: "인접함"},
                        'living_area_grouped': {1: "1그룹(핵심)", 2: "2그룹(주거)", 3: "3그룹(외곽)"}
                    }

                    feature_keys = list(features_to_plot.keys())

                    # 피처 리스트를 2개씩 묶어 반복 처리
                    for i in range(0, len(feature_keys), 2):
                        cols = st.columns(2)
                        
                        # 첫 번째 컬럼 그래프
                        feature1_key = feature_keys[i]
                        with cols[0]:
                            churn_by_feature = df_analysis.groupby(feature1_key)['churn_probability'].mean().sort_values(ascending=False)
                            if feature1_key in reverse_maps:
                                churn_by_feature.index = churn_by_feature.index.map(reverse_maps[feature1_key].get)
                            
                            fig, ax = plt.subplots()
                            sns.barplot(x=churn_by_feature.index, y=churn_by_feature.values, ax=ax, palette='viridis', order=churn_by_feature.index)
                            ax.set_title(f'{features_to_plot[feature1_key]}별 평균 이탈 확률', fontsize=14)
                            ax.set_ylabel('평균 이탈 확률', fontsize=10)
                            ax.set_xlabel('')
                            ax.tick_params(axis='x', rotation=45, labelsize=9)
                            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                            plt.tight_layout()
                            st.pyplot(fig)
                            if feature1_key == 'living_area_grouped':
                                st.caption("1(핵심), 2(주거), 3(외곽)")

                        # 두 번째 컬럼 그래프 (피처가 홀수 개일 경우를 대비)
                        if i + 1 < len(feature_keys):
                            feature2_key = feature_keys[i+1]
                            with cols[1]:
                                churn_by_feature = df_analysis.groupby(feature2_key)['churn_probability'].mean().sort_values(ascending=False)
                                if feature2_key in reverse_maps:
                                    churn_by_feature.index = churn_by_feature.index.map(reverse_maps[feature2_key].get)

                                fig, ax = plt.subplots()
                                sns.barplot(x=churn_by_feature.index, y=churn_by_feature.values, ax=ax, palette='viridis', order=churn_by_feature.index)
                                ax.set_title(f'{features_to_plot[feature2_key]}별 평균 이탈 확률', fontsize=14)
                                ax.set_ylabel('평균 이탈 확률', fontsize=10)
                                ax.set_xlabel('')
                                ax.tick_params(axis='x', rotation=45, labelsize=9)
                                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
                                plt.tight_layout()
                                st.pyplot(fig)
                                if feature2_key == 'living_area_grouped':
                                    st.caption("1(핵심), 2(주거), 3(외곽)")

                else:
                    st.error("모델을 로드하지 못했습니다. 관리자에게 문의하세요.")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")