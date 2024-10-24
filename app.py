import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 스트림릿 페이지 설정
st.title('LDA 교육용 자료 - 주제 모델링')

# 사이드바에서 입력값 받기
st.sidebar.header('LDA 설정')
n_components = st.sidebar.slider('주제 수 (n_components)', min_value=2, max_value=10, value=4)
doc_topic_prior = st.sidebar.selectbox('Alpha (문서-주제 분포)', [None, 0.1, 0.5, 1.0])
topic_word_prior = st.sidebar.selectbox('Beta (주제-단어 분포)', [None, 0.1, 0.5, 1.0])

# 예제 데이터를 사용할지 여부 선택
use_example_data = st.sidebar.checkbox('예제 데이터 사용', value=True)

# 메인창에 데이터 입력 또는 예제 데이터 사용
if use_example_data:
    st.write("예제 데이터를 사용합니다.")
    documents = [
        # A 연구
        'water quality pollution monitoring river lake ecosystem',
        'wastewater treatment plant management contamination reduction',
        'water pollution control strategies wetland restoration fish habitat',
        'drinking water purification desalination flood control',
        'water sampling analysis sediment nutrients pollution levels',
        'ecological restoration water health aquatic life river habitat',

        # B 연구
        'earthquake resistant buildings structural analysis seismic waves',
        'soil stabilization foundation engineering landslide prevention',
        'geotechnical engineering rock mechanics tunnel excavation support',
        'structural integrity bridges skyscrapers earthquake design',
        'underground construction deep foundation pile drilling',
        'ground subsidence settlement monitoring slope stability',

        # C 연구
        'groundwater contamination hydrogeology aquifer recharge',
        'groundwater quality monitoring well drilling subsurface flow',
        'hydraulic conductivity aquifer tests water table level monitoring',
        'groundwater pollution nitrate contamination agriculture runoff',
        'groundwater management policies sustainable water use',
        'aquifer storage recovery drought resilience water conservation',

        # D 연구 
        'membrane filtration water treatment reverse osmosis microfiltration',
        'ultrafiltration membrane systems desalination process water reuse',
        'membrane bioreactor technology wastewater treatment filtration',
        'reverse osmosis desalination brackish water treatment permeate flux',
        'nanofiltration technologies industrial water treatment fouling control',
        'membrane performance optimization water purification energy efficiency',
    ]
else:
    st.write("직접 데이터를 입력하세요.")
    user_input = st.text_area("입력 문서 데이터 (한 줄에 하나의 문서)", "water pollution ecosystem")
    documents = user_input.split('\n')  # 입력 데이터를 라인별로 나눕니다.

# 문서 데이터가 있는지 확인
if len(documents) > 0:
    # 텍스트 데이터를 행렬로 변환
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)

    # LDA 모델 생성
    lda = LatentDirichletAllocation(
        n_components=n_components,  # 주제 수
        doc_topic_prior=doc_topic_prior,  # alpha
        topic_word_prior=topic_word_prior,  # beta
        random_state=100
    )

    # 모델 학습
    lda.fit(X)

    # 주제별 상위 단어 출력
    st.subheader('LDA 결과')
    terms = vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda.components_):
        st.write(f"Topic {idx + 1}: ", [terms[i] for i in topic.argsort()[-4:]])

else:
    st.write("문서를 입력하세요.")
