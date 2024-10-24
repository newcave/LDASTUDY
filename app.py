import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# 스트림릿 페이지 설정
st.title('LDA 교육용 자료 - 연구주제 Topic Modeling')

# 사이드바에서 입력값 받기
st.sidebar.header('LDA 설정')
n_components = st.sidebar.slider('주제 수 (n_components)', min_value=2, max_value=10, value=4)
top_n_words = st.sidebar.slider('상위 단어 수 (Top N Words)', min_value=2, max_value=10, value=4)  # 주제별 상위 단어 수 설정
doc_topic_prior = st.sidebar.selectbox('Alpha (문서-주제 분포)', [0.01, 0.05, 0.1, None])
topic_word_prior = st.sidebar.selectbox('Beta (주제-단어 분포)', [0.01, 0.02, 0.05, None])
random_state = st.sidebar.number_input('Random State', min_value=0, value=42, step=1)

# 예제 데이터를 사용할지 여부 선택
use_example_data = st.sidebar.checkbox('Input Data Reset', value=True)

# 예제 데이터
example_documents = [
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

    #  C 연구 
    'groundwater contamination hydrogeology aquifer recharge',
    'groundwater quality monitoring well drilling subsurface flow',
    'hydraulic conductivity aquifer tests water table level monitoring',
    'groundwater pollution nitrate contamination agriculture runoff',
    'groundwater management policies sustainable water use',
    'aquifer storage recovery drought resilience water conservation',

    #  D 연구 
    'membrane filtration water treatment reverse osmosis microfiltration',
    'ultrafiltration membrane systems desalination process water reuse',
    'membrane bioreactor technology wastewater treatment filtration',
    'reverse osmosis desalination brackish water treatment permeate flux',
    'nanofiltration technologies industrial water treatment fouling control',
    'membrane performance optimization water purification energy efficiency',
]

# 메인창에 데이터 입력 또는 예제 데이터 사용
if use_example_data:
    st.write("Example Data : ")
    # 예제 데이터를 표시하고 수정할 수 있는 텍스트 박스를 제공
    modified_documents = st.text_area("데이터 수정:", "\n".join(example_documents))
    documents = modified_documents.split('\n')  # 사용자가 수정한 데이터로 대체
else:
    st.write("직접 데이터를 입력하세요.")
    user_input = st.text_area("입력 문서 데이터 (한 줄에 하나의 문서)", "water pollution ecosystem")
    documents = user_input.split('\n')  # 입력 데이터를 라인별로 나눕니다.

# LDA 수행 버튼
if st.button('LDA 수행'):
    # 문서 데이터가 있는지 확인
    if len(documents) > 0 and documents[0] != '':
        # 텍스트 데이터를 행렬로 변환
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(documents)

        # LDA 모델 생성
        lda = LatentDirichletAllocation(
            n_components=n_components,  # 주제 수
            doc_topic_prior=doc_topic_prior,  # alpha
            topic_word_prior=topic_word_prior,  # beta
            random_state=random_state  # random_state 값을 사용자가 지정
        )

        # 모델 학습
        lda.fit(X)

        # 주제별 상위 단어 출력
        st.subheader('LDA 결과')
        terms = vectorizer.get_feature_names_out()
        for idx, topic in enumerate(lda.components_):
            st.write(f"Topic {idx + 1}: ", [terms[i] for i in topic.argsort()[-top_n_words:]])

    else:
        st.write("문서 내용 입력")
