import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import numpy as np

# 스트림릿 페이지 설정
st.title('LDA 교육용 자료 - 연구주제 Topic Modeling')

# 사이드바에서 A와 B 모델 설정
st.sidebar.header('LDA 설정 (모델 A)')
n_components_a = st.sidebar.slider('A 모델 - 주제 수 (n_components)', min_value=2, max_value=10, value=4)
top_n_words_a = st.sidebar.slider('A 모델 - 상위 단어 수 (Top N Words)', min_value=2, max_value=10, value=3)
doc_topic_prior_a = st.sidebar.selectbox('A 모델 - Alpha (문서-주제 분포)', [0.01, 0.05, 0.1, "Auto"])
topic_word_prior_a = st.sidebar.selectbox('A 모델 - Beta (주제-단어 분포)', [0.01, 0.02, 0.05, "Auto"])
random_state_a = st.sidebar.number_input('A 모델 - Random State', min_value=0, value=42, step=1)

st.sidebar.header('LDA 설정 (모델 B)')
n_components_b = st.sidebar.slider('B 모델 - 주제 수 (n_components)', min_value=2, max_value=10, value=4)
top_n_words_b = st.sidebar.slider('B 모델 - 상위 단어 수 (Top N Words)', min_value=2, max_value=10, value=3)
doc_topic_prior_b = st.sidebar.selectbox('B 모델 - Alpha (문서-주제 분포)', [0.01, 0.05, 0.1, "Auto"])
topic_word_prior_b = st.sidebar.selectbox('B 모델 - Beta (주제-단어 분포)', [0.01, 0.02, 0.05, "Auto"])
random_state_b = st.sidebar.number_input('B 모델 - Random State', min_value=0, value=42, step=1)

# 불용어 입력: 기본값으로 'water', 'flow' 추가
st.sidebar.header('stop word 설정')
default_stopwords = "water, flow, analysis"
stop_words_input = st.sidebar.text_area('추가 입력(콤마로 구분)', default_stopwords)
stop_words = stop_words_input.split(',')

# 예제 데이터를 사용할지 여부 선택
use_example_data = st.sidebar.checkbox('Input Data Reset', value=True)

# 예제 데이터
example_documents = [
    'water quality pollution monitoring river lake ecosystem',
    'wastewater treatment plant management contamination reduction',
    'earthquake resistant buildings structural analysis seismic waves',
    'groundwater contamination hydrogeology aquifer recharge',
]

# 메인창에 데이터 입력 또는 예제 데이터 사용
if use_example_data:
    st.write("Example Data : ")
    modified_documents = st.text_area("데이터 수정:", "\n".join(example_documents))
    documents = modified_documents.split('\n')  # 사용자가 수정한 데이터로 대체
else:
    st.write("직접 데이터를 입력하세요.")
    user_input = st.text_area("입력 문서 데이터 (한 줄에 하나의 문서)", "water pollution ecosystem")
    documents = user_input.split('\n')  # 입력 데이터를 라인별로 나눕니다.

# LDA 수행 함수 정의
def perform_lda(documents, n_components, doc_topic_prior, topic_word_prior, random_state, top_n_words, stop_words):
    # 텍스트 데이터를 행렬로 변환 (불용어 포함)
    vectorizer = CountVectorizer(stop_words=stop_words)
    X = vectorizer.fit_transform(documents)

    # LDA 모델 생성
    lda = LatentDirichletAllocation(
        n_components=n_components,
        doc_topic_prior=doc_topic_prior if doc_topic_prior != "Auto" else None,
        topic_word_prior=topic_word_prior if topic_word_prior != "Auto" else None,
        random_state=random_state
    )

    # 모델 학습
    lda.fit(X)

    # 주제별 상위 단어 출력
    terms = vectorizer.get_feature_names_out()
    topics = [[terms[i] for i in topic.argsort()[-top_n_words:]] for topic in lda.components_]

    # 퍼플렉서티 계산
    perplexity = lda.perplexity(X)

    # 코히어런스 계산
    tokenized_documents = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_documents)
    corpus = [dictionary.doc2bow(text) for text in tokenized_documents]

    lda_topics = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    coherence_model = CoherenceModel(
        topics=topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence='c_v'
    )
    coherence = coherence_model.get_coherence()

    return topics, perplexity, coherence

# LDA 수행 버튼
if st.button('LDA 수행'):
    if len(documents) > 0 and documents[0] != '':
        # 모델 A 수행
        topics_a, perplexity_a, coherence_a = perform_lda(
            documents, n_components_a, doc_topic_prior_a, topic_word_prior_a, random_state_a, top_n_words_a, stop_words
        )

        # 모델 B 수행
        topics_b, perplexity_b, coherence_b = perform_lda(
            documents, n_components_b, doc_topic_prior_b, topic_word_prior_b, random_state_b, top_n_words_b, stop_words
        )

        # 결과 표시
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("모델 A 결과")
            for idx, topic in enumerate(topics_a):
                st.write(f"Topic {idx + 1}: {topic}")
            st.write(f"Perplexity (낮을수록 좋음): {perplexity_a:.4f}")
            st.write(f"Coherence (높을수록 좋음): {coherence_a:.4f}")

        with col2:
            st.subheader("모델 B 결과")
            for idx, topic in enumerate(topics_b):
                st.write(f"Topic {idx + 1}: {topic}")
            st.write(f"Perplexity (낮을수록 좋음): {perplexity_b:.4f}")
            st.write(f"Coherence (높을수록 좋음): {coherence_b:.4f}")

    else:
        st.write("문서 내용을 입력하세요.")
