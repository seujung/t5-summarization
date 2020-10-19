import streamlit as st
from infer import Summary_Predict

@st.cache
def load_model():
    cls = Summary_Predict('binary', './vocab/tokenizer_35000.model')
    return cls

cls = load_model()
st.title("Summarization Model Test")
text = st.text_area("Input news :")

st.markdown("## Original News Data")
st.write(text)

if text:
    st.markdown("## Predict Summary")
    with st.spinner('processing..'):
        pred = cls.predict(text)
    st.write(pred)