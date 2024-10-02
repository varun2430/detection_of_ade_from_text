import streamlit as st
from main import main
from drug_drug_interaction import get_potential_ddi


def read_text_file(uploaded_file):
    if uploaded_file is not None:
        return uploaded_file.read().decode('utf-8')
    return ""


st.set_page_config(
    page_title='MediShield',
    page_icon="💊",
    layout="wide", )

st.header('MediShield')
st.markdown('Identify Adverse Drug Effects and Drug-Drug Interactions from Clinical Text 📋🔍')

uploaded_file = st.file_uploader('Upload a text document:', type=['txt'])
text_content = read_text_file(uploaded_file)

text_input = st.text_area('Enter text here or upload a text file:', value=text_content)

button = st.button('Submit', use_container_width=True)

if button:
    text_content = text_input if text_input else read_text_file(uploaded_file)
    res, drugs, relations = main(text_content)

    if res != '':
        st.text('  ')
        st.markdown('#### ADE related sentences:')
        st.text(res)

        st.text('  ')
        st.divider()
        st.markdown('#### Drug-Drug Combinations To Avoid:')
        for drug in drugs:
            res = get_potential_ddi(drug)

            for elm in res:
                if drug == elm[0]:
                    continue

                st.text(f'{drug}    +    {elm[0]}    ->    {elm[1]}')
    else:
        st.text('  ')
        st.text('No ADE-related sentences found.')
