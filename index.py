import streamlit as st

import show_parametricmodel , show_parametricmix, show_otherfunc,show_comingsoon, show_repairable, show_alt, show_fitter, show_reliabilitytest
from PIL import Image
image_ufpe = Image.open('./src/logo.png')
image_pip = Image.open('./src/logopip.png')

st.set_page_config(page_title="Reliability",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

st.sidebar.image(image_ufpe, caption='UFPE - CEERMA')

st.sidebar.image(image_pip)#, caption='UFPE - CEERMA')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
#ReportStatus {visibility: hidden;}

</style>

"""
#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

## First option
# PARAMETRIC MODELS
st.sidebar.text("""This page is a master course project.
Streamlit and Reliability libraries
were used to carry out the project.""")

first_menu = st.sidebar.selectbox(
    "Which module do you want to use?",
    ("Select a module","Parametrics and Non-Parametrics Models", "Accelerated life testing", "Repairable Systems","Reliability Testing", "Other Functions")
)

if first_menu == "Parametrics and Non-Parametrics Models":
    add_selectbox = st.sidebar.selectbox(
        "Which submodule do you want to use?",
        ("Select a submodule", "Parametric Model", "Parametric Mix Model", "Non-Parametric model", "Fitter")
    )

    if add_selectbox == "Select a module":
        pass
    if add_selectbox == "Parametric Model":
        show_parametricmodel.show()
    if add_selectbox == "Parametric Mix Model":
        show_parametricmix.show()
    if add_selectbox == "Non-Parametric model":
        show_comingsoon.show()
    if add_selectbox == "Fitter":
        show_fitter.show()


if first_menu == "Accelerated life testing":
    add_selectbox = st.sidebar.selectbox(
        "Which submodule do you want to use?",
        (["Accelerated life testing"])
    )

    if add_selectbox == "Select a module":
        pass
    if add_selectbox == "Accelerated life testing":
        show_alt.show()


if first_menu == "Repairable Systems":
    add_selectbox = st.sidebar.selectbox(
        "Which submodule do you want to use?",
        (["Repairable Systems"])
    )

    if add_selectbox == "Select a module":
        pass
    if add_selectbox == "Repairable Systems":
        show_repairable.show()

        
if first_menu == "Reliability Testing":
    add_selectbox = st.sidebar.selectbox(
        "Which submodule do you want to use?",
        (["Reliability Testing"])
    )

    if add_selectbox == "Select a module":
        pass
    if add_selectbox == "Reliability Testing":
        show_reliabilitytest.show()


if first_menu == "Other Functions":
    add_selectbox = st.sidebar.selectbox(
        "Which submodule do you want to use?",
        ("Select a submodule", "Stress and Strentgh")
    )
    if add_selectbox == "Select a module":
        pass

    if add_selectbox == "Stress and Strentgh":
        show_otherfunc.show()