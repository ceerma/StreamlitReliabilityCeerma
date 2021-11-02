
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from reliability.Reliability_testing import one_sample_proportion, two_proportion_test, sample_size_no_failures


def show():
    #st.set_page_config(page_title="Parametric Model",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    #ReportStatus {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    #href_homepage = f'<a href="https://reliability.ceerma.com/" style="text-decoration: none; color :black;" > <button kind="primary" class="css-qbe2hs edgvbvh1">Go to Homepage</button></a>'
    #st.markdown(href_homepage, unsafe_allow_html=True)

    updatemenus_log = [
        dict(
            buttons=list([
                dict(
                    args=[{'xaxis.type': '-', 'yaxis.type': '-'}],
                    label='Linear',
                    method='relayout'

                ),
                dict(
                    args=[{'xaxis.type': 'log', 'yaxis.type': '-'}],
                    label='Log-x',
                    method='relayout'

                ),
                dict(
                    args=[{'xaxis.type': '-', 'yaxis.type': 'log'}],
                    label='Log-y',
                    method='relayout'

                ),
                dict(
                    args=[{'xaxis.type': 'log', 'yaxis.type': 'log'}],
                    label='Log-xy',
                    method='relayout'

                ),
            ]),
            direction="right",
            type="buttons",
            pad={"r": 10, "t": 10},
            x=0.0,
            xanchor="left",
            y=1.115,
            yanchor="top"                       

        )
    ]






    st.title("Reliability Test")
    st.write("In this module, you can calculates the upper and lower bounds of reliability for a given number of trials and successes")

    submodel = st.selectbox( 'Select a reliability test.', ('One Sample proportion',
    'Two proportion test'))
    #'Sample size required for no failure', 'Sequential sampling chart' ,'Reliability test planner', 'Reliability test duration', 'Chi-Squared test','Kolmogorov-smirnov test')


    if submodel == "One Sample proportion":
        st.write("This function calculates the upper and lower bounds of reliability for a given number of trials and successes. It is most applicable to analysis of test results in which there are only success/failure results and the analyst wants to know the reliability of the batch given those sample results.")
        var1 = st.number_input("Trials: " , min_value= 0, value=30, step=1)
        var2 = st.number_input("Successes: " , min_value= 0, value=29, step=1)
        var3 = st.number_input("Confidence interval : " , min_value= 0.5 , max_value= 0.999, value=0.95, step=0.001)
        result = one_sample_proportion(trials=int(var1),successes=int(var2), CI = float(var3), print_results=False)
    elif submodel =="Two proportion test":
        st.write("This function determines if there is a statistically significant difference in the results from two different tests. ")
        st.write(" Sample 1")
        var1 = st.number_input("Trials (Sample 1):" , min_value= 0, value=500, step=1)
        var2 = st.number_input("Successes (Sample 1): " , min_value= 0, value=490, step=1)
        st.write(" Sample 2")
        var3 = st.number_input("Trials (Sample 2): " , min_value= 0, value=800, step=1)
        var4 = st.number_input("Successes (Sample 3): " , min_value= 0, value=770, step=1)
        st.write(" ")
        var5 = st.number_input("Confidence interval: " , min_value= 0.500 , max_value= 0.999, value=0.95, step=0.001)
        result = two_proportion_test(sample_1_trials=int(var1),sample_1_successes=int(var2),sample_2_trials=int(var3),sample_2_successes=int(var4), CI = float(var5), print_results=False)

    else:
        st.write("Select a reliability test")


    #st.write( sample_size_no_failures(reliability=0.999) )
    st.write(" ")
    if st.button("Calculate"):
        if submodel == "One Sample proportion":
            text_print =  '''
            Results from One Sample Proportion: \n
            For a test with 30 trials of which there were 29 successes and 1 failures, the bounds on reliability are: \n
            Lower {}% confidence bound: {} \n
            Upper {}% confidence bound: {} \n
            '''
            st.write(text_print.format(float(var3*100),result[0],float(var3*100),result[1]))
        elif submodel =="Two proportion test":
            text_print = '''
            Results from two_proportion_test:   \n 
            Sample 1 test results (successes/tests): {}/{}  \n 
            Sample 2 test results (successes/tests): {}/{}  \n 
            The {}% confidence bounds on the difference in these results is: {} to {}  \n 
            Since the confidence bounds contain 0 the result is statistically {}. \n 
            '''
            st.write(text_print.format(var1,var2,var3,var4,float(var5*100),result[0],result[1],result[2]))




if __name__ == "__main__":
    show()