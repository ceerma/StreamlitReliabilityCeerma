
import streamlit as st
import reliability 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Normal_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution
import scipy.stats as ss
from reliability.Utils import round_to_decimals


#st.set_page_config(page_title="Parametric Model",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

def show():
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

    st.title("Other Functions")


    st.write("Stress and Strentgh")

    st.write("Stress")
    col1, col2, col3, col4 = st.columns([3, 1,1,1 ])
    st.write("Strentgh")
    col5, col6, col7, col8 = st.columns([3, 1,1,1 ])

    dist_stress = col1.selectbox( 'Select the distribution for the stress.', ('Normal Distribution','Exponential Distribution',
    'Gumbel Distribution' ,'Weibull Distribution', 'Lognormal Distribution',
    'Gamma Distribution','Loglogistic Distribution'))
    i=1
    if dist_stress == "Exponential Distribution":
        stress_var1 = col2.number_input("Scale parameter (Lambda)" , min_value= float(np.finfo(float).eps), value=10.0 , key = str(i)+"_var1sed") 
        stress_var2 = col3.number_input("Displacement parameter (Gamma)" , key = str(i)+"_var2sed") 
        stress_var3 = col4.number_input("Not Parameter" , key = str(i)+"_var3sed" , min_value= float(0) , max_value= float(0) ,step=float(0) )  
        stress_fun  = Exponential_Distribution(stress_var1,stress_var2)
    elif dist_stress == "Normal Distribution":
        stress_var1 = col2.number_input("Location parameter (Mu)", key= str(i) + "_Ad" ) 
        stress_var2 = col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0 , key= str(i) + "_Ads")  
        stress_var3 = col4.number_input("Not Parameter" , key = str(i)+"_var3nom" , min_value= float(0) , max_value= float(0) ,step=float(0) ) 
        stress_fun = Normal_Distribution(stress_var1,stress_var2)
    elif dist_stress=="Beta Distribution": 
        stress_var1 = col2.number_input("Shape parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1bet") 
        stress_var2 = col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2bet")
        stress_var3 = col4.number_input("Not Parameter" , key = str(i)+"_var3bet" , min_value= float(0) , max_value= float(0) ,step=float(0) )  
        stress_fun = Beta_Distribution(stress_var1,stress_var2)
    elif dist_stress =="Gumbel Distribution":
        stress_var1 = col2.number_input("Location parameter (Mu)" , key = str(i)+"_var1")
        stress_var2 = col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") 
        stress_var3 = col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0)) 
        stress_fun = Gumbel_Distribution(stress_var1,stress_var2)
    elif dist_stress =="Weibull Distribution":
        stress_var1 = col2.number_input("Scale parameter (Alpha)" , min_value=  float(np.finfo(float).eps), value=2.0, key = str(i)+"_var1" )
        stress_var2 = col3.number_input("Shape parameter (Beta)" , min_value=  float(np.finfo(float).eps), value=3.0, key = str(i)+"_var2") 
        stress_var3 = col4.number_input("Location parameter (Gamma)", value=1.0, key = str(i)+"_var3" ) 
        stress_fun = Weibull_Distribution(stress_var1,stress_var2,stress_var3)
    elif dist_stress =="Lognormal Distribution":
        stress_var1 = col2.number_input("Location parameter (Mu)" , key = str(i)+"_var1") 
        stress_var2 = col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2" ) 
        stress_var3 = col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) 
        stress_fun = Lognormal_Distribution(stress_var1,stress_var2,stress_var3)
    elif dist_stress  =="Gamma Distribution":
        stress_var1 = col2.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") 
        stress_var2 = col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") 
        stress_var3 = col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) 
        stress_fun = Gamma_Distribution(stress_var1,stress_var2,stress_var3)
    elif dist_stress =="Loglogistic Distribution":
        stress_var1 = col2.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") 
        stress_var2 = col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") 
        stress_var3 = col4.number_input("Location parameter (Gamma)" , key = str(i)+"_var3") 
        stress_fun = Loglogistic_Distribution(stress_var1,stress_var2,stress_var3)


    dist_strentgh = col5.selectbox( 'Select the distribution for the strentgh.', ('Normal Distribution', 'Exponential Distribution'
        , 'Gumbel Distribution' ,'Weibull Distribution', 'Lognormal Distribution',
    'Gamma Distribution','Loglogistic Distribution'))
    i=2
    if dist_strentgh == "Exponential Distribution":
        strentgh_var1 = col6.number_input("Scale parameter (Lambda)" , min_value= float(np.finfo(float).eps), value=10.0 , key = str(i)+"_var1sed") 
        strentgh_var2 = col7.number_input("Displacement parameter (Gamma)" , key = str(i)+"_var2sed") 
        strentgh_var3 = col8.number_input("Not Parameter" , key = str(i)+"_var3sed" , min_value= float(0) , max_value= float(0) ,step=float(0) )  
        strentgh_fun  = Exponential_Distribution(strentgh_var1,strentgh_var2)
    elif dist_strentgh == "Normal Distribution":
        strentgh_var1 = col6.number_input("Location parameter (Mu)",  value=1.0, key= str(i) + "_Adaaa" ) 
        strentgh_var2 = col7.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0 , key= str(i) + "_Adew") 
        strentgh_var3 = col8.number_input("Not Parameter" , key = str(i)+"_var3nom" , min_value= float(0) , max_value= float(0) ,step=float(0) ) 
        strentgh_fun = Normal_Distribution(strentgh_var1,strentgh_var2)
    elif dist_strentgh=="Beta Distribution": 
        strentgh_var1 = col6.number_input("Shape parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1bet") 
        strentgh_var2 = col7.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2bet")
        strentgh_var3 = col8.number_input("Not Parameter" , key = str(i)+"_var3bet" , min_value= float(0) , max_value= float(0) ,step=float(0) )  
        strentgh_fun = Beta_Distribution(strentgh_var1,strentgh_var2)
    elif dist_strentgh =="Gumbel Distribution":
        strentgh_var1 = col6.number_input("Location parameter (Mu)" , key = str(i)+"_var1")
        strentgh_var2 = col7.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") 
        strentgh_var3 = col8.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0)) 
        strentgh_fun = Gumbel_Distribution(strentgh_var1,strentgh_var2)
    elif dist_strentgh =="Weibull Distribution":
        strentgh_var1 = col6.number_input("Scale parameter (Alpha)" , min_value=  float(np.finfo(float).eps), value=10.0, key = str(i)+"_var1" )
        strentgh_var2 = col7.number_input("Shape parameter (Beta)" , min_value=  float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") 
        strentgh_var3 = col8.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) 
        strentgh_fun = Weibull_Distribution(strentgh_var1,strentgh_var2,strentgh_var3)
    elif dist_strentgh =="Lognormal Distribution":
        strentgh_var1 = col6.number_input("Location parameter (Mu)" , key = str(i)+"_var1") 
        strentgh_var2 = col7.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2" ) 
        strentgh_var3 = col8.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) 
        strentgh_fun = Lognormal_Distribution(strentgh_var1,strentgh_var2,strentgh_var3)
    elif dist_strentgh  =="Gamma Distribution":
        strentgh_var1 = col6.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=2.0, key = str(i)+"_var1") 
        strentgh_var2 = col7.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=3.0, key = str(i)+"_var2") 
        strentgh_var3 = col8.number_input("Location parameter (Gamma)", value=3.0, key = str(i)+"_var3" ) 
        strentgh_fun = Gamma_Distribution(strentgh_var1,strentgh_var2,strentgh_var3)
    elif dist_strentgh =="Loglogistic Distribution":
        strentgh_var1 = col6.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") 
        strentgh_var2 = col7.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") 
        strentgh_var3 = col8.number_input("Location parameter (Gamma)" , key = str(i)+"_var3") 
        strentgh_fun = Loglogistic_Distribution(strentgh_var1,strentgh_var2,strentgh_var3)

    stress = stress_fun
    strength = strentgh_fun

    if st.button("Calculate"):
        if (type(stress) == Normal_Distribution and type(strength) == Normal_Distribution):
        ## NORMAL FUNCTION   
            sigma_strength = strength.sigma
            mu_strength = strength.mu
            sigma_stress = stress.sigma
            mu_stress = stress.mu
            prob_of_failure = ss.norm.cdf( -(mu_strength - mu_stress) / ((sigma_strength ** 2 + sigma_stress ** 2) ** 0.5) )
            title = "'Stress - Strength Normal"
        else:
            x = np.linspace(min(stress.quantile(1e-8), strength.quantile(1e-8)),max(strength.quantile(1 - 1e-8), stress.quantile(1 - 1e-8)),1000,)
            prob_of_failure = np.trapz(strength.PDF(x, show_plot=False) * stress.SF(x, show_plot=False), x)
            title = 'Stress - Strength Interference'


        xlims = plt.xlim(auto=None)
        xmin = stress.quantile(0.00001)
        xmax = strength.quantile(0.99999)
        if abs(xmin) < (xmax - xmin) / 4:
            xmin = 0  # if the lower bound on xmin is near zero (relative to the entire range) then just make it zero
        if type(stress) == Beta_Distribution:
            xmin = 0
        if type(strength) == Beta_Distribution:
            xmax = 1
        xvals = np.linspace(xmin, xmax, 10000)
        stress_PDF = stress.PDF(xvals=xvals, show_plot=False)
        strength_PDF = strength.PDF(xvals=xvals, show_plot=False)
        Y = [(min(strength_PDF[i], stress_PDF[i])) for i in range(len(xvals))]  # finds the lower of the two lines which is used as the upper boundary for fill_between

        failure_text = str("Probability of\nfailure = " + str(round_to_decimals(prob_of_failure, 4)))

        st.write("Stress Distribution:", stress.param_title_long)
        st.write("Strength Distribution:", strength.param_title_long)
        st.write( "Probability of failure (stress > strength):", round_to_decimals(prob_of_failure * 100), "%",)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xvals, y=stress_PDF, mode='lines', name = 'Stress',  marker=dict(color = 'rgba(0, 0, 255, 0.5)'), visible = True, fill='tozeroy'))
        fig.add_trace(go.Scatter(x=xvals, y=strength_PDF, mode='lines', name = 'Strength',  marker=dict(color = 'rgba(0, 255, 0, 0.5)'), visible = True, fill='tozeroy'))
        fig.update_layout(width = 1900, height = 600, title = title, yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log,title_text=title ) #size of figure
        fig.update_xaxes(title = 'Stress and Strength Units')
        fig.update_yaxes(title = 'Probability density')	
        st.plotly_chart(fig)

    else:

        if  stress.mean > strength.mean:
            st.error("Warning: strength mean must be greather than stress mean")