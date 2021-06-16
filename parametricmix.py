import streamlit as st
import reliability 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Normal_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution, Mixture_Model, Competing_Risks_Model


st.set_page_config(page_title="Parametric Model",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
#ReportStatus {visibility: hidden;}

</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
href_homepage = f'<a href="https://reliability.ceerma.com/" style="text-decoration: none; color :black;" > <button kind="primary" class="css-qbe2hs edgvbvh1">Go to Homepage</button></a>'
st.markdown(href_homepage, unsafe_allow_html=True)

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






st.title("Mixture models & Competing risks Model")
## Select Model
Model_selected = st.selectbox( 'Select Model.', ('Mixture Model', 'Competing Risk model'))
if Model_selected == 'Mixture Model':
    st.write("Mixture models are a combination of two or more distributions added together to create a distribution that has a shape with more flexibility than a single distribution. Each of the mixture’s components must be multiplied by a proportion, and the sum of all the proportions is equal to 1.")
if Model_selected == 'Competing Risk model':
    st.write("Competing risks models are a combination of two or more distributions that represent failure modes which are “competing” to end the life of the system being modelled. This model is similar to a mixture model in the sense that it uses multiple distributions to create a new model that has a shape with more flexibility than a single distribution. However, unlike in mixture models, we are not adding proportions of the PDF or CDF, but are instead multiplying the survival functions.")



number_distribution = st.number_input("N° of distributions:", min_value=1, max_value= 20)
init_val= 1/number_distribution

if Model_selected == 'Mixture Model':

    list_distribution=[]
    var1_list=[]
    var2_list=[]
    var3_list=[]
    weight_list=[]
    dist_fun_list=[]
    col1, col2, col3, col4, col5 = st.beta_columns([3, 1,1,1,2])

    for i in range(number_distribution):
        list_distribution.append(col1.selectbox( 'Select the distribution.', ('Exponential Distribution',
        'Normal Distribution', 'Gumbel Distribution' ,'Weibull Distribution', 'Lognormal Distribution',
        'Gamma Distribution','Loglogistic Distribution') , key = str(i) + "_dist" ) )
        if list_distribution[i] == "Exponential Distribution":
            var1_list.append( col2.number_input("Scale parameter (Lambda)" , min_value= float(np.finfo(float).eps), value=10.0 , key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Displacement parameter (Gamma)" , key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0) ) ) 
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Exponential_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i] == "Normal Distribution":
            var1_list.append( col2.number_input("Location parameter (Mu)" ) )
            var2_list.append( col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0) )
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0) ) ) 
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Normal_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i]=="Beta Distribution": 
            var1_list.append( col2.number_input("Shape parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2")) 
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0) ) ) 
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Beta_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i] =="Gumbel Distribution":
            var1_list.append( col2.number_input("Location parameter (Mu)" , key = str(i)+"_var1")) 
            var2_list.append( col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2")) 
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0)) ) 
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Gumbel_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i] =="Weibull Distribution":
            var1_list.append( col2.number_input("Scale parameter (Alpha)" , min_value=  float(np.finfo(float).eps), value=10.0, key = str(i)+"_var1" ))
            var2_list.append( col3.number_input("Shape parameter (Beta)" , min_value=  float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) )
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Weibull_Distribution(var1_list[i],var2_list[i],var3_list[i]))
        elif list_distribution[i] =="Lognormal Distribution":
            var1_list.append( col2.number_input("Location parameter (Mu)" , key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2" ) )
            var3_list.append( col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) )
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Lognormal_Distribution(var1_list[i],var2_list[i],var3_list[i]))
        elif list_distribution[i]  =="Gamma Distribution":
            var1_list.append( col2.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) )
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Gamma_Distribution(var1_list[i],var2_list[i],var3_list[i]))
        elif list_distribution[i] =="Loglogistic Distribution":
            var1_list.append( col2.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Location parameter (Gamma)" , key = str(i)+"_var3") )
            weight_list.append( col5.number_input("Proportions" , min_value= float(0) , max_value= float(1), value=init_val , key = str(i)+"_weight") )
            dist_fun_list.append(Loglogistic_Distribution(var1_list[i],var2_list[i],var3_list[i]))

if Model_selected == 'Competing Risk model':

    list_distribution=[]
    var1_list=[]
    var2_list=[]
    var3_list=[]
    dist_fun_list=[]
    col1, col2, col3, col4 = st.beta_columns([3, 1,1,1])

    for i in range(number_distribution):
        list_distribution.append(col1.selectbox( 'Select the distribution.', ('Exponential Distribution',
        'Normal Distribution', 'Gumbel Distribution' ,'Weibull Distribution', 'Lognormal Distribution',
        'Gamma Distribution','Loglogistic Distribution') , key = str(i) + "_dist" ) )
        if list_distribution[i] == "Exponential Distribution":
            var1_list.append( col2.number_input("Scale parameter (Lambda)" , min_value= float(np.finfo(float).eps), value=10.0 , key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Displacement parameter (Gamma)" , key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0) ) ) 
            dist_fun_list.append(Exponential_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i] == "Normal Distribution":
            var1_list.append( col2.number_input("Location parameter (Mu)" ) )
            var2_list.append( col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0) )
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0) ) ) 
            dist_fun_list.append(Normal_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i]=="Beta Distribution": 
            var1_list.append( col2.number_input("Shape parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2")) 
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0) ) ) 
            dist_fun_list.append(Beta_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i] =="Gumbel Distribution":
            var1_list.append( col2.number_input("Location parameter (Mu)" , key = str(i)+"_var1")) 
            var2_list.append( col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2")) 
            var3_list.append( col4.number_input("Not Parameter" , key = str(i)+"_var3" , min_value= float(0) , max_value= float(0) ,step=float(0)) ) 
            dist_fun_list.append(Gumbel_Distribution(var1_list[i],var2_list[i]))
        elif list_distribution[i] =="Weibull Distribution":
            var1_list.append( col2.number_input("Scale parameter (Alpha)" , min_value=  float(np.finfo(float).eps), value=10.0, key = str(i)+"_var1" ))
            var2_list.append( col3.number_input("Shape parameter (Beta)" , min_value=  float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) )
            dist_fun_list.append(Weibull_Distribution(var1_list[i],var2_list[i],var3_list[i]))
        elif list_distribution[i] =="Lognormal Distribution":
            var1_list.append( col2.number_input("Location parameter (Mu)" , key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2" ) )
            var3_list.append( col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) )
            dist_fun_list.append(Lognormal_Distribution(var1_list[i],var2_list[i],var3_list[i]))
        elif list_distribution[i]  =="Gamma Distribution":
            var1_list.append( col2.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Location parameter (Gamma)", key = str(i)+"_var3" ) )
            dist_fun_list.append(Gamma_Distribution(var1_list[i],var2_list[i],var3_list[i]))
        elif list_distribution[i] =="Loglogistic Distribution":
            var1_list.append( col2.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var1") )
            var2_list.append( col3.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0, key = str(i)+"_var2") )
            var3_list.append( col4.number_input("Location parameter (Gamma)" , key = str(i)+"_var3") )
            dist_fun_list.append(Loglogistic_Distribution(var1_list[i],var2_list[i],var3_list[i]))


expander = st.beta_expander("Plot parameter")
points_quality = expander.number_input('Number of points to plot', min_value=5,value = 1000, max_value = 100000 )
show_variable = expander.checkbox("Show distribution properties.", value=True, key=None)
st.write(" ")

if st.button("Plot distribution"):
        
    if Model_selected == 'Mixture Model':    
        dist = Mixture_Model(distributions=dist_fun_list, proportions=weight_list)
    if Model_selected == 'Competing Risk model':
        dist = Competing_Risks_Model(distributions=dist_fun_list)
    
    
    properties_dist = st.empty()

    properties_dist.text("""
    Mean: {}
    Median: {}
    Mode:  {}
    Variance: {}
    Standard Deviation: {}
    Skewness: {} 
    Kurtosis: {}
    Excess Kurtosis: {} 
    """.format(dist.mean,dist.median ,dist.mode, dist.variance, dist.standard_deviation, dist.skewness, dist.kurtosis, dist.excess_kurtosis ) )

    dist.PDF()    
    x_min,x_max = plt.gca().get_xlim()
    x = np.linspace(x_min,x_max,points_quality)
    y_PDF = dist.PDF(xvals=x)
    y_CDF = dist.CDF(xvals=x)
    y_SF = dist.SF(xvals=x)
    y_HF = dist.HF(xvals=x)
    y_CHF = dist.CHF(xvals=x)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_PDF, mode='lines', name = 'PDF',  marker=dict(color = 'rgba(255, 223, 118, 0.9)'), visible = True))
    fig.add_trace(go.Scatter(x=x, y=y_CDF, mode='lines', name = 'CDF',  marker=dict(color = 'rgba(255, 0, 0, 0.9)'), visible = 'legendonly'))
    fig.add_trace(go.Scatter(x=x, y=y_SF, mode='lines', name = 'SF',  marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = 'legendonly'))
    fig.add_trace(go.Scatter(x=x, y=y_HF, mode='lines', name = 'HF',  marker=dict(color = 'rgba(0, 0, 255, 0.9)'), visible = 'legendonly'))
    fig.add_trace(go.Scatter(x=x, y=y_CHF, mode='lines', name = 'CHF',  marker=dict(color = 'rgba(135, 45, 54, 0.9)'), visible = 'legendonly'))
    fig.update_layout(width = 1900, height = 600, title = 'Dados analisados', yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log,title_text=' {}  '.format(Model_selected)) #size of figure
    fig.update_xaxes(title = 'Time')
    fig.update_yaxes(title = 'Probability density')			
    st.plotly_chart(fig)
