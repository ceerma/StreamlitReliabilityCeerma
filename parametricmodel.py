import streamlit as st
import reliability 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Normal_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution


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






st.title("Parametric Model")
st.write("In this module, you can select and analyze the most common probability distributions in reliability ")






distribution_name = st.selectbox( 'Select the distribution.', ('Exponential Distribution',
 'Normal Distribution', 'Beta Distribution', 'Gumbel Distribution' ,'Weibull Distribution', 'Lognormal Distribution',
 'Gamma Distribution','Loglogistic Distribution'))



if distribution_name == "Exponential Distribution":
    var1_name, var2_name, var3_name= "Lambda", "Gamma" , "None"
    var1 = st.number_input("Scale parameter (Lambda)" , min_value= float(np.finfo(float).eps), value=10.0)
    var2 = st.number_input("Displacement parameter (Gamma)")
    dist_fun= Exponential_Distribution
    
    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \lambda = \text{Scale parameter } (\lambda > 0 )''' ) 
    equation.latex(r''' \text{Limits: } ( t \leq 0) ''' )
    equation.latex(r''' \text{PDF: } f(t) = \lambda e^{-\lambda t}''') 
    equation.latex(r''' \text{CDF: } F(t) = 1 - \lambda e^{-\lambda t}''')
    equation.latex(r''' \text{SF: } R(t) = e^{-\lambda t}''') 
    equation.latex(r''' \text{HF: } h(t) = \lambda''')  
    equation.latex(r''' \text{CHF: } H(t) = \lambda t''') 
    #dist = Exponential_Distribution(Lambda=var1,gamma=var2)
elif distribution_name =="Normal Distribution":
    var1_name, var2_name, var3_name = "mu","sigma" ,"None"
    
    var1 = st.number_input("Location parameter (Mu)" )
    var2 = st.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0)
    dist_fun= Normal_Distribution

    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \mu = \text{Location parameter } ( -\infty < \mu < \infty )''' ) 
    equation.latex(r''' \sigma = \text{Scale parameter } ( \sigma > 0)''' ) 
    equation.latex(r''' \text{Limits: } ( - \infty < t < \infty ) ''' )
    equation.latex(r''' \text{PDF: } f(t) = \frac{1}{\sigma \sqrt{2\pi}} e^{\frac{1}{2}  \left(\frac{t-\mu}{\sigma}\right)^2 }  =   \frac{1}{\sigma}  \phi  \left[ \frac{t-\mu}{\sigma} \right]     ''') 
    equation.latex(r''' \text{Where } \phi \text{  is the standard normal PDF with } \mu = 0  \text{ and } \sigma = 1''') 
    equation.latex(r''' \text{CDF: } F(t) = \frac{1}{\sigma \sqrt{2\pi}} \int^t_{-\infty} e^{\left[ - \frac{1}{2} \left(  \frac{\theta -\mu}{\sigma}\right)^2 \right] d\theta   }   = \frac{1}{2} + \frac{1}{2} erf \left(\frac{t-\mu}{\sigma\sqrt{2}}\right) = \Phi \left(\frac{t-\mu}{\sigma}\right) ''')
    equation.latex(r''' \text{Where } \Phi \text{  is the standard normal CDF with } \mu = 0  \text{ and } \sigma = 1''')
    equation.latex(r''' \text{SF: } R(t) = 1- \Phi \left(\frac{t-\mu}{\sigma}\right)  = \Phi \left(\frac{\mu-t}{\sigma}\right) ''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{ \phi  \left[ \frac{t-\mu}{\sigma} \right]  }{ \sigma \left(  \Phi \left[\frac{\mu-t}{\sigma}\right]  \right)}   ''')  
    equation.latex(r''' \text{CHF: } H(t) =  -ln \left[ \Phi \left(\frac{\mu-t}{\sigma}\right) \right]    ''') 
    #dist = Normal_Distribution(mu=var1,sigma=var2)
elif distribution_name =="Beta Distribution":
    var1_name, var2_name, var3_name = "alpha","beta", "None"   
    var1 = st.number_input("Shape parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0)
    var2 = st.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0)
    dist_fun= Beta_Distribution
    
    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \alpha = \text{Shape parameter } (  \alpha > 0) ''' ) 
    equation.latex(r''' \beta = \text{Shape parameter } ( \beta > 0) ''' ) 
    equation.latex(r''' \text{Limits: } ( 0 \leq t \leq 1 ) ''' )
    equation.latex(r''' \text{PDF: } f(t) = \frac{ \Gamma(\alpha +\beta ) }{\Gamma(\alpha) \Gamma(\beta) } t^{\alpha -1} (1-t)^{\beta-1 } =   \frac{1}{ B(\alpha,\beta}  t^{\alpha -1} (1-t)^{\beta-1 } ''') 
    equation.latex(r''' \text{Where } \Gamma(x) \text{ is the complete gamma function. } \Gamma(x) = \int^\infty_{0}  t^{x-1} e^{-t} dt ''') 
    equation.latex(r''' \text{Where } B(x,y) \text{ is the complete beta function. } B(x,y) = \int^{1}_{0} (1- t)^{y-1} dt ''') 
    equation.latex(r''' \text{CDF: } F(t) = \frac{ \Gamma(\alpha +\beta ) }{\Gamma(\alpha) \Gamma(\beta) } \int^{t}_{0}  \theta^{\alpha-1} (1-\theta^{\beta - 1} d\theta  = \frac{B_t (t| \alpha,\beta) }{ B (\alpha,\beta)}  = I_t (t| \alpha,\beta)   ''')
    equation.latex(r''' \text{Where }  B_t(t|x,y) \text{  is the incomplete beta function.} B_t (t|x,y) =   \int^{t}_{0}  \theta^{x-1} (1- \theta)^{y-1} d\theta   ''')
    equation.latex(r''' \text{Where }  I_t(t|x,y) \text{ is the regularized incomplete beta function which is defined in terms} ''')
    equation.latex(r''' \text{ of the incomplete beta function and the complete beta function.} I_t(t|x,y) =   \frac{B_t(t|x,y)}{B_t(x,y)}  ''')
    equation.latex(r''' \text{SF: } R(t) = 1 - I_t(t|\alpha,\beta)  ''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{t^{y-1} (1-t) }{ B(\alpha,\beta) - B_t(t|\alpha,\beta)    }   ''')  
    equation.latex(r''' \text{CHF: } H(t) =  -ln \left[ 1 - I_t(t|\alpha,\beta) \right]    ''') 
    #dist = Beta_Distribution(alpha=var1, beta=var2) 
elif distribution_name =="Gumbel Distribution":
    var1_name, var2_name, var3_name  = "mu","sigma" , "None"
    var1 = st.number_input("Location parameter (Mu)" )
    var2 = st.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0)

    dist_fun= Gumbel_Distribution

    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \mu = \text{Location parameter } (-\infty < \mu < \infty )''' ) 
    equation.latex(r''' \sigma = \text{Scale parameter } (\sigma > 0 )''' ) 
    equation.latex(r''' \text{Limits: } ( -\infty <t < \infty ) ''' )
    equation.latex(r''' \text{PDF: } f(t) = \frac{1}{\sigma} e^{z -e^{z}}''') 
    equation.latex(r''' \text{Where }  z = \frac{t-\mu}{ \sigma}  ''')
    equation.latex(r''' \text{CDF: } F(t) = 1 - e^{-e^{z}}''')
    equation.latex(r''' \text{SF: } R(t) =  e^{-e^{z}}''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{e^{z}}{\sigma} ''')  
    equation.latex(r''' \text{CHF: } H(t) =  e^{z}  ''') 
    #dist = Gumbel_Distribution(mu=var1,sigma=var2)
elif distribution_name =="Weibull Distribution":
    var1_name, var2_name, var3_name = "alpha", "beta", "gamma"
    var1 = st.number_input("Scale parameter (Alpha)" , min_value= float(np.finfo(float).eps), value=10.0)
    var2 = st.number_input("Shape parameter (Beta)" , min_value= float(np.finfo(float).eps), value=1.0)
    var3 = st.number_input("Location parameter (Gamma)" )
    dist_fun= Weibull_Distribution
    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \alpha = \text{Scale parameter } (\alpha > 0 )''' ) 
    equation.latex(r''' \beta = \text{Shape parameter } (\beta > 0 )''' ) 
    equation.latex(r''' \text{Limits: } ( t \leq 0) ''' )
    equation.latex(r''' \text{PDF: } f(t) = \frac{\beta t^{\beta-1} }{\alpha^\beta} e^{ \frac{t}{\alpha}^\beta } = \frac{\beta}{\alpha} \left( \frac{t}{\alpha}\right)^{(\beta -1)} e^{-(\frac{t}{\alpha})^\beta }  ''') 
    equation.latex(r''' \text{CDF: } F(t) = 1 - e^{-(\frac{t}{\alpha})^\beta }  ''')
    equation.latex(r''' \text{SF: } R(t) = e^{-(\frac{t}{\alpha})^\beta }  ''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{\beta}{\alpha} \left( \frac{t}{\alpha}\right)^{(\beta -1)}  ''')  
    equation.latex(r''' \text{CHF: } H(t) = (\frac{t}{\alpha})^\beta  ''') 
    #dist = Weibull_Distribution(alpha=var1, beta=var2,gamma=var3)
elif distribution_name =="Lognormal Distribution":
    var1_name, var2_name,var3_name = "mu","sigma", "gamma" 
    var1 = st.number_input("Location parameter (Mu)" )
    var2 = st.number_input("Scale parameter (Sigma)", min_value= float(np.finfo(float).eps), value=1.0)
    var3 = st.number_input("Location parameter (Gamma)" )
    dist_fun= Lognormal_Distribution
    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \alpha = \text{Scale parameter } (-\infty  < \mu < \infty  )''' ) 
    equation.latex(r''' \beta = \text{Shape parameter } (\sigma > 0 )''' ) 
    equation.latex(r''' \text{Limits: } ( t \leq 0) ''' )
    equation.latex(r''' \text{PDF: } f(t) = \frac{1}{\sigma t \sqrt{2 \pi}}   e^{ -\frac{1}{2} \left( \frac{ln(t) -\mu}{\sigma} \right)^2  } = \frac{1}{\sigma t} \phi \left[ \frac{ln(t) -\mu}{\sigma}  \right]  ''') 
    equation.latex(r''' \text{Where }  \phi  \text{is the standard normal PDF with } \mu = 0 \text{and }  \sigma =1 ''')
    equation.latex(r''' \text{CDF: } F(t) = \frac{1}{\sigma \sqrt{2\pi}} \int^t_{0} \frac{1}{\theta} e^{\left[ - \frac{1}{2} \left(  \frac{ln(\theta) -\mu}{\sigma}\right)^2 \right] d\theta   }   = \frac{1}{2} + \frac{1}{2} erf \left(\frac{ln(t)-\mu}{\sigma\sqrt{2}}\right) = \Phi \left(\frac{ln(t)-\mu}{\sigma}\right) ''')
    equation.latex(r''' \text{Where } \Phi \text{  is the standard normal CDF with } \mu = 0  \text{ and } \sigma = 1''')
    equation.latex(r''' \text{SF: } R(t) = 1- \Phi \left(\frac{ln(t)-\mu}{\sigma}\right)  = \Phi \left(\frac{\mu- ln(t)}{\sigma}\right) ''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{ \phi  \left[ \frac{ln(t)-\mu}{\sigma} \right]  }{ \sigma \left(  \Phi \left[\frac{\mu-ln(t)}{\sigma}\right]  \right)}   ''')  
    equation.latex(r''' \text{CHF: } H(t) =  -ln \left[  1- \Phi \left(\frac{ln(t) -\mu}{\sigma}\right) \right]    ''') 
    #dist = Lognormal_Distribution(mu=var1,sigma=var2,gamma=var3)
elif distribution_name =="Gamma Distribution":
    var1_name, var2_name, var3_name = "alpha", "beta", "gamma"  
    var1 = st.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0)
    var2 = st.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=1.0)
    var3 = st.number_input("Location parameter (Gamma)" )
    dist_fun= Gamma_Distribution

    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \alpha = \text{Scale parameter } ( \alpha > 0)''' ) 
    equation.latex(r''' \beta = \text{Shape parameter } ( \beta > 0)''' ) 
    equation.latex(r''' \text{Limits: } ( t \leq 0 ) ''' )
    equation.latex(r''' \text{PDF: } f(t) =  \frac{t^{\beta-1}}{\Gamma(\beta)\alpha^\beta} e^{\frac{t}{\alpha}}    ''')
    equation.latex(r''' \text{Where } \Gamma(x) \text{  is the complete gamma function. } \Gamma(x) = \int^\infty_{0}  t^{x-1} e^{-t} dt''') 
    equation.latex(r''' \text{CDF: } F(t) =  \frac{1}{\Gamma(\beta)} \gamma( \beta, \frac{t}{\alpha}) ''')
    equation.latex(r''' \text{Where } \gamma(x,y) \text{  is the lower incomplete gamma function. } \gamma(x,y) = \frac{1}{\Gamma(x)}  \int^y_{0}  t^{x-1} e^{-t} dt  ''') 
    equation.latex(r''' \text{SF: } R(t) =   \frac{1}{\Gamma(\beta)} \Gamma(\beta,\frac{t}{\alpha}) ''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{t^{\beta-1}e^{-\frac{t}{\alpha}} }{\alpha^\beta \Gamma(\beta,\frac{t}{\alpha})}''')  
    equation.latex(r''' \text{CHF: } H(t) =  -ln \left[  \frac{1}{\Gamma(\beta)} \Gamma(\beta,\frac{t}{\alpha}) \right]    ''') 

    #dist = Gamma_Distribution(alpha=var1, beta=var2,gamma=var3) 
elif distribution_name =="Loglogistic Distribution":
    var1_name, var2_name, var3_name = "alpha", "beta", "gamma"
    var1 = st.number_input("Scale parameter (Alpha)", min_value= float(np.finfo(float).eps), value=1.0)
    var2 = st.number_input("Shape parameter (Beta)", min_value= float(np.finfo(float).eps), value=5.0)
    var3 = st.number_input("Location parameter (Gamma)" )
    dist_fun= Loglogistic_Distribution
    
    equation = st.beta_expander("Equation Information")
    equation.latex(r''' \alpha = \text{Scale parameter } ( \alpha > 0)''' ) 
    equation.latex(r''' \beta = \text{Shape parameter } ( \beta > 0)''' ) 
    equation.latex(r''' \text{Limits: } ( t \leq 0 ) ''' )
    equation.latex(r''' \text{PDF: } f(t) =  \frac{  \frac{\beta}{\alpha}  (\frac{t}{\alpha})^{\beta-1} }{ \left( 1 + (\frac{t}{\alpha})^\beta \right)^2 } ''')
    equation.latex(r''' \text{CDF: } F(t) = \frac{1}{1 + (\frac{t}{\alpha})^{-\beta} } = \frac{(\frac{t}{\alpha})^{\beta}}{ 1 + (\frac{t}{\alpha})^{\beta}} = \frac{t^{\beta}}{\alpha^{\beta} + t^{\beta}} ''')
    equation.latex(r''' \text{SF: } R(t) =  \frac{1}{1 + (\frac{t}{\alpha})^{\beta}} ''') 
    equation.latex(r''' \text{HF: } h(t) =  \frac{  \frac{\beta}{\alpha}  (\frac{t}{\alpha})^{\beta-1} }{  1 + (\frac{t}{\alpha})^\beta } ''')  
    equation.latex(r''' \text{CHF: } H(t) =  -ln \left( 1 + (\frac{t}{\alpha})^{\beta} \right)    ''') 

else:
    st.write("Select a distribution")


expander = st.beta_expander("Plot parameter")
points_quality = expander.number_input('Number of points to plot', min_value=5,value = 1000, max_value = 100000 )
show_variable = expander.checkbox("Show distribution properties.", value=True, key=None)
st.write(" ")
if st.button("Plot distribution"):
    
    properties_dist = st.empty()

    if var3_name == "None":
        dist = dist_fun(var1,var2)
    else:
        dist = dist_fun(var1,var2,var3)

    if distribution_name =="Beta Distribution" and (var1 <=1 or var2 <=1):
        properties_dist.text("""
        Mean: {}
        Median: {}
        Mode:  No mode exists unless Alpha and Beta are greater than 1.
        Variance: {}
        Standard Deviation: {}
        Skewness: {} 
        Kurtosis: {}
        Excess Kurtosis: {} 
        """.format(dist.mean,dist.median, dist.variance, dist.standard_deviation, dist.skewness, dist.kurtosis, dist.excess_kurtosis ) )
    else:
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

    if distribution_name == "Loglogistic Distribution" and var2 <=1:
        st.write("No plot when beta less or equal than 1")
    else:
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
        if show_variable:
            if distribution_name =="Normal Distribution":
                fig.add_vline(x=dist.mean, line_dash="dash", annotation_text="Mean, Median, Mode", annotation_position="top right")
            elif distribution_name =="Beta Distribution" and (var1 <=1 or var2 <=1):
                fig.add_vline(x=dist.mean, line_dash="dash", annotation_text="Mean", annotation_position="top right")
                fig.add_vline(x=dist.median, line_dash="dash", annotation_text="Median", annotation_position="top right")
            else:
                fig.add_vline(x=dist.mean, line_dash="dash", annotation_text="Mean", annotation_position="top right")
                fig.add_vline(x=dist.median, line_dash="dash", annotation_text="Median", annotation_position="top right")
                fig.add_vline(x=dist.mode, line_dash="dash", annotation_text="Mode", annotation_position="top right")
        fig.update_layout(width = 1900, height = 600, title = 'Dados analisados', yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log,title_text='Parametric Model - {} ({}) '.format(distribution_name,dist.param_title)) #size of figure
        fig.update_xaxes(title = 'Time')
        fig.update_yaxes(title = 'Probability density')			
        st.plotly_chart(fig)
