import streamlit as st
import reliability 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Beta_2P, Fit_Everything, Fit_Exponential_1P, Fit_Exponential_2P, Fit_Gamma_2P, Fit_Gumbel_2P, Fit_Loglogistic_2P, Fit_Loglogistic_3P, Fit_Lognormal_2P, Fit_Lognormal_3P, Fit_Normal_2P, Fit_Weibull_2P, Fit_Weibull_3P, Fit_Gamma_3P
from reliability.Distributions import Weibull_Distribution, Lognormal_Distribution, Exponential_Distribution, Normal_Distribution, Gamma_Distribution, Beta_Distribution, Loglogistic_Distribution, Gumbel_Distribution
from reliability.Probability_plotting import plot_points, plotting_positions
from reliability.ALT_fitters import Fit_Everything_ALT

st.set_page_config(page_title="Accelerated Life Testing",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

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






st.title("Accelerated Life Testing")
st.write("In this module, you can provide your Accelerated Life Testing (ALT) data (complete or incomplete) and fit the most common probability distributions in reliability ")

with st.beta_expander(label='Help'):
    st.write('When using this module, please take into consideration the following points:')
    st.write('- There is no need to sort the data in any particular order as this is all done automatically;')
    st.write('- For each stress level, there must be at least one failure data;')

expander = st.beta_expander("Data format")
expander.info('Upload an excel file thar contains the following columns: failure or right-censored time ("Time"), \
    the time type, if failure or right censored ("Type"), and the stress level ("Stress").')
df = {'Time': [10, 15, 8, 20, 21, 12, 13, 30, 5], \
    'Type': ['F', 'F', 'C', 'F', 'C', 'C', 'F', 'F', 'C'], \
    'Stress': [20, 20, 20, 40, 40, 40, 60, 60, 60]}
df = pd.DataFrame.from_dict(df)
expander.write(df, width=50)

col2_1, col2_2 = st.beta_columns(2)
uploaded_file = col2_1.file_uploader("Upload a XLSX file", \
    type="xlsx", accept_multiple_files=False)
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    # sdf = df.style.format('{:10n}'.format)
    col2_2.dataframe(df)
    fdata = df[df['Type'] == 'F']
    ftime = np.array(fdata.iloc[:,0])
    fstress = np.array(fdata.iloc[:,2])
    cdata = df[df['Type'] == 'C']
    ctime = np.array(cdata.iloc[:,0])
    cstress = np.array(cdata.iloc[:,2])

use_level = st.number_input("Use level stress (optional)", min_value=0, value=0)
IC = 0.8

include = st.multiselect('Choose which distribution(s) you want to fit to your data', ['Weibull Exponential', 'Weibull Eyring', 'Weibull Power', \
    'Lognormal Exponential', 'Lognormal Eyring', 'Lognormal Power', 'Normal Exponential', 'Normal Eyring', 'Normal Power', \
    'Exponential Exponential', 'Exponential Eyring', 'Exponential Power'])

method = st.radio('Choose the optimizer', ('TNC', 'L-BFGS-B', 'Powell'))
st.info('The optimizers are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned.')
metric = st.radio('Choose a goodness of fit criteria', ('BIC', 'AICc', 'AD', 'Log-likelihood'))

all_dist = ['Weibull Exponential', 'Weibull Eyring', 'Weibull Power', \
    'Lognormal Exponential', 'Lognormal Eyring', 'Lognormal Power', 'Normal Exponential', 'Normal Eyring', 'Normal Power', \
    'Exponential Exponential', 'Exponential Eyring', 'Exponential Power']

exclude = list(set(all_dist)-set(include))

exc = []
for dis in exclude:
    if dis == 'Weibull Exponential':
        exc.append('Weibull_Exponential')
    if dis == 'Weibull Eyring':
        exc.append('Weibull_Eyring')
    if dis == 'Weibull Power':
        exc.append('Weibull_Power')
    if dis == 'Lognormal Exponential':
        exc.append('Lognormal_Exponential')
    if dis == 'Lognormal Eyring':
        exc.append('Lognormal_Eyring')
    if dis == 'Lognormal Power':
        exc.append('Lognormal_Power')
    if dis == 'Normal Exponential':
        exc.append('Normal_Exponential')
    if dis == 'Normal Eyring':
        exc.append('Normal_Eyring')
    if dis == 'Normal Power':
        exc.append('Normal_Power')
    if dis == 'Exponential Exponential':
        exc.append('Exponential_Exponential')
    if dis == 'Exponential Eyring':
        exc.append('Exponential_Eyring')
    if dis == 'Exponential Power':
        exc.append('Exponential_Power')

expander = st.beta_expander("Plot parameter")
points_quality = expander.number_input('Number of points to plot', min_value=5,value = 1000, max_value = 100000 )
show_variable = expander.checkbox("Show distribution properties.", value=True, key=None)
st.write(" ")

if st.button("Fit ALT model"):


    if not exc:
        exc = None
    elif len(exc) == 12:
        st.error('Please, choose at least one distribution to fit.')
        st.stop()
    
    if use_level == 0:
        use_level = None

    results = Fit_Everything_ALT(failures=ftime, failure_stress_1=fstress, right_censored=ctime, right_censored_stress_1=cstress, use_level_stress=use_level, exclude=exc, sort_by=metric, print_results=False,\
        show_probability_plot=False, show_best_distribution_probability_plot=False, CI=IC, optimizer=method)

    st.write('## Results of all fitted ALT models')
    st.write(results.results)

    # st.write('## Results of the best fitted distribution')
    # dist = results.best_distribution
    # distribution_name = results.best_distribution_name

    # percentiles = np.linspace(1, 99, num=99)						
    # IC = 0.8

    # if distribution_name == 'Exponential_1P':
    #     new_fit = Fit_Exponential_1P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Exponential_Distribution(Lambda=new_fit.Lambda_upper)
    #     fit_lw = Exponential_Distribution(Lambda=new_fit.Lambda_lower)
    # elif distribution_name == 'Exponential_2P':
    #     new_fit = Fit_Exponential_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Exponential_Distribution(Lambda=new_fit.Lambda_upper, gamma=new_fit.gamma_upper)
    #     fit_lw = Exponential_Distribution(Lambda=new_fit.Lambda_lower, gamma=new_fit.gamma_lower)
    # elif distribution_name == 'Normal_2P':
    #     new_fit = Fit_Normal_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Normal_Distribution(mu=new_fit.mu_upper, sigma=new_fit.sigma_lower)
    #     fit_lw = Normal_Distribution(mu=new_fit.mu_lower, sigma=new_fit.sigma_lower)
    # elif distribution_name == 'Beta_2P':
    #     new_fit = Fit_Beta_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Beta_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper)
    #     fit_lw = Beta_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower)
    # elif distribution_name == 'Gumbel_2P':
    #     new_fit = Fit_Gumbel_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Gumbel_Distribution(mu=new_fit.mu_upper, sigma=new_fit.sigma_upper)
    #     fit_lw = Gumbel_Distribution(mu=new_fit.mu_lower, sigma=new_fit.sigma_lower)
    # elif distribution_name == 'Weibull_2P':
    #     new_fit = Fit_Weibull_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Weibull_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper)
    #     fit_lw = Weibull_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower)
    # elif distribution_name == 'Weibull_3P':
    #     new_fit = Fit_Weibull_3P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Weibull_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper, gamma=new_fit.gamma_upper)
    #     fit_lw = Weibull_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower, gamma=new_fit.gamma_lower)
    # elif distribution_name == 'Lognormal_2P':
    #     new_fit = Fit_Lognormal_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Lognormal_Distribution(mu=new_fit.mu_upper, sigma=new_fit.sigma_upper)
    #     fit_lw = Lognormal_Distribution(mu=new_fit.mu_lower, sigma=new_fit.sigma_lower)
    # elif distribution_name == 'Lognormal_3P':
    #     new_fit = Fit_Lognormal_3P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Lognormal_Distribution(mu=new_fit.mu_upper, sigma=new_fit.sigma_upper, gamma=new_fit.gamma_upper)
    #     fit_lw = Lognormal_Distribution(mu=new_fit.mu_lower, sigma=new_fit.sigma_lower, gamma=new_fit.gamma_lower)
    # elif distribution_name == 'Gamma_2P':
    #     new_fit = Fit_Gamma_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Gamma_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper)
    #     fit_lw = Gamma_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower)
    # elif distribution_name == 'Gamma_3P':
    #     new_fit = Fit_Gamma_3P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Gamma_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper, gamma=new_fit.gamma_upper)
    #     fit_lw = Gamma_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower, gamma=new_fit.gamma_lower)
    # elif distribution_name == 'Loglogistic_2P':
    #     new_fit = Fit_Loglogistic_2P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Loglogistic_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper)
    #     fit_lw = Loglogistic_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower)
    # elif distribution_name == 'Loglogistic_3P':
    #     new_fit = Fit_Loglogistic_3P(failures=ftime, right_censored=ctime, show_probability_plot=False,print_results=False, percentiles=percentiles, CI=IC, method=method)
    #     fit_up = Loglogistic_Distribution(alpha=new_fit.alpha_upper, beta=new_fit.beta_upper, gamma=new_fit.gamma_upper)
    #     fit_lw = Loglogistic_Distribution(alpha=new_fit.alpha_lower, beta=new_fit.beta_lower, gamma=new_fit.gamma_lower)

    # # properties_dist = st.empty()

    # # if distribution_name =="Beta_2P":
    # #     properties_dist.text("""
    # #     Mean: {}
    # #     Median: {}
    # #     Mode:  No mode exists unless Alpha and Beta are greater than 1.
    # #     Variance: {}
    # #     Standard Deviation: {}
    # #     Skewness: {} 
    # #     Kurtosis: {}
    # #     Excess Kurtosis: {} 
    # #     """.format(dist.mean,dist.median, dist.variance, dist.standard_deviation, dist.skewness, dist.kurtosis, dist.excess_kurtosis ) )
    # # else:
    # #     properties_dist.text("""
    # #     Mean: {}
    # #     Median: {}
    # #     Mode:  {}
    # #     Variance: {}
    # #     Standard Deviation: {}
    # #     Skewness: {} 
    # #     Kurtosis: {}
    # #     Excess Kurtosis: {} 
    # #     """.format(dist.mean,dist.median ,dist.mode, dist.variance, dist.standard_deviation, dist.skewness, dist.kurtosis, dist.excess_kurtosis ) )

    # if distribution_name =="Beta_2P":
    #     properties_dist = {
    #     'Mean': dist.mean, 
    #     'Median': dist.median,
    #     'Mode':  'No mode exists unless Alpha and Beta are greater than 1.',
    #     'Variance': dist.variance,
    #     'Standard Deviation': dist.standard_deviation,
    #     'Skewness': dist.skewness, 
    #     'Kurtosis': dist.kurtosis,
    #     'Excess Kurtosis': dist.excess_kurtosis 
    #     }
    # else:
    #     properties_dist = {
    #     'Mean': dist.mean, 
    #     'Median': dist.median,
    #     'Mode':  dist.mode,
    #     'Variance': dist.variance,
    #     'Standard Deviation': dist.standard_deviation,
    #     'Skewness': dist.skewness, 
    #     'Kurtosis': dist.kurtosis,
    #     'Excess Kurtosis': dist.excess_kurtosis 
    #     }

    # df = pd.DataFrame.from_dict(properties_dist, orient='index', columns=['Valor'])
    # st.dataframe(df)

    # upper_est = new_fit.percentiles['Upper Estimate'].values
    # lower_est = new_fit.percentiles['Lower Estimate'].values
    # y_teste = np.linspace(0.99,0.01, num=99)

    # dist.PDF()    
    # x_min,x_max = plt.gca().get_xlim()
    # x = np.linspace(x_min,x_max,points_quality)
    # y_PDF = dist.PDF(xvals=x)
    # y_CDF = dist.CDF(xvals=x)
    # y_SF = dist.SF(xvals=x)
    # y_HF = dist.HF(xvals=x)
    # y_HF_up = fit_up.HF(xvals=x)
    # y_HF_lw = fit_lw.HF(xvals=x)
    # y_CHF = dist.CHF(xvals=x)
    # x_f, y_f = plotting_positions(failures=ftime, right_censored=ctime)
    # fig = go.Figure()
    # fig.add_trace(go.Histogram(x=ftime, histnorm='probability density', name = 'Dados originais (falha)', marker=dict(color = 'rgba(0, 255, 0, 0.9)'), opacity=0.3))
    # fig.add_trace(go.Scatter(x=x, y=y_PDF, mode='lines', name = 'PDF',  marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True))
    # fig.add_trace(go.Scatter(x=x_f, y=y_f, mode='markers', name = 'Failure data',  marker=dict(color = 'rgba(255, 0, 0, 0.9)'), visible = 'legendonly'))
    # fig.add_trace(go.Scatter(x=x, y=y_CDF, mode='lines', name = 'CDF',  marker=dict(color = 'rgba(255, 0, 0, 0.9)'), visible = 'legendonly'))
    # fig.add_trace(go.Scatter(x=x, y=y_SF, mode='lines', name = 'SF',  marker=dict(color = 'rgba(255, 223, 118, 0.9)'), visible = 'legendonly'))
    
    # fig.add_trace(go.Scatter(name='SF Upper Bound', x=upper_est, y=y_teste[0:len(upper_est)+1], mode='lines', marker=dict(color="#444"), line=dict(width=0), showlegend=False, visible = 'legendonly'))
    # fig.add_trace(go.Scatter(name='SF Lower Bound',x=lower_est, y=y_teste[0:len(lower_est)+1], marker=dict(color="#444"), line=dict(width=0),	mode='lines', fillcolor='rgba(255, 20, 147, 0.2)', showlegend=False, visible = 'legendonly'))
    # fig.add_trace(go.Scatter(name='SF CI', x=upper_est, y=y_teste[0:len(upper_est)+1], marker=dict(color="#444"), line=dict(width=0), mode='lines', fillcolor='rgba(255, 255, 0, 0.2)', fill='tonexty', visible='legendonly'))
    # fig.add_trace(go.Scatter(x=x, y=y_HF, mode='lines', name = 'HF',  marker=dict(color = 'rgba(0, 0, 255, 0.9)'), visible = 'legendonly'))
    # fig.add_trace(go.Scatter(x=x, y=y_CHF, mode='lines', name = 'CHF',  marker=dict(color = 'rgba(135, 45, 54, 0.9)'), visible = 'legendonly'))
    
    

    # if show_variable:
    #     if distribution_name =="Normal Distribution":
    #         fig.add_vline(x=dist.mean, line_dash="dash", annotation_text="Mean, Median, Mode", annotation_position="top right")
    #     elif distribution_name =="Beta Distribution" and (var1 <=1 or var2 <=1):
    #         fig.add_vline(x=dist.mean, line_dash="dash", annotation_text="Mean", annotation_position="top right")
    #         fig.add_vline(x=dist.median, line_dash="dash", annotation_text="Median", annotation_position="top right")
    #     else:
    #         fig.add_vline(x=dist.mean, line_dash="dash", annotation_text="Mean", annotation_position="top right")
    #         fig.add_vline(x=dist.median, line_dash="dash", annotation_text="Median", annotation_position="top right")
    #         fig.add_vline(x=dist.mode, line_dash="dash", annotation_text="Mode", annotation_position="top right")
    # fig.update_layout(width = 1900, height = 600, title = 'Dados analisados', yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log,title_text='Parametric Model - {} ({}) '.format(distribution_name,dist.param_title)) #size of figure
    # fig.update_xaxes(title = 'Time')
    # fig.update_yaxes(title = 'Probability density')			
    # st.plotly_chart(fig)






    # # fig.add_trace(go.Histogram(x=data_fai, histnorm='probability density', name = 'Dados originais (falha)', marker_color='red', opacity=0.3))
