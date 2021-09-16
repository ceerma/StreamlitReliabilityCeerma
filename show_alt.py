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
from reliability.ALT_fitters import Fit_Weibull_Exponential, Fit_Weibull_Eyring, Fit_Weibull_Power, \
Fit_Weibull_Dual_Exponential, Fit_Weibull_Power_Exponential, Fit_Weibull_Dual_Power, Fit_Lognormal_Exponential, Fit_Lognormal_Eyring, \
Fit_Lognormal_Power, Fit_Lognormal_Dual_Exponential, Fit_Lognormal_Power_Exponential, Fit_Lognormal_Dual_Power, Fit_Normal_Exponential, \
Fit_Normal_Eyring, Fit_Normal_Power, Fit_Normal_Dual_Exponential, Fit_Normal_Dual_Power, Fit_Exponential_Exponential, \
Fit_Exponential_Eyring, Fit_Exponential_Power, Fit_Exponential_Dual_Exponential, Fit_Exponential_Power_Exponential, Fit_Exponential_Dual_Power
import pickle

#st.set_page_config(page_title="Accelerated Life Testing",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

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

    def st_tonumlist(txt_input):
        txt_input = txt_input.rsplit(sep=",")
        num_list =[]
        for i in txt_input:
            try:
                num_list.append(float(i))
            except:
                pass
        return num_list

    single_stress_ALT_models_list = [
        "Weibull_Exponential",
        "Weibull_Eyring",
        "Weibull_Power",
        "Lognormal_Exponential",
        "Lognormal_Eyring",
        "Lognormal_Power",
        "Normal_Exponential",
        "Normal_Eyring",
        "Normal_Power",
        "Exponential_Exponential",
        "Exponential_Eyring",
        "Exponential_Power",
    ]


    dual_stress_ALT_models_list = [
        "Weibull_Dual_Exponential",
        "Weibull_Power_Exponential",
        "Weibull_Dual_Power",
        "Lognormal_Dual_Exponential",
        "Lognormal_Power_Exponential",
        "Lognormal_Dual_Power",
        "Normal_Dual_Exponential",
        "Normal_Power_Exponential",
        "Normal_Dual_Power",
        "Exponential_Dual_Exponential",
        "Exponential_Power_Exponential",
        "Exponential_Dual_Power",
    ]



    st.title("Accelerated Life Testing")
    st.write("In this module, you can provide your Accelerated Life Testing (ALT) data (complete or incomplete) and fit the most common probability distributions in reliability ")

    with st.expander(label='Help'):
        st.write('When using this module, please take into consideration the following points:')
        st.write('- There is no need to sort the data in any particular order as this is all done automatically;')
        st.write('- For each stress level, there must be at least one failure data;')

    expander = st.expander("Data format")
    expander.info('Upload an excel file thar contains the following columns: failure or right-censored time ("Time"), \
        the time type, if failure or right censored ("Type"), and the stress level (only "Stress1" or also "Stress2" for dual stress models).')
    df_show = {'Time': [10, 15, 8, 20, 21, 12, 13, 30, 5], \
        'Type': ['F', 'F', 'C', 'F', 'C', 'C', 'F', 'F', 'C'], \
        'Stress1': [20, 20, 20, 40, 40, 40, 60, 60, 60],
        'Stress2': [100, 100, 100, 200, 200, 200, 300, 300, 300],}
    df_show = pd.DataFrame.from_dict(df_show)
    expander.write(df_show, width=50)
    expander.info('The use level stress parameter is optional. If single stress model, enter only one value. For example:')
    expander.write('10')
    expander.info('If dual stress model, enter two values separated by ",". For example: ')
    expander.write('10, 50')

    col2_1, col2_2 = st.columns(2)
    uploaded_file = col2_1.file_uploader("Upload a XLSX file", \
        type="xlsx", accept_multiple_files=False)

    dual = False

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        # sdf = df.style.format('{:10n}'.format)
        col2_2.dataframe(df)
        fdata = df[df['Type'] == 'F']
        ftime = np.array(fdata.iloc[:,0])
        fstress_1 = np.array(fdata.iloc[:,2])
        cdata = df[df['Type'] == 'C']
        ctime = np.array(cdata.iloc[:,0])
        cstress_1 = np.array(cdata.iloc[:,2])
        use_level = st.text_input("Use level stress (optional)")
        use_level = st_tonumlist(use_level)
        if len(df.columns) == 4:
            fstress_2 = np.array(fdata.iloc[:,3])
            cstress_2 = np.array(cdata.iloc[:,3])
            dual = True
            if use_level:
                if len(use_level) != 2:
                    st.error('Enter two use level stresses')
        elif len(df.columns) == 3:
            if use_level:
                if len(use_level) > 1:
                    st.error('Enter one use level stress')
                else:
                    use_level = use_level[0]        
        #     include = st.multiselect('Choose which distribution(s) you want to fit to your data', dual_stress_ALT_models_list)
        # else:
        #     include = st.multiselect('Choose which distribution(s) you want to fit to your data', single_stress_ALT_models_list)

    if dual == False:
        include = st.multiselect('Choose which distribution(s) you want to fit to your data', single_stress_ALT_models_list)
    else:
        include = st.multiselect('Choose which distribution(s) you want to fit to your data', dual_stress_ALT_models_list)

    method = st.radio('Choose the optimizer', ('TNC', 'L-BFGS-B'))
    st.info('The optimizers are all bound constrained methods. If the bound constrained method fails, nelder-mead will be used. If nelder-mead fails the initial guess (using least squares) will be returned.')
    metric = st.radio('Choose a goodness of fit criteria', ('BIC', 'AICc', 'Log-likelihood'))

    IC = 0.8
    print_results = False
    show_probability_plot = True
    show_life_stress_plot = True

    st.write(" ")

    if st.button("Fit ALT model"):
        
        if use_level == 0:
            use_level = None

        # plt.savefig('test.png')

        # st.write('## Results of all fitted ALT models')
        # st.write(results.results)

        # st.write('## Results of the best fitted ALT model')
        # distribution_name = results.best_model_name

        best_BIC, best_AICc, best_loglik = np.inf, np.inf, np.inf
        best_model = None
        best_model_name = None
        results = []

        if include:
            if dual == False:
                # create empty dataframe to append results
                results = pd.DataFrame(
                    columns=[
                        "ALT_model",
                        "a",
                        "b",
                        "c",
                        "n",
                        "beta",
                        "sigma",
                        "Log-likelihood",
                        "AICc",
                        "BIC",
                    ]
                )
                if 'Weibull_Exponential' in include:
                    res = Fit_Weibull_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Exponential",
                            "a": res.a,
                            "b": res.b,
                            "c": "",
                            "n": "",
                            "beta": res.beta,
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Exponential'

                if 'Weibull_Eyring' in include:
                    res = Fit_Weibull_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Eyring",
                            "a": res.a,
                            "b": "",
                            "c": res.c,
                            "n": "",
                            "beta": res.beta,
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Eyring'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Eyring'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Eyring'


                if 'Weibull_Power' in include:
                    res = Fit_Weibull_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Power",
                            "a": res.a,
                            "b": "",
                            "c": "",
                            "n": res.n,
                            "beta": res.beta,
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Power'

                if 'Lognormal_Exponential' in include:
                    res = Fit_Lognormal_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Exponential",
                            "a": res.a,
                            "b": res.b,
                            "c": "",
                            "n": "",
                            "beta": "",
                            "sigma": res.sigma,
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Exponential'

                if 'Lognormal_Eyring' in include:
                    res_Lognormal_Eyring = Fit_Lognormal_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Eyring",
                            "a": res_Lognormal_Eyring.a,
                            "b": "",
                            "c": res_Lognormal_Eyring.c,
                            "n": "",
                            "beta": "",
                            "sigma": res_Lognormal_Eyring.sigma,
                            "Log-likelihood": res_Lognormal_Eyring.loglik,
                            "AICc": res_Lognormal_Eyring.AICc,
                            "BIC": res_Lognormal_Eyring.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Lognormal_Eyring.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Lognormal_Eyring.BIC
                        best_model = res_Lognormal_Eyring
                        best_model_name = 'Lognormal_Eyring'
                    if res_Lognormal_Eyring.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Lognormal_Eyring.AICc
                        best_model = res_Lognormal_Eyring
                        best_model_name = 'Lognormal_Eyring'
                    if -res_Lognormal_Eyring.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Lognormal_Eyring.loglik
                        best_model = res_Lognormal_Eyring
                        best_model_name = 'Lognormal_Eyring'

                if 'Lognormal_Power' in include:
                    res_Lognormal_Power = Fit_Lognormal_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Power",
                            "a": res_Lognormal_Power.a,
                            "b": "",
                            "c": "",
                            "n": res_Lognormal_Power.n,
                            "beta": "",
                            "sigma": res_Lognormal_Power.sigma,
                            "Log-likelihood": res_Lognormal_Power.loglik,
                            "AICc": res_Lognormal_Power.AICc,
                            "BIC": res_Lognormal_Power.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Lognormal_Power.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Lognormal_Power.BIC
                        best_model = res_Lognormal_Power
                        best_model_name = 'Lognormal_Power'
                    if res_Lognormal_Power.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Lognormal_Power.AICc
                        best_model = res_Lognormal_Power
                        best_model_name = 'Lognormal_Power'
                    if -res_Lognormal_Power.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Lognormal_Power.loglik
                        best_model = res_Lognormal_Power
                        best_model_name = 'Lognormal_Power'

                if 'Normal_Exponential' in include:
                    res_Normal_Exponential = Fit_Normal_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Exponential",
                            "a": res_Normal_Exponential.a,
                            "b": res_Normal_Exponential.b,
                            "c": "",
                            "n": "",
                            "beta": "",
                            "sigma": res_Normal_Exponential.sigma,
                            "Log-likelihood": res_Normal_Exponential.loglik,
                            "AICc": res_Normal_Exponential.AICc,
                            "BIC": res_Normal_Exponential.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Normal_Exponential.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Normal_Exponential.BIC
                        best_model = res_Normal_Exponential
                        best_model_name = 'Normal_Exponential'
                    if res_Normal_Exponential.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Normal_Exponential.AICc
                        best_model = res_Normal_Exponential
                        best_model_name = 'Normal_Exponential'
                    if -res_Normal_Exponential.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Normal_Exponential.loglik
                        best_model = res_Normal_Exponential
                        best_model_name = 'Normal_Exponential'

                if 'Normal_Eyring' in include:
                    res_Normal_Eyring = Fit_Normal_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Eyring",
                            "a": res_Normal_Eyring.a,
                            "b": "",
                            "c": res_Normal_Eyring.c,
                            "n": "",
                            "beta": "",
                            "sigma": res_Normal_Eyring.sigma,
                            "Log-likelihood": res_Normal_Eyring.loglik,
                            "AICc": res_Normal_Eyring.AICc,
                            "BIC": res_Normal_Eyring.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Normal_Eyring.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Normal_Eyring.BIC
                        best_model = res_Normal_Eyring
                        best_model_name = 'Normal_Eyring'
                    if res_Normal_Eyring.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Normal_Eyring.AICc
                        best_model = res_Normal_Eyring
                        best_model_name = 'Normal_Eyring'
                    if -res_Normal_Eyring.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Normal_Eyring.loglik
                        best_model = res_Normal_Eyring
                        best_model_name = 'Normal_Eyring'

                if 'Normal_Power' in include:
                    res_Normal_Power = Fit_Normal_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Power",
                            "a": res_Normal_Power.a,
                            "b": "",
                            "c": "",
                            "n": res_Normal_Power.n,
                            "beta": "",
                            "sigma": res_Normal_Power.sigma,
                            "Log-likelihood": res_Normal_Power.loglik,
                            "AICc": res_Normal_Power.AICc,
                            "BIC": res_Normal_Power.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Normal_Power.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Normal_Power.BIC
                        best_model = res_Normal_Power
                        best_model_name = 'Normal_Power'
                    if res_Normal_Power.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Normal_Power.AICc
                        best_model = res_Normal_Power
                        best_model_name = 'Normal_Power'
                    if -res_Normal_Power.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Normal_Power.loglik
                        best_model = res_Normal_Power
                        best_model_name = 'Normal_Power'

                if 'Exponential_Exponential' in include:
                    res_Exponential_Exponential = Fit_Exponential_Exponential(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Exponential",
                            "a": res_Exponential_Exponential.a,
                            "b": res_Exponential_Exponential.b,
                            "c": "",
                            "n": "",
                            "beta": "",
                            "sigma": "",
                            "Log-likelihood": res_Exponential_Exponential.loglik,
                            "AICc": res_Exponential_Exponential.AICc,
                            "BIC": res_Exponential_Exponential.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Exponential_Exponential.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Exponential_Exponential.BIC
                        best_model = res_Exponential_Exponential
                        best_model_name = 'Exponential_Exponential'
                    if res_Exponential_Exponential.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Exponential_Exponential.AICc
                        best_model = res_Exponential_Exponential
                        best_model_name = 'Exponential_Exponential'
                    if -res_Exponential_Exponential.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Exponential_Exponential.loglik
                        best_model = res_Exponential_Exponential
                        best_model_name = 'Exponential_Exponential'

                if 'Exponential_Eyring' in include:
                    res_Exponential_Eyring = Fit_Exponential_Eyring(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Eyring",
                            "a": res_Exponential_Eyring.a,
                            "b": "",
                            "c": res_Exponential_Eyring.c,
                            "n": "",
                            "beta": "",
                            "sigma": "",
                            "Log-likelihood": res_Exponential_Eyring.loglik,
                            "AICc": res_Exponential_Eyring.AICc,
                            "BIC": res_Exponential_Eyring.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Exponential_Eyring.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Exponential_Eyring.BIC
                        best_model = res_Exponential_Eyring
                        best_model_name = 'Exponential_Eyring'
                    if res_Exponential_Eyring.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Exponential_Eyring.AICc
                        best_model = res_Exponential_Eyring
                        best_model_name = 'Exponential_Eyring'
                    if -res_Exponential_Eyring.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Exponential_Eyring.loglik
                        best_model = res_Exponential_Eyring
                        best_model_name = 'Exponential_Eyring'

                if 'Exponential_Power' in include:
                    res_Exponential_Power = Fit_Exponential_Power(failures=ftime, failure_stress=fstress_1, right_censored=ctime, \
                    right_censored_stress=cstress_1, use_level_stress=use_level, print_results=print_results, \
                    show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Power",
                            "a": res_Exponential_Power.a,
                            "b": "",
                            "c": "",
                            "n": res_Exponential_Power.n,
                            "beta": "",
                            "sigma": "",
                            "Log-likelihood": res_Exponential_Power.loglik,
                            "AICc": res_Exponential_Power.AICc,
                            "BIC": res_Exponential_Power.BIC,
                        },
                        ignore_index=True,
                    )
                    if res_Exponential_Power.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res_Exponential_Power.BIC
                        best_model = res_Exponential_Power
                        best_model_name = 'Exponential_Power'
                    if res_Exponential_Power.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res_Exponential_Power.AICc
                        best_model = res_Exponential_Power
                        best_model_name = 'Exponential_Power'
                    if -res_Exponential_Power.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res_Exponential_Power.loglik
                        best_model = res_Exponential_Power
                        best_model_name = 'Exponential_Power'

                # # Recreate plt figure in plotly
                # lines = best_model.probability_plot

                # fig = go.Figure()
                # x_min,x_max = lines.get_xlim()
                # y_min,y_max = lines.get_ylim()

                # for line in lines.get_lines():

                #     if line.get_linestyle() == '--':
                #         dash = 'dash'
                #     else:
                #         dash = None

                #     if line.get_label() in list(set(map(str, fstress_1))):
                #         label = line.get_label()
                #     else:
                #         label = label

                #     fig.add_trace(go.Scatter(x=line.get_xdata(), y=line.get_ydata(), name = label,  line=dict(color = line.get_color(), dash=dash), visible = True))

                # fig.update_xaxes(range=[x_min, x_max])
                # fig.update_yaxes(range=[y_min, y_max])
                # fig.update_layout(width = 600, height = 600, title = 'Probability plot', yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log, title_text='Probability plot') #- {} - a = {}, b = {}, beta = {}'.format('Weibull Exponential', results.a, results.b, results.beta))
                # st.plotly_chart(fig)

            else:
                # create empty dataframe to append results
                results = pd.DataFrame(
                    columns=[
                        "ALT_model",
                        "a",
                        "b",
                        "c",
                        "m",
                        "n",
                        "beta",
                        "sigma",
                        "Log-likelihood",
                        "AICc",
                        "BIC",
                    ]
                )

                if 'Weibull_Dual_Exponential' in include:
                    res = Fit_Weibull_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Dual_Exponential",
                            "a": res.a,
                            "b": res.b,
                            "c": res.c,
                            "m": "",
                            "n": "",
                            "beta": res.beta,
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Dual_Exponential'

                if 'Weibull_Power_Exponential' in include:
                    res = Fit_Weibull_Power_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Power_Exponential",
                            "a": res.a,
                            "b": "",
                            "c": res.c,
                            "m": "",
                            "n": res.n,
                            "beta": res.beta,
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Power_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Power_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Power_Exponential'

                if 'Weibull_Dual_Power' in include:
                    res = Fit_Weibull_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Weibull_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": res.c,
                            "m": res.m,
                            "n": res.n,
                            "beta": res.beta,
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Weibull_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Weibull_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Weibull_Dual_Power'

                if 'Lognormal_Dual_Exponential' in include:
                    res = Fit_Lognormal_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Dual_Exponential",
                            "a": res.a,
                            "b": res.b,
                            "c": res.c,
                            "m": "",
                            "n": "",
                            "beta": "",
                            "sigma": res.sigma,
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Exponential'
                        
                if 'Lognormal_Power_Exponential' in include:
                    res = Fit_Lognormal_Power_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Power_Exponential",
                            "a": res.a,
                            "b": "",
                            "c": res.c,
                            "m": "",
                            "n": res.n,
                            "beta": "",
                            "sigma": res.sigma,
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Power_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Power_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Power_Exponential'

                if 'Lognormal_Dual_Power' in include:
                    res = Fit_Lognormal_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Lognormal_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": res.c,
                            "m": res.m,
                            "n": res.n,
                            "beta": "",
                            "sigma": res.sigma,
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Lognormal_Dual_Power'

                if 'Normal_Dual_Exponential' in include:
                    res = Fit_Normal_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Dual_Exponential",
                            "a": res.a,
                            "b": res.b,
                            "c": res.c,
                            "m": "",
                            "n": "",
                            "beta": "",
                            "sigma": res.sigma,
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Dual_Exponential'
                        
                if 'Normal_Dual_Power' in include:
                    res = Fit_Normal_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Normal_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": res.c,
                            "m": res.m,
                            "n": res.n,
                            "beta": "",
                            "sigma": res.sigma,
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Normal_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Normal_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Normal_Dual_Power'

                if 'Exponential_Dual_Exponential' in include:
                    res = Fit_Exponential_Dual_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Dual_Exponential",
                            "a": res.a,
                            "b": res.b,
                            "c": res.c,
                            "m": "",
                            "n": "",
                            "beta": "",
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Dual_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Dual_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Dual_Exponential'

                if 'Exponential_Power_Exponential' in include:
                    res = Fit_Exponential_Power_Exponential(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Power_Exponential",
                            "a": res.a,
                            "b": "",
                            "c": res.c,
                            "m": "",
                            "n": res.n,
                            "beta": "",
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Power_Exponential'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Power_Exponential'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Power_Exponential'

                if 'Exponential_Dual_Power' in include:
                    res = Fit_Exponential_Dual_Power(failures=ftime, failure_stress_1=fstress_1, failure_stress_2=fstress_2, \
                    right_censored=ctime, right_censored_stress_1=cstress_1, right_censored_stress_2=cstress_2, use_level_stress=use_level, \
                    print_results=print_results, show_probability_plot=show_probability_plot, show_life_stress_plot=show_life_stress_plot, \
                    CI=IC, optimizer=method)
                    results = results.append(
                        {
                            "ALT_model": "Exponential_Dual_Power",
                            "a": "",
                            "b": "",
                            "c": res.c,
                            "m": res.m,
                            "n": res.n,
                            "beta": "",
                            "sigma": "",
                            "Log-likelihood": res.loglik,
                            "AICc": res.AICc,
                            "BIC": res.BIC,
                        },
                        ignore_index=True,
                    )
                    if res.BIC < best_BIC and metric == 'BIC':
                        best_BIC = res.BIC
                        best_model = res
                        best_model_name = 'Exponential_Dual_Power'
                    if res.AICc < best_AICc and metric == 'AICc':
                        best_AICc = res.AICc
                        best_model = res
                        best_model_name = 'Exponential_Dual_Power'
                    if -res.loglik < best_loglik and metric == 'Log-likelihood':
                        best_loglik = -res.loglik
                        best_model = res
                        best_model_name = 'Exponential_Dual_Power'

            st.write('## Results of all fitted ALT models')
            # results = pd.DataFrame.from_dict(results)
            st.write(results)

            st.write('## Results of the best fitted ALT model')
            st.write(best_model_name)
            st.write(best_model.results)

            probability_plot = best_model.probability_plot.figure
            life_stress_plot = best_model.life_stress_plot.figure
            col1, col2 = st.columns(2)
            col1.pyplot(probability_plot)
            col2.pyplot(life_stress_plot)

        else:
            st.error('Please, choose at least one model to fit.')
            st.stop()