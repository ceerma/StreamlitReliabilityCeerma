import streamlit as st
import reliability 
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy import integrate
from matplotlib.axes import SubplotBase
from reliability.Utils import colorprint, round_to_decimals
from scipy.optimize import curve_fit

class reliability_growth:
    """
    Uses the Duane method to find the instantaneous MTBF and produce a
    reliability growth plot.
    Parameters
    ----------
    times : list, array
        The failure times.
    xmax : int, float, optional
        The xlim to plot up to. Default is 1.5*max(times)
    target_MTBF : int, float, optional
        Specify the target MTBF to obtain the total time on test required to
        reach it. Default = None
    show_plot : bool, optional
        If True the plot will be produced. Default is True.
    print_results : bool, optional
        If True the results will be printed to console. Default = True.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).
    Returns
    -------
    Lambda : float
        The lambda parameter from the Duane model
    Beta : float
        The beta parameter from the Duane model
    time_to_target : float
        The time to reach the target is only returned if target_MTBF is
        specified.
    Notes
    -----
    If show_plot is True, the reliability growth plot will be produced. Use
    plt.show() to show the plot.
    """

    def __init__(
        self,
        times=None,
        xmax=None,
        target_MTBF=None,
        show_plot=True,
        print_results=True,
        **kwargs
    ):
        if times is None:
            raise ValueError("times must be an array or list of failure times")
        if type(times) == list:
            times = np.sort(np.array(times))
        elif type(times) == np.ndarray:
            times = np.sort(times)
        else:
            raise ValueError("times must be an array or list of failure times")
        if min(times) < 0:
            raise ValueError(
                "failure times cannot be negative. times must be an array or list of failure times"
            )
        if xmax is None:
            xmax = int(max(times) * 1.5)
        if "color" in kwargs:
            c = kwargs.pop("color")
        else:
            c = "steelblue"

        N = np.arange(1, len(times) + 1)
        theta_c = times / N
        ln_t = np.log(times)
        ln_theta_c = np.log(theta_c)
        z = np.polyfit(
            ln_t, ln_theta_c, 1
        )  # fit a straight line to the data to get the parameters lambda and beta
        beta = 1 - z[0]
        Lambda = np.exp(-z[1])
        xvals = np.linspace(0, xmax, 1000)
        theta_i = (xvals ** (1 - beta)) / (Lambda * beta)  # the smooth line
        theta_i_points = (times ** (1 - beta)) / (
            Lambda * beta
        )  # the failure times highlighted along the line
        self.Lambda = Lambda
        self.Beta = beta

        if print_results is True:
            st.write("## Reliability growth model parameters")
            st.write("lambda:", round(Lambda,4))
            st.write("beta:", round(beta,4))


        if target_MTBF is not None:
            t_target = (target_MTBF * Lambda * beta) ** (1 / (1 - beta))
            self.time_to_target = t_target
            print("Time to reach target MTBF:", t_target)
        else:
            self.time_to_target = "specify a target to obtain the time_to_target"

        if show_plot is True:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xvals, y=theta_i, mode='lines', name = 'Reliability Growth curve',  marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True))
            fig.add_trace(go.Scatter(x=times, y=theta_i_points, mode='markers', marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True, showlegend=False))
            if target_MTBF is not None:
                fig.add_trace(go.Scatter(x=[0, t_target, t_target], y=[target_MTBF, target_MTBF, 0], mode='lines', name = "Reliability target",  marker=dict(color = 'rgba(255, 0, 0, 0.9)'), visible = True))

            fig.update_layout(width = 1900, height = 600, yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log) #size of figure
            fig.update_xaxes(title = "Total time on test")
            fig.update_yaxes(title = "Instantaneous MTBF")	
            fig.update_xaxes(range=[0, max(xvals)])
            fig.update_yaxes(range=[0, max(theta_i) * 1.2])			
            st.plotly_chart(fig)


class ROCOF:
    """
    Uses the failure times or failure interarrival times to determine if there
    is a trend in those times. The test for statistical significance is the
    Laplace test which compares the Laplace test statistic (U) with the z value
    (z_crit) from the standard normal distribution. If there is a statistically
    significant trend, the parameters of the model (Lambda_hat and Beta_hat) are
    calculated. By default the results are printed and a plot of the times and
    MTBF is plotted.
    Parameters
    ----------
    times_between_failures : array, list, optional
        The failure interarrival times. See the Notes below.
    failure_times : array, list, optional
        The actual failure times. See the Notes below.
    test_end : int, float, optional
        Use this to specify the end of the test if the test did not end at the
        time of the last failure. Default = None which will result in the last
        failure being treated as the end of the test.
    CI : float
        The confidence interval for the Laplace test. Must be between 0 and 1.
        Default is 0.95 for 95% CI.
    show_plot : bool, optional
        If True the plot will be produced. Default = True.
    print_results : bool, optional
        If True the results will be printed to console. Default = True.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).
    Returns
    -------
    U : float
        The Laplace test statistic
    z_crit : tuple
        (lower,upper) bound on z value. This is based on the CI.
    trend : str
        'improving','worsening','constant'. This is based on the comparison of U
        with z_crit
    Beta_hat : float, str
        The Beta parameter for the NHPP Power Law model. Only calculated if the
        trend is not constant, else a string is returned.
    Lambda_hat : float, str
        The Lambda parameter for the NHPP Power Law model. Only calculated if
        the trend is not constant.
    ROCOF : float, str
        The Rate of OCcurrence Of Failures. Only calculated if the trend is
        constant. If trend is not constant then ROCOF changes over time in
        accordance with Beta_hat and Lambda_hat. In this case a string will be
        returned.
    Notes
    -----
    You can specify either times_between_failures OR failure_times but not both.
    Both options are provided for convenience so the conversion between the two
    is done internally. failure_times should be the same as
    np.cumsum(times_between_failures).
    The repair time is assumed to be negligible. If the repair times are not
    negligibly small then you will need to manually adjust your input to factor
    in the repair times.
    If show_plot is True, the ROCOF plot will be produced. Use plt.show() to
    show the plot.
    """

    def __init__(
        self,
        times_between_failures=None,
        failure_times=None,
        CI=0.95,
        test_end=None,
        show_plot=True,
        print_results=True,
        **kwargs
    ):
        if times_between_failures is not None and failure_times is not None:
            raise ValueError(
                "You have specified both times_between_failures and failure times. You can specify one but not both. Use times_between_failures for failure interarrival times, and failure_times for the actual failure times. failure_times should be the same as np.cumsum(times_between_failures)"
            )
        if times_between_failures is not None:
            if any(t <= 0 for t in times_between_failures):
                raise ValueError("times_between_failures cannot be less than zero")
            if type(times_between_failures) == list:
                ti = times_between_failures
            elif type(times_between_failures) == np.ndarray:
                ti = list(times_between_failures)
            else:
                raise ValueError("times_between_failures must be a list or array")
        if failure_times is not None:
            if any(t <= 0 for t in failure_times):
                raise ValueError("failure_times cannot be less than zero")
            if type(failure_times) == list:
                failure_times = np.sort(np.array(failure_times))
            elif type(failure_times) == np.ndarray:
                failure_times = np.sort(failure_times)
            else:
                raise ValueError("failure_times must be a list or array")
            failure_times[1:] -= failure_times[
                :-1
            ].copy()  # this is the opposite of np.cumsum
            ti = list(failure_times)
        if test_end is not None and type(test_end) not in [float, int]:
            raise ValueError(
                "test_end should be a float or int. Use test_end to specify the end time of a test which was not failure terminated."
            )
        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% confidence interval."
            )
        if test_end is None:
            tn = sum(ti)
            n = len(ti) - 1
        else:
            tn = test_end
            n = len(ti)
            if tn < sum(ti):
                raise ValueError("test_end cannot be less than the final test time")

        if "linestyle" in kwargs:
            ls = kwargs.pop("linestyle")
        else:
            ls = "--"
        if "label" in kwargs:
            label_1 = kwargs.pop("label")
        else:
            label_1 = "Failure interarrival times"

        tc = np.cumsum(ti[0:n])
        sum_tc = sum(tc)
        z_crit = ss.norm.ppf((1 - CI) / 2)  # z statistic based on CI
        U = (sum_tc / n - tn / 2) / (tn * (1 / (12 * n)) ** 0.5)
        self.U = U
        self.z_crit = (z_crit, -z_crit)
        # results_str = str(
        #     "Laplace test results: U = "
        #     + str(round(U, 3))
        #     + ", z_crit = ("
        #     + str(round(z_crit, 2))
        #     + ",+"
        #     + str(round(-z_crit, 2))
        #     + ")"
        # )

        x = np.arange(1, len(ti) + 1)
        if U < z_crit:
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn ** B)
            self.trend = "improving"
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            if test_end is not None:
                x_to_plot = x
            else:
                x_to_plot = x[:-1]
        elif U > -z_crit:
            B = len(ti) / (sum(np.log(tn / np.array(tc))))
            L = len(ti) / (tn ** B)
            self.trend = "worsening"
            self.Beta_hat = B
            self.Lambda_hat = L
            self.ROCOF = "ROCOF is not provided when trend is not constant. Use Beta_hat and Lambda_hat to calculate ROCOF at a given time t."
            _rocof = L * B * tc ** (B - 1)
            MTBF = np.ones_like(tc) / _rocof
            if test_end is not None:
                x_to_plot = x
            else:
                x_to_plot = x[:-1]
        else:
            rocof = (n + 1) / sum(ti)
            self.trend = "constant"
            self.ROCOF = rocof
            self.Beta_hat = "not calculated when trend is constant"
            self.Lambda_hat = "not calculated when trend is constant"
            x_to_plot = x
            MTBF = np.ones_like(x_to_plot) / rocof

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)

        if print_results is True:
            st.write("## Results from ROCOF analysis")
            st.write(
            "Laplace test results: U = ",
            round(U, 3),
            ", z_crit = (",
            round(z_crit, 2),
            ",",
            round(-z_crit, 2),
            ")"
        )
            if U < z_crit:
                st.write(
                    str(
                        "At "
                        + str(CI_rounded)
                        + "% confidence level the ROCOF is **IMPROVING**."
                    )
                )
                st.write(
                    "ROCOF assuming NHPP has parameters: Beta_hat =",
                    round_to_decimals(B, 3),
                    ", Lambda_hat =",
                    round_to_decimals(L, 4),
                )
            elif U > -z_crit:
                st.write(
                    str(
                        "At "
                        + str(CI_rounded)
                        + "% confidence level the ROCOF is **WORSENING**."
                    )
                )
                st.write(
                    "ROCOF assuming NHPP has parameters: Beta_hat =",
                    round_to_decimals(B, 3),
                    ", Lambda_hat =",
                    round_to_decimals(L, 4),
                )
            else:
                st.write(
                    str(
                        "At "
                        + str(CI_rounded)
                        + "% confidence level the ROCOF is **CONSTANT**."
                    )
                )
                st.write(
                    "ROCOF assuming HPP is",
                    round_to_decimals(rocof, 4),
                    "failures per unit time.",
                )

        if show_plot is True:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_to_plot, y=MTBF, mode='lines', name = 'MTBF',  marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True))
            fig.add_trace(go.Scatter(x=x, y=ti, mode='markers', name=label_1, marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True))
            fig.update_layout(width = 1900, height = 600, yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log) #size of figure
            fig.update_xaxes(title = "Failure number")
            fig.update_yaxes(title = "Times between failures")			
            st.plotly_chart(fig)

class optimal_replacement_time:
    """
    Calculates the cost model to determine how cost varies with replacement time.
    The cost model may be HPP (good as new replacement) or NHPP (as good as old
    replacement). Default is HPP.
    Parameters
    ----------
    Cost_PM : int, float
        The cost of preventative maintenance (must be smaller than Cost_CM)
    Cost_CM : int, float
        The cost of corrective maintenance (must be larger than Cost_PM)
    weibull_alpha : int, float
        The scale parameter of the underlying Weibull distribution.
    weibull_beta : int, float
        The shape parameter of the underlying Weibull distribution. Should be
        greater than 1 otherwise conducting PM is not economical.
    q : int, optional
        The restoration factor. Must be 0 or 1. Use q=1 for Power Law NHPP
        (as good as old) or q=0 for HPP (as good as new). Default is q=0 (as
        good as new).
    show_time_plot : bool, optional
        If True the plot of replacment time vs cost per unit time will be
        produced in a new figure. If False then no plot will be generated.
        Default is True.
    show_ratio_plot : bool, optional
        If True the plot of cost ratio vs replacement interval will be
        produced in a new figure. If False then no plot will be generated.
        Default is True.
    print_results : bool, optional
        If True the results will be printed to console. Default = True.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).
    Returns
    -------
    ORT : float
        The optimal replacement time
    min_cost : float
        The minimum cost per unit time
    """

    def __init__(
        self,
        cost_PM,
        cost_CM,
        weibull_alpha,
        weibull_beta,
        show_time_plot=True,
        show_ratio_plot=True,
        print_results=True,
        q=0,
        **kwargs
    ):
        if "color" in kwargs:
            c = kwargs.pop("color")
        else:
            c = "steelblue"
        if cost_PM > cost_CM:
            st.error(
                "Preventative Maintenance Cost must be less than Corrective Maintenance Cost otherwise preventative maintenance should not be conducted."
            )
        if weibull_beta < 1:
            st.warning(
                "Shape Parameter of the Weibull Distribution is < 1 so the hazard rate is decreasing, therefore preventative maintenance should not be conducted."
            )
            st.stop()

        if q == 1:  # as good as old
            alpha_multiple = 4
            t = np.linspace(1, weibull_alpha * alpha_multiple, 100000)
            CPUT = ((cost_PM * (t / weibull_alpha) ** weibull_beta) + cost_CM) / t
            ORT = weibull_alpha * (
                (cost_CM / (cost_PM * (weibull_beta - 1))) ** (1 / weibull_beta)
            )
            min_cost = (
                (cost_PM * (ORT / weibull_alpha) ** weibull_beta) + cost_CM
            ) / ORT
        elif q == 0:  # as good as new
            alpha_multiple = 3
            t = np.linspace(1, weibull_alpha * alpha_multiple, 10000)

            # survival function and its integral
            calc_SF = lambda x: np.exp(-((x / weibull_alpha) ** weibull_beta))
            integrate_SF = lambda x: integrate.quad(calc_SF, 0, x)[0]

            # vectorize them
            vcalc_SF = np.vectorize(calc_SF)
            vintegrate_SF = np.vectorize(integrate_SF)

            # calculate the SF and intergral at each time
            sf = vcalc_SF(t)
            integral = vintegrate_SF(t)

            CPUT = (cost_PM * sf + cost_CM * (1 - sf)) / integral
            idx = np.argmin(CPUT)
            min_cost = CPUT[idx]  # minimum cost per unit time
            ORT = t[idx]  # optimal replacement time
        else:
            raise ValueError(
                'q must be 0 or 1. Default is 0. Use 0 for "as good as new" and use 1 for "as good as old".'
            )
        self.ORT = ORT
        self.min_cost = min_cost
        min_cost_rounded = round_to_decimals(min_cost, 2)
        ORT_rounded = round_to_decimals(ORT, 2)

        if print_results is True:
            st.write("## Results from the Optimal Replacement Time analysis")
            if q == 0:
                st.write("Cost model assuming as good as new replacement (q=0):")
            else:
                st.write("Cost model assuming as good as old replacement (q=1):")
            st.write(
                "The minimum cost per unit time is",
                min_cost_rounded,
                "\nThe optimal replacement time is",
                ORT_rounded,
            )

        if (
            show_time_plot is True
        ):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=CPUT, mode='lines',  marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True))
            fig.add_trace(go.Scatter(x=np.array(ORT), y=np.array(min_cost), mode='markers', marker=dict(color = 'rgba(255, 0, 0, 0.9)'), visible = True))
            fig.add_annotation(x=ORT, y=min_cost,
            text="Optimal replacement time",
            showarrow=True,
            arrowhead=1)
            fig.update_layout(width = 1900, height = 600, title="Optimal replacement time estimation", yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log, showlegend=False) #size of figure
            fig.update_xaxes(title = "Replacement time")
            fig.update_yaxes(title = "Cost per unit time")
            fig.update_xaxes(range=[0, weibull_alpha * alpha_multiple])
            fig.update_yaxes(range=[0, min_cost * 2])				
            st.plotly_chart(fig)

        if (
            show_ratio_plot is True
        ):
            xupper = np.round(cost_CM / cost_PM, 0) * 2
            CC_CP = np.linspace(1, xupper, 200)  # cost CM / cost PM
            CC = CC_CP * cost_PM
            ORT_array = []  # optimal replacement time

            # get the ORT from the minimum CPUT for each CC
            if q == 1:
                calc_ORT = lambda x: weibull_alpha * (
                    (x / (cost_PM * (weibull_beta - 1))) ** (1 / weibull_beta)
                )
            else:  # q = 0
                calc_ORT = lambda x: t[
                    np.argmin((cost_PM * sf + x * (1 - sf)) / integral)
                ]

            vcalc_ORT = np.vectorize(calc_ORT)
            ORT_array = vcalc_ORT(CC)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=CC_CP, y=ORT_array, mode='lines',  marker=dict(color = 'rgba(0, 255, 0, 0.9)'), visible = True))
            fig.add_trace(go.Scatter(x=np.array(cost_CM / cost_PM), y=np.array(self.ORT), mode='markers', marker=dict(color = 'rgba(255, 0, 0, 0.9)'), visible = True))
            fig.add_annotation(x=cost_CM / cost_PM, y=self.ORT,
            text="Optimal replacement time",
            showarrow=True,
            arrowhead=1)
            fig.update_layout(width = 1900, height = 600, title="Optimal replacement interval across a range of CM costs", yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log, showlegend=False) #size of figure
            fig.update_xaxes(title = "Cost ratio")
            fig.update_yaxes(title = "Replacement Interval")
            fig.update_xaxes(range=[1, xupper])
            fig.update_yaxes(range=[0, self.ORT * 2])			
            st.plotly_chart(fig)

class MCF_nonparametric:
    """
    The Mean Cumulative Function (MCF) is a cumulative history function that
    shows the cumulative number of recurrences of an event, such as repairs over
    time. In the context of repairs over time, the value of the MCF can be
    thought of as the average number of repairs that each system will have
    undergone after a certain time. It is only applicable to repairable systems
    and assumes that each event (repair) is identical, but it does not assume
    that each system's MCF is identical (which is an assumption of the
    parametric MCF). The non-parametric estimate of the MCF provides both the
    estimate of the MCF and the confidence bounds at a particular time.
    The shape of the MCF is a key indicator that shows whether the systems are
    improving, worsening, or staying the same over time. If the MCF is concave
    down (appearing to level out) then the system is improving. A straight line
    (constant increase) indicates it is staying the same. Concave up (getting
    steeper) shows the system is worsening as repairs are required more
    frequently as time progresses.
    Parameters
    ----------
    data : list
        The repair times for each system. Format this as a list of lists. eg.
        data=[[4,7,9],[3,8,12]] would be the data for 2 systems. The largest
        time for each system is assumed to be the retirement time and is treated
        as a right censored value. If the system was retired immediately after
        the last repair then you must include a repeated value at the end as
        this will be used to indicate a right censored value. eg. A system that
        had repairs at 4, 7, and 9 then was retired after the last repair would
        be entered as data = [4,7,9,9] since the last value is treated as a
        right censored value. If you only have data from 1 system you may enter
        the data in a single list as data = [3,7,12] and it will be nested
        within another list automatically.
    print_results : bool, optional
        Prints the table of MCF results (state, time, MCF_lower, MCF, MCF_upper,
        variance). Default = True.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default = 0.95 for 95% CI
        (one sided).
    show_plot : bool, optional
        If True the plot will be shown. Default = True. Use plt.show() to show
        it.
    plot_CI : bool, optional
        If True, the plot will include the confidence intervals. Default = True.
        Set as False to remove the confidence intervals from the plot.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).
    Returns
    -------
    results : dataframe
        This is a dataframe of the results that are printed. It includes the
        blank lines for censored values.
    time : array
        This is the time column from results. Blank lines for censored values
        are removed.
    MCF : array
        This is the MCF column from results. Blank lines for censored values are
        removed.
    variance : array
        This is the Variance column from results. Blank lines for censored
        values are removed.
    lower : array
        This is the MCF_lower column from results. Blank lines for censored
        values are removed.
    upper : array
        This is the MCF_upper column from results. Blank lines for censored
        values are removed
    Notes
    -----
    This example is taken from Reliasoft's example (available at
    http://reliawiki.org/index.php/Recurrent_Event_Data_Analysis). The failure
    times and retirement times (retirement time is indicated by +) of 5 systems
    are:
    +------------+--------------+
    | System     | Times        |
    +------------+--------------+
    | 1          | 5,10,15,17+  |
    +------------+--------------+
    | 2          | 6,13,17,19+  |
    +------------+--------------+
    | 3          | 12,20,25,26+ |
    +------------+--------------+
    | 4          | 13,15,24+    |
    +------------+--------------+
    | 5          | 16,22,25,28+ |
    +------------+--------------+
    .. code:: python
        from reliability.Repairable_systems import MCF_nonparametric
        times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
        MCF_nonparametric(data=times)
    """

    def __init__(
        self, data, CI=0.95, print_results=True, show_plot=True, plot_CI=True, **kwargs
    ):

        # check input is a list
        if type(data) == list:
            pass
        elif type(data) == np.ndarray:
            data = list(data)
        else:
            raise ValueError("data must be a list or numpy array")

        # check each item is a list and fix up any ndarrays to be lists.
        test_for_single_system = []
        for i, item in enumerate(data):
            if type(item) == list:
                test_for_single_system.append(False)
            elif type(item) == np.ndarray:
                data[i] = list(item)
                test_for_single_system.append(False)
            elif type(item) == int or type(item) == float:
                test_for_single_system.append(True)
            else:
                raise ValueError(
                    "Each item in the data must be a list or numpy array. eg. data = [[1,3,5],[3,6,8]]"
                )
        # Wraps the data in another list if all elements were numbers.
        if all(test_for_single_system):  # checks if all are True
            data = [data]
        elif not any(test_for_single_system):  # checks if all are False
            pass
        else:
            raise ValueError(
                "Mixed data types found in the data. Each item in the data must be a list or numpy array. eg. data = [[1,3,5],[3,6,8]]."
            )

        end_times = []
        repair_times = []
        for system in data:
            system.sort()  # sorts the values in ascending order
            for i, t in enumerate(system):
                if i < len(system) - 1:
                    repair_times.append(t)
                else:
                    end_times.append(t)

        if CI < 0 or CI > 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% confidence intervals (two sided)."
            )

        if max(end_times) < max(repair_times):
            raise ValueError(
                "The final end time must not be less than the final repair time."
            )
        last_time = max(end_times)
        C_array = ["C"] * len(end_times)
        F_array = ["F"] * len(repair_times)

        Z = -ss.norm.ppf(1 - CI)  # confidence interval converted to Z-value

        # sort the inputs and extract the sorted values for later use
        times = np.hstack([repair_times, end_times])
        states = np.hstack([F_array, C_array])
        data = {"times": times, "states": states}
        df = pd.DataFrame(data, columns=["times", "states"])
        df_sorted = df.sort_values(
            by=["times", "states"], ascending=[True, False]
        )  # sorts the df by times and then by states, ensuring that states are F then C where the same time occurs. This ensures a failure is counted then the item is retired.
        times_sorted = df_sorted.times.values
        states_sorted = df_sorted.states.values

        # MCF calculations
        MCF_array = []
        Var_array = []
        MCF_lower_array = []
        MCF_upper_array = []
        r = len(end_times)
        r_inv = 1 / r
        C_seq = 0  # sequential number of censored values
        for i in range(len(times)):
            if i == 0:
                if states_sorted[i] == "F":  # first event is a failure
                    MCF_array.append(r_inv)
                    Var_array.append(
                        (r_inv ** 2) * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2)
                    )
                    MCF_lower_array.append(
                        MCF_array[i] / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                    )
                    MCF_upper_array.append(
                        MCF_array[i] * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                    )
                else:  # first event is censored
                    MCF_array.append("")
                    Var_array.append("")
                    MCF_lower_array.append("")
                    MCF_upper_array.append("")
                    r -= 1
                    if (
                        times_sorted[i] not in end_times
                    ):  # check if this system only has one event. If not then increment the number censored count for this system
                        C_seq += 1
            else:  # everything after the first time
                if states_sorted[i] == "F":  # failure event
                    i_adj = i - C_seq
                    r_inv = 1 / r
                    if (
                        MCF_array[i_adj - 1] == ""
                    ):  # this is the case where the first system only has one event that was censored and there is no data on the first line
                        MCF_array.append(r_inv)
                        Var_array.append(
                            (r_inv ** 2)
                            * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2)
                        )
                        MCF_lower_array.append(
                            MCF_array[i]
                            / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                        MCF_upper_array.append(
                            MCF_array[i]
                            * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                    else:  # this the normal case where there was previous data
                        MCF_array.append(r_inv + MCF_array[i_adj - 1])
                        Var_array.append(
                            (r_inv ** 2)
                            * ((1 - r_inv) ** 2 + (r - 1) * (0 - r_inv) ** 2)
                            + Var_array[i_adj - 1]
                        )
                        MCF_lower_array.append(
                            MCF_array[i]
                            / np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                        MCF_upper_array.append(
                            MCF_array[i]
                            * np.exp((Z * Var_array[i] ** 0.5) / MCF_array[i])
                        )
                    C_seq = 0
                else:  # censored event
                    r -= 1
                    C_seq += 1
                    MCF_array.append("")
                    Var_array.append("")
                    MCF_lower_array.append("")
                    MCF_upper_array.append("")
                    if r > 0:
                        r_inv = 1 / r

        # format output as dataframe
        data = {
            "state": states_sorted,
            "time": times_sorted,
            "MCF_lower": MCF_lower_array,
            "MCF": MCF_array,
            "MCF_upper": MCF_upper_array,
            "variance": Var_array,
        }
        printable_results = pd.DataFrame(
            data, columns=["state", "time", "MCF_lower", "MCF", "MCF_upper", "variance"]
        )

        indices_to_drop = printable_results[printable_results["state"] == "C"].index
        plotting_results = printable_results.drop(indices_to_drop, inplace=False)
        RESULTS_time = plotting_results.time.values
        RESULTS_MCF = plotting_results.MCF.values
        RESULTS_variance = plotting_results.variance.values
        RESULTS_lower = plotting_results.MCF_lower.values
        RESULTS_upper = plotting_results.MCF_upper.values

        self.results = printable_results
        self.time = RESULTS_time
        self.MCF = RESULTS_MCF
        self.lower = RESULTS_lower
        self.upper = RESULTS_upper
        self.variance = RESULTS_variance

        CI_rounded = CI * 100
        if CI_rounded % 1 == 0:
            CI_rounded = int(CI * 100)

        if print_results is True:
            st.write(
                "## Mean Cumulative Function results"
            )
            st.write(self.results)

        if show_plot is True:
            x_MCF = [0, RESULTS_time[0]]
            y_MCF = [0, 0]
            y_upper = [0, 0]
            y_lower = [0, 0]
            x_MCF.append(RESULTS_time[0])
            y_MCF.append(RESULTS_MCF[0])
            y_upper.append(RESULTS_upper[0])
            y_lower.append(RESULTS_lower[0])
            for i, t in enumerate(RESULTS_time):
                if i > 0:
                    x_MCF.append(RESULTS_time[i])
                    y_MCF.append(RESULTS_MCF[i - 1])
                    y_upper.append(RESULTS_upper[i - 1])
                    y_lower.append(RESULTS_lower[i - 1])
                    x_MCF.append(RESULTS_time[i])
                    y_MCF.append(RESULTS_MCF[i])
                    y_upper.append(RESULTS_upper[i])
                    y_lower.append(RESULTS_lower[i])
            x_MCF.append(last_time)  # add the last horizontal line
            y_MCF.append(RESULTS_MCF[-1])
            y_upper.append(RESULTS_upper[-1])
            y_lower.append(RESULTS_lower[-1])
            title_str = "Non-parametric estimate of the Mean Cumulative Function"

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_MCF, y=y_upper, mode='lines',  marker=dict(color = 'gray'), visible = True))
            fig.add_trace(go.Scatter(x=x_MCF, y=y_lower, mode='lines',  marker=dict(color = 'gray'), visible = True, fill='tonexty'))
            fig.add_trace(go.Scatter(x=x_MCF, y=y_MCF, mode='lines',  marker=dict(color = 'red'), visible = True))
            
            fig.update_layout(width = 1900, height = 600, title="Non-parametric estimate of the Mean Cumulative Function", yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log, showlegend=False) #size of figure
            fig.update_xaxes(title = "Time")
            fig.update_yaxes(title = "Mean cumulative number of failures")
            fig.update_xaxes(range=[0, last_time])
            fig.update_yaxes(range=[0, max(RESULTS_upper) * 1.05])			
            st.plotly_chart(fig)


class MCF_parametric:
    """
    The Mean Cumulative Function (MCF) is a cumulative history function that
    shows the cumulative number of recurrences of an event, such as repairs over
    time. In the context of repairs over time, the value of the MCF can be
    thought of as the average number of repairs that each system will have
    undergone after a certain time. It is only applicable to repairable systems
    and assumes that each event (repair) is identical. In the case of the fitted
    paramertic MCF, it is assumed that each system's MCF is identical.
    The shape (beta parameter) of the MCF is a key indicator that shows whether
    the systems are improving (beta<1), worsening (beta>1), or staying the same
    (beta=1) over time. If the MCF is concave down (appearing to level out) then
    the system is improving. A straight line (constant increase) indicates it is
    staying the same. Concave up (getting steeper) shows the system is worsening
    as repairs are required more frequently as time progresses.
    Parameters
    ----------
    data : list
        The repair times for each system. Format this as a list of lists. eg.
        data=[[4,7,9],[3,8,12]] would be the data for 2 systems. The largest
        time for each system is assumed to be the retirement time and is treated
        as a right censored value. If the system was retired immediately after
        the last repair then you must include a repeated value at the end as
        this will be used to indicate a right censored value. eg. A system that
        had repairs at 4, 7, and 9 then was retired after the last repair would
        be entered as data = [4,7,9,9] since the last value is treated as a
        right censored value. If you only have data from 1 system you may enter
        the data in a single list as data = [3,7,12] and it will be nested
        within another list automatically.
    print_results : bool, optional
        Prints the table of MCF results (state, time, MCF_lower, MCF, MCF_upper,
        variance). Default = True.
    CI : float, optional
        Confidence interval. Must be between 0 and 1. Default = 0.95 for 95% CI
        (one sided).
    show_plot : bool, optional
        If True the plot will be shown. Default = True. Use plt.show() to show
        it.
    plot_CI : bool, optional
        If True, the plot will include the confidence intervals. Default = True.
        Set as False to remove the confidence intervals from the plot.
    kwargs
        Plotting keywords that are passed directly to matplotlib (e.g. color,
        label, linestyle).
    Returns
    -------
    times : array
        This is the times (x values) from the scatter plot. This value is
        calculated using MCF_nonparametric.
    MCF : array
        This is the MCF (y values) from the scatter plot. This value is
        calculated using MCF_nonparametric.
    alpha : float
        The calculated alpha parameter from MCF = (t/alpha)^beta
    beta : float
        The calculated beta parameter from MCF = (t/alpha)^beta
    alpha_SE : float
        The standard error in the alpha parameter
    beta_SE : float
        The standard error in the beta parameter
    cov_alpha_beta : float
        The covariance between the parameters
    alpha_upper : float
        The upper CI estimate of the parameter
    alpha_lower : float
        The lower CI estimate of the parameter
    beta_upper : float
        The upper CI estimate of the parameter
    beta_lower : float
        The lower CI estimate of the parameter
    results : dataframe
        A dataframe of the results (point estimate, standard error, Lower CI and
        Upper CI for each parameter)
    Notes
    -----
    This example is taken from Reliasoft's example (available at
    http://reliawiki.org/index.php/Recurrent_Event_Data_Analysis). The failure
    times and retirement times (retirement time is indicated by +) of 5 systems
    are:
    +------------+--------------+
    | System     | Times        |
    +------------+--------------+
    | 1          | 5,10,15,17+  |
    +------------+--------------+
    | 2          | 6,13,17,19+  |
    +------------+--------------+
    | 3          | 12,20,25,26+ |
    +------------+--------------+
    | 4          | 13,15,24+    |
    +------------+--------------+
    | 5          | 16,22,25,28+ |
    +------------+--------------+
    .. code:: python
        from reliability.Repairable_systems import MCF_parametric
        times = [[5, 10, 15, 17], [6, 13, 17, 19], [12, 20, 25, 26], [13, 15, 24], [16, 22, 25, 28]]
        MCF_parametric(data=times)
    """

    def __init__(
        self, data, CI=0.95, plot_CI=True, print_results=True, show_plot=True, **kwargs
    ):

        if CI <= 0 or CI >= 1:
            raise ValueError(
                "CI must be between 0 and 1. Default is 0.95 for 95% Confidence interval."
            )

        MCF_NP = MCF_nonparametric(
            data=data, print_results=False, show_plot=False
        )  # all the MCF calculations to get the plot points are done in MCF_nonparametric
        self.times = MCF_NP.time
        self.MCF = MCF_NP.MCF

        # initial guess using least squares regression of linearised function
        # we must convert this back to list due to an issue within numpy dealing with the log of floats
        ln_x = np.log(list(self.times))
        ln_y = np.log(list(self.MCF))
        guess_fit = np.polyfit(ln_x, ln_y, deg=1)
        beta_guess = guess_fit[0]
        alpha_guess = np.exp(-guess_fit[1] / beta_guess)
        guess = [
            alpha_guess,
            beta_guess,
        ]  # guess for curve_fit. This guess is good but curve fit makes it much better.

        # actual fitting using curve_fit with initial guess from least squares
        def __MCF_eqn(t, a, b):  # objective function for curve_fit
            return (t / a) ** b

        fit = curve_fit(__MCF_eqn, self.times, self.MCF, p0=guess)
        alpha = fit[0][0]
        beta = fit[0][1]
        var_alpha = fit[1][0][
            0
        ]  # curve_fit returns the variance and covariance from the optimizer
        var_beta = fit[1][1][1]
        cov_alpha_beta = fit[1][0][1]

        Z = -ss.norm.ppf((1 - CI) / 2)
        self.alpha = alpha
        self.alpha_SE = var_alpha ** 0.5
        self.beta = beta
        self.beta_SE = var_beta ** 0.5
        self.cov_alpha_beta = cov_alpha_beta
        self.alpha_upper = self.alpha * (np.exp(Z * (self.alpha_SE / self.alpha)))
        self.alpha_lower = self.alpha * (np.exp(-Z * (self.alpha_SE / self.alpha)))
        self.beta_upper = self.beta * (np.exp(Z * (self.beta_SE / self.beta)))
        self.beta_lower = self.beta * (np.exp(-Z * (self.beta_SE / self.beta)))

        Data = {
            "Parameter": ["Alpha", "Beta"],
            "Point Estimate": [self.alpha, self.beta],
            "Standard Error": [self.alpha_SE, self.beta_SE],
            "Lower CI": [self.alpha_lower, self.beta_lower],
            "Upper CI": [self.alpha_upper, self.beta_upper],
        }
        self.results = pd.DataFrame(
            Data,
            columns=[
                "Parameter",
                "Point Estimate",
                "Standard Error",
                "Lower CI",
                "Upper CI",
            ],
        )

        if print_results is True:
            CI_rounded = CI * 100
            if CI_rounded % 1 == 0:
                CI_rounded = int(CI * 100)
            st.write(
                
                    "## Mean Cumulative Function Parametric Model"
            )
            st.write("MCF = (t/α)^β")
            st.write(self.results)
            if self.beta_upper <= 1:
                st.write(
                    "Since Beta is less than 1, the system repair rate is **IMPROVING** over time."
                )
            elif self.beta_lower < 1 and self.beta_upper > 1:
                st.write(
                    "Since Beta is approximately 1, the system repair rate is remaining **CONSTANT** over time."
                )
            else:
                st.write(
                    "Since Beta is greater than 1, the system repair rate is **WORSENING** over time."
                )

        if show_plot is True:


            x_line = np.linspace(0.001, max(self.times) * 10, 1000)
            y_line = (x_line / alpha) ** beta
            p1 = -(beta / alpha) * (x_line / alpha) ** beta
            p2 = ((x_line / alpha) ** beta) * np.log(x_line / alpha)
            var = (
                var_alpha * p1 ** 2
                + var_beta * p2 ** 2
                + 2 * p1 * p2 * cov_alpha_beta
            )
            SD = var ** 0.5
            y_line_lower = y_line * np.exp((-Z * SD) / y_line)
            y_line_upper = y_line * np.exp((Z * SD) / y_line)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_line, y=y_line_upper, mode='lines',  marker=dict(color = 'gray'), visible = True))
            fig.add_trace(go.Scatter(x=x_line, y=y_line_lower, mode='lines',  marker=dict(color = 'gray'), visible = True, fill='tonexty'))
            fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines',  marker=dict(color = 'red'), visible = True))
            fig.add_trace(go.Scatter(x=self.times, y=self.MCF, mode='markers', marker=dict(color = 'red'), visible = True))
            
            fig.update_layout(width = 1900, height = 600, title="Parametric estimate of the Mean Cumulative Function", yaxis=dict(tickformat='.2e'), xaxis=dict(tickformat='.2e'), updatemenus=updatemenus_log, showlegend=False) #size of figure
            fig.update_xaxes(title = "Time")
            fig.update_yaxes(title = "Mean cumulative number of failures")
            fig.update_xaxes(range=[0, max(self.times) * 1.2])
            fig.update_yaxes(range=[0, max(self.MCF) * 1.4])			
            st.plotly_chart(fig)

#st.set_page_config(page_title="Repairable system",page_icon="📈",layout="wide", initial_sidebar_state="expanded")

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


    st.title("Repairable Systems")


    mod = st.selectbox("Which analysis would you like to perform?",("Reliability Growth", "Optimal replacement time", "Rate of occurrence of failures (ROCOF)", "Mean cumulative function (MCF)"))
    # mod = "Mean cumulative function (MCF)"
    if mod == 'Optimal replacement time':
        # replacement
        helpbutton = st.expander("Help")
        helpbutton.write("This function calculates the cost per unit time to determine how cost varies with replacement time. \
                        The cost model may be HPP (as good as new replacement, with Restoration Factor equal to 0) \
                        or NHPP (as good as old replacement, with the Restoration Factor equal to 1). \
                        Default is HPP, but this can be controled with the parameter Restorarion Factor.  ")
        helpbutton.info("Costs in the above context should include all associated costs of Preventive Maintenance and \
                        Corective Maintenance. These are not just the costs associated with parts and labor but may also \
                        include other costs such as: system downtime, loss of production output, and customer satisfaction.")
        pm = st.number_input(label='Preventative Maintenance Cost', min_value=0, format='%d')
        cm = st.number_input(label='Corrective Maintenance Cost', min_value=0, format='%d')
        alpha = st.number_input(label='Scale Parameter of the Weibull Distribution', min_value=0.)
        beta = st.number_input(label='Shape Parameter of the Weibull Distribution', min_value=1.)
        q = st.radio(label='Restoration Factor.',options=np.array([0,1]))

    if mod == "Mean cumulative function (MCF)":
        helpbutton = st.expander("Help")
        helpbutton.write("The Mean Cumulative Function (MCF) is a cumulative history function that \
                        shows the cumulative number of recurrences of an event, such as repairs over \
                        time. In the context of repairs over time, the value of the MCF can be \
                        thought of as the average number of repairs that each system will have \
                        undergone after a certain time.")
        helpbutton.write("If repairs are assumed to be identical, than the parametric MCF is estimated. \
                        If repairs are assumed not to be identical, than the non-parametric MCF is calculated.")
        helpbutton.write("The non-parametric estimate of the MCF provides both the estimate of the MCF \
                        and the one-sided confidence bounds at a particular time.")
        helpbutton.write("The estimates of the parametric MCF are obtained using first the non-parametric MCF \
                        to obtain the points for the plot. From these points, a Non-Homogeneous Poisson Process (NHPP) is fitted \
                        considering the equation below.")
        helpbutton.latex(r''' MCF(t) = \frac{t^\beta}{\alpha^\beta}''' )
        helpbutton.write("The purpose of fitting a parametric model is to obtain the shape parameter ($\\beta$)\
                        which indicates the long term health of the system. If the MCF is concave down ($\\beta$<1) \
                        then the system is improving. A straight line ($\\beta$=1) indicates it is staying the same.\
                        Concave up ($\\beta$>1) shows the system is worsening as repairs are required more frequently as time progresses.")
                        
        # The shape of the MCF is a key indicator that shows whether the systems are
        # improving, worsening, or staying the same over time. If the MCF is concave
        # down (appearing to level out) then the system is improving. A straight line
        # (constant increase) indicates it is staying the same. Concave up (getting
        # steeper) shows the system is worsening as repairs are required more
        # frequently as time progresses.")
        # helpbutton.info("If repairs are assumed to be identical, than the parametric MCF is estimated. \
        #                 If repairs are assumed not to be identical, than the non-parametric MCF is calculated.")
        
        expander = st.expander("Data format")
        expander.info('Upload an excel file that contains the failure interarrival times of each equipment that should be considered in the analysis. Note: the last interval will be considered as censored.')
        df = {'Eq1': [10, 15, 8, 20, 21, 12, 13, 30, 5], 'Eq2': [10, 15, 8, 20, 21, 12, 13, 30, 5]}
        df = pd.DataFrame.from_dict(df)
        expander.write(df, width=50)  
        
        col2_1, col2_2 = st.columns(2)
        uploaded_file = col2_1.file_uploader("Upload a XLSX file", \
            type="xlsx", accept_multiple_files=False)
        if uploaded_file:
            aux = pd.read_excel(uploaded_file)
            col2_2.dataframe(aux)
            data = []
            for col in aux.columns:
                cleanedList = [x for x in aux[col].to_list() if str(x) != 'nan']
                data.append(cleanedList)
        
        parametric = st.radio('Should we assume that repairs are identical?',('No', 'Yes'))
        ci = st.number_input('CI', min_value=0., max_value=1., value=0.9)

    if mod == 'Reliability Growth':
        # Reliability growth
        helpbutton = st.expander("Help")

        helpbutton.write("Uses the Duane method to find the instantaneous MTBF and produce a reliability \
                        growth plot. The instantaneous MTBF is given by the equation below.")
        helpbutton.latex(r'''MTBF = \frac{t^{1-\beta}}{\lambda \beta}''')
        helpbutton.write("The estimation of the parameters $\lambda$ and $\\beta$ is obtained as follows. Firstly, \
                        we need to calculate the Cumulative Mean Time Between Failures (CMTBF) as the equation \
                        below, where *t* is the time when the failure occured and *N* is the sequence of the failure.")
        helpbutton.latex(r''' CMTBF = \frac{t}{N}''' )
        helpbutton.write("By plotting $\ln(t)$ vs $\ln(t/N)$ we obtain a straight line which is used get the \
                        parameters $\lambda$ and $\\beta$.")
        helpbutton.info("Note that the maximum achieveable reliability is locked in by design, so reliability \
                        growth above the design reliability is only possible through design changes.")

        expander = st.expander("Data format")
        expander.info('Upload an excel file that contains one column with failure interarrival times ("Time").')
        df = {'Time': [10, 15, 8, 20, 21, 12, 13, 30, 5]}
        df = pd.DataFrame.from_dict(df)
        expander.write(df, width=50)

        col2_1, col2_2 = st.columns(2)
        uploaded_file = col2_1.file_uploader("Upload a XLSX file", \
            type="xlsx", accept_multiple_files=False)
        if uploaded_file:
            data = pd.read_excel(uploaded_file)
            if df.shape[1] == 1:
                col2_2.dataframe(data)
                for col in data.columns:
                    times = np.cumsum(data[col].to_numpy())
            else:
                st.warning('Check the data format.')

        mtbf = st.number_input(label='Target MTBF',min_value=0, format='%d')
    if mod == 'Rate of occurrence of failures (ROCOF)':
        # ROCOF
        helpbutton = st.expander("Help")
        helpbutton.write("Rate of occurrence of failures (ROCOF) is used to model the trend \
                        (constant, increasing, decreasing) in the failure interarrival times. \
                        The ROCOF is only calculated if the trend is constant. If trend is not \
                        constant then ROCOF changes over time in accordance with the equation below \
                        and the result only poits whether it is increasing or decreasing. \
                        First, it is necessary to conduct a statistical test to determine if \
                        there is a statistically significant trend, and if there is a trend we \
                        can then model that trend, by estimating the parameters $\\beta$ and $\lambda$, \
                        using a Power Law NHPP.")
        helpbutton.latex(r''' ROCOF(t) = \lambda \beta {t}^{\beta -1}''' )
        helpbutton.info("The statistical test is the Laplace test which compares the \
                        Laplace test statistic (*U*) with the z value (*z_crit*) from the standard Normal Distribution.")
        
        expander = st.expander("Data format")
        expander.info('Upload an excel file that contains one column with failure interarrival times ("Time").')
        df = {'Time': [10, 15, 8, 20, 21, 12, 13, 30, 5]}
        df = pd.DataFrame.from_dict(df)
        expander.write(df, width=50)

        col2_1, col2_2 = st.columns(2)
        uploaded_file = col2_1.file_uploader("Upload a XLSX file", \
            type="xlsx", accept_multiple_files=False)
        if uploaded_file:
            data = pd.read_excel(uploaded_file)
            if df.shape[1] == 1:
                col2_2.dataframe(data)
                for col in data.columns:
                    times = np.cumsum(data[col].to_numpy())
            else:
                st.warning('Check the data format.')
        
        censored = st.number_input(label='Censored time (optional)', min_value=0, format='%d')
        if censored == 0:
            censored = None
        ci = st.number_input('CI', min_value=0., max_value=1., value=0.9)



    st.write('---')
    if st.button("Run"):
        if mod == 'Optimal replacement time':
            optimal_replacement_time(cost_PM=pm, cost_CM=cm, weibull_alpha=alpha, weibull_beta=beta,q=q)
        elif mod == 'Reliability Growth' and data is not None and mtbf is not None:
            reliability_growth(times=times,target_MTBF=mtbf,label='Reliability growth curve',xmax=500000)
        elif mod == 'ROCOF' and data is not None:    
            ROCOF(times_between_failures=times, test_end=censored, CI=ci)
        elif mod == 'Mean cumulative function (MCF)':
            if parametric == 'Yes':
                MCF_parametric(data=data, CI=ci)
            else:
                MCF_nonparametric(data=data, CI=ci)

