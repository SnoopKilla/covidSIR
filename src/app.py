import os
import gradio as gr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from src.sampler import mcmc_sampler
from src.data_parser import Parser
matplotlib.use('Agg')
font = {'size': 30}
matplotlib.rc('font', **font)


def sample(country, d, n_iterations, burnin):
    P = parser.parse_population(country)
    start_date = "2020-03-01"
    end_date = "2020-06-15"
    i, r = parser.parse_data(start_date, end_date, country)
    i, r = i.values, r.values
    s = np.repeat(P, i.shape[0]) - i - r

    p, lam, t, lam_ar, t_ar = mcmc_sampler(s, i, d, P, n_iterations, burnin,
                                           M=3, sigma=0.01,
                                           alpha=np.repeat(2, d),
                                           beta=np.repeat(0.1, d),
                                           a=1, b=1, phi=0.995)

    lam_estimated = np.average(lam, axis=1)
    t_estimated = np.average(t, axis=1)
    p_estimated = np.average(p)

    # Plot the series.
    fig, axs = plt.subplots(nrows=2)
    fig.set_figheight(30)
    fig.set_figwidth(30)
    ax1_left = axs[0]
    ax2_left = axs[1]
    ax1_right = ax1_left.twinx()
    ax2_right = ax2_left.twinx()
    ax1_left.plot(s, color='red', label="Susceptible")
    ax1_right.plot(i, color='blue', label="Infected")
    ax1_left.legend(loc=2)
    ax1_right.legend(loc=1)
    delta_i = -np.diff(s)
    ax2_left.plot(delta_i, color="blue", label="Newly Infected Individuals")
    ax2_right.plot(i, color='blue', linestyle='dashed', label="Infected")
    ax2_left.legend(loc=2)
    ax2_right.legend(loc=1)
    # Display obtained breakpoints on plot.
    for breakpoint in np.average(t, axis=1):
        ax1_right.axvline(breakpoint, color="green")
        ax2_right.axvline(breakpoint, color="green")

    # Get output strings
    lam_string = ""
    for j, lam_component in enumerate(lam_estimated):
        lam_string += f"Component {j+1}: {round(lam_component, 4)}\n"
    lam_string = lam_string.rstrip()
    t_string = ""
    for j, t_component in enumerate(t_estimated):
        t_string += f"Breakpoint {j+1}: {int(round(t_component, 0))}\n"
    t_string = t_string.rstrip()
    p_string = f"{round(p_estimated, 4)}"

    return fig, lam_string, t_string, p_string


if __name__ == "__main__":
    confirmed_path = "confirmed.csv"
    deaths_path = "deaths.csv"
    recovered_path = "recovered.csv"
    population_path = "population.csv"
    data_path = os.path.join(os.getcwd(), "data")
    parser = Parser(os.path.join(data_path, confirmed_path),
                    os.path.join(data_path, deaths_path),
                    os.path.join(data_path, recovered_path),
                    os.path.join(data_path, population_path))
    countries = parser.countries

    # Inputs
    dropdown = gr.Dropdown(choices=countries, value="Germany",
                           label="Select the Country")
    slider = gr.Slider(minimum=1, maximum=5, value=3, step=1,
                       label="Select the Number of Breakpoints")
    n_iterations = gr.Number(value=10000, precision=0,
                             label="Select the Number of iterations")
    burnin = gr.Number(value=1000, precision=0,
                       label="Select the Number of Burn-In Iterations",
                       info="Such iterations will be discarded.")

    # Outputs
    plot = gr.Plot(label="Results")
    lam = gr.Text(label="Estimated Lambda")
    t = gr.Text(label="Estimated Breakpoints")
    p = gr.Text(label="Estimated Recovery Probability")
    interface = gr.Interface(sample,
                             inputs=[dropdown, slider, n_iterations, burnin],
                             outputs=[plot, lam, t, p])
    interface.launch()
