
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
from scipy.optimize import curve_fit
import sys

def fit_(x_data, y_data):
    # Define the model function
    def model_func(x, a, b):
        return a * x + b

    # Fit the model to the data
    popt, _ = curve_fit(model_func, x_data, y_data)
    return model_func(x_data, *popt)

def activation_energy(temperatures, sigmas):
    # Fit a straight line (first-degree polynomial) to the Arrhenius plot
    coeff = np.polyfit(1/temperatures, sigmas, 1)
    slope = coeff[0]
    return -round(slope * 0.00008617, 2)
    
def arrhenius_plot(temperatures_2d, ionic_condu_2d, labels):
    colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown']
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.02, horizontal_spacing=0.1)

    for c, temp, ionic_condu, label in zip(colors, temperatures_2d, ionic_condu_2d, labels):
        temperature = 1000 / temp
        y_fit = fit_(temperature, ionic_condu)
        E_a = activation_energy(temp, y_fit)
        fig.add_trace(go.Scatter(x=temperature, 
                                 y=ionic_condu, 
                                 mode='markers', # circle-open, markers
                                 marker=dict(size=5, color=c, symbol='circle-open'), 
                                 name=f"{label} E<sub>a</sub>={E_a}eV",
                                 showlegend=False), 
                      row=1, col=1)

        
        fig.add_trace(go.Scatter(x=temperature, y=y_fit, mode='lines',
                                 line=dict(color=c, width=2, dash='dash'),  #solid, 'dot', 'dashdot'
                                 name=f"{label} E<sub>a</sub>={E_a}eV",
                                 showlegend=True), 
                      row=1, col=1)
    
    font_size = 18
    x_axis_details = dict(
        showline=True, linecolor='black', linewidth=2, mirror=True,
        titlefont=dict(size=font_size + 2, color="black"), ticks="inside",
        tickwidth=2, ticklen=10, minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True),
    )
    y_axis_details = dict(
        showline=True, linecolor='black', linewidth=2, mirror=True,
        titlefont=dict(size=font_size + 2, color="black"), ticks="inside",
        tickwidth=2, ticklen=10, minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=False),
    )
    fig.update_layout(
        xaxis=x_axis_details, yaxis=y_axis_details,
        legend=dict(orientation='h', 
                    xref='paper', 
                    yref='paper',
                    xanchor="center",
                    yanchor="bottom",
                    x=0.5, 
                    y=1.01, 
                    font_size=font_size, 
                    itemsizing = 'constant',                    
                    ),
        
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        height=600, 
        width=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor='rgba(0,0,0,0)',
        font_size=font_size, font=dict(size=font_size, color="black"),)

    fig.update_xaxes(title_text="T/1000(K<sup>-1</sup>)", row=1, col=1, title_standoff=10)
    fig.update_yaxes(title_text="ln(σ) (S/cm)", row=1, col=1)

    fig.write_image("arrhenius_plot.png", format='png', scale=2)
    fig.write_image("arrhenius_plot.svg")
    return fig.show()

def plot_arrhenius():
    ionic_conductivity_files = []
    for i in range(1, 10):
        try:
            ionic_conductivity_files.append(sys.argv[i])
        except:
            if i == 1:
                print("No autopsy folders are availabel HERE!!!")
                print("Usage: python plot_autopsy_msd.py folder1 folder2 folder3 ...")
                exit()
            else:
                pass

    ionic_conductivitys_2d, temperatures_2d, sigmas = [], [], []
    for file in ionic_conductivity_files:
        # Read data
        sigma_data = pd.read_csv(file)
        ionic_conductivitys_2d.append(np.log(np.array(sigma_data['Li_self_sigma'])))
        sigmas.append(np.array(sigma_data['Li_self_sigma']).astype(float))
        temperatures_2d.append(np.array(sigma_data['Li_self_temperature']))
        
    
    # Labels for the plot
    labels = [label[:-4] for label in ionic_conductivity_files] 
    #ionic_conductivitys_2d = np.log(np.array(ionic_conductivitys_2d))
    #temperatures_2d = np.array(temperatures_2d) 

    # Print as table
    for sigma in sigmas:
        print([f"{float(s):.1e}" for s in sigma])

    # Plot
    arrhenius_plot(temperatures_2d, ionic_conductivitys_2d, labels)
    return

