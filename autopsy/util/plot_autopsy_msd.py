import warnings

# Filter and ignore specific warnings
warnings.filterwarnings('ignore')

from ase.io import read, write
from ase import Atoms

from plotly.subplots import make_subplots
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
import re, os, sys


    
def plot_autopsy_msd():
    # Define the folder name
    out_folder_name = "msd_plots"
    
    # Check if the folder exists
    if not os.path.exists(out_folder_name):
        # If it doesn't exist, create the folder
        os.makedirs(out_folder_name)
        print(f"Folder '{out_folder_name}' created.")


    folders = []
    for i in range(1, 10):
        try:
            folders.append(sys.argv[i])
        except:
            if i == 1:
                print("No autopsy folders are availabel HERE!!!")
                print("Usage: python plot_autopsy_msd.py folder1 folder2 folder3 ...")
                exit()
            else:
                pass
    species_ = grep_autopsy_species(folders)
    df_sigmas = pd.DataFrame() 
    for specie in species_:
        print("#############")
        print(f"    {specie}")
        print("#############")
        for msd_type in ['self', 'distinct', 'total']:        
            msds, ds, times, labels, volumes, n_particles, temperatures = grep_autopsy_data(folders, msd_type, specie) #ds is diffusion coeff
            # Plot msds        
            if labels:
                output_name = f"./{out_folder_name}/{specie}_{msd_type}"
                slopes, dts, betas, selected_msd_slopes = plot_msd(msds, ds, times, labels, output_name)# ds is diff_coeff
                if 'distinct' != msd_type:
                   df_sigma =  plot_sigma(slopes, selected_msd_slopes, dts, betas, labels, temperatures, volumes, n_particles, output_name)
                   df_sigmas[f"{specie}_{msd_type}_temperature"]=df_sigma['Temperature']
                   df_sigmas[f"{specie}_{msd_type}_sigma"]=df_sigma['sigma']

                #plot_sigma(slopes, dts, labels, output_name)
    df_sigmas.to_csv('./msd_plots/ionic_conductivity.txt', index=False)
    plot_autopsy_msd_species_wise()
    return
    

def grep_autopsy_species(folders):
    species = []
    for folder in folders:
        data_df = pd.read_csv(f"{folder}/msd_and_ngp.csv", delimiter=',')
        details_df = pd.read_csv(f"{folder}/details.csv", delimiter=',')
        species.extend(list(details_df['species_name']))
    species = list(set(species))
    return species

def grep_autopsy_data(folders, msd_type, specie):   
    '''
    grep the data specie-wise for all foders
    '''

    msds, times, temperatures, volumes, n_particles = [], [], [], [], []
    labels = [] 
    for folder, label in zip(folders, folders):   
        label = label.split('/')[-1]
        data_df = pd.read_csv(f"{folder}/msd_and_ngp.csv", delimiter=',').iloc[20:-2500]
        details_df = pd.read_csv(f"{folder}/details.csv", delimiter=',')

        if specie in list(details_df['species_name']):   
            msd = (
                data_df[f'{specie}_xyz_{msd_type}_MSD']) * 1e-20 

            msds.append(msd)
            labels.append(f"{specie}@{label}")
            time = None
            try:
                time = np.array(data_df['Time']) * 1e-12
            except:
                time = np.arange(0, len(msd)) * 100 * 1e-15
            times.append(time)

            a = np.array(details_df['a']).astype(float)[np.array(details_df['species_name']) == specie][0]
            b = np.array(details_df['b']).astype(float)[np.array(details_df['species_name']) == specie][0]
            c = np.array(details_df['c']).astype(float)[np.array(details_df['species_name']) == specie][0]
            zmin = np.array(details_df['zmin']).astype(float)[np.array(details_df['species_name']) == specie][0]
            zmax = np.array(details_df['zmax']).astype(float)[np.array(details_df['species_name']) == specie][0]
            c_ = zmax-zmin
            volumes.append(a*b*c_)
            n_particles.append(np.array(details_df['n_atoms']).astype(int)[np.array(details_df['species_name']) == specie][0])
            try:
                temperatures.append(np.array(details_df['temp_final']).astype(float)[np.array(details_df['species_name']) == specie][0])
            except:
                temperatures.append(300)



    ds = [None]*len(msds)
    return msds, ds, times, labels, volumes, n_particles, temperatures 


def plot_msd(msds, diff_co, times, labels, output_name):  

    # Sample colors for different datasets
    #colors = ['green', 'red', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black', 'brown']
    colors =['#008000', '#FF0000', '#0000FF', '#FFA500', '#800080', '#00FFFF', '#FF00FF', '#FFFF00', '#000000', '#A52A2A']
    fig = make_subplots(rows=2, 
                        cols=2, 
                        shared_xaxes=False, 
                        vertical_spacing=0.15, 
                        horizontal_spacing=0.15,
                        insets=[dict(cell=(2,1), b=0.45, l = 0.45, w=0.45, h= 0.45)],
                        specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
                       )
    
    legend_length = 0
    slopes, dts, betas, selected_msd_slopes = [], [], [], []
    for msd, d, c, label, time in zip(msds, diff_co, colors, labels, times):
        window_size = int(len(msd)*0.01)
        poly_order = 1
        legend_length += len(label)

        # Apply Savitzky-Golay filter to smooth the noisy data
        smoothed_msd = savgol_filter(msd, window_size, poly_order)

        #################################
        #Original MSD 
        #################################
        c_ = tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) + (0.4,)  # Add 0.5 for 50% opacity
        fig.add_trace(go.Scatter(x=time*1e+9, 
                                 y=msd, 
                                 line=dict(color=f'rgba{c_}'),
                                 name = label, 
                                 showlegend=False), 
                      row=1, 
                      col=1)   

        #################################
        # Smoothed MSD       
        #################################
        fig.add_trace(go.Scatter(x=time*1e+9, 
                                 y=smoothed_msd, 
                                 line=dict(color=c),
                                 name = label, 
                                 showlegend=True), 
                      row=1, 
                      col=1)
        
        #################################
        # Diffusion coefficient
        #################################
        if isinstance(d, np.ndarray):
            print(f"{label} D: {d[100:-100].mean():.1e} m^2/s")
            fig.add_trace(go.Scatter(x=time*1e+9, 
                                     y=d[skip_initial_indices:skip_final_indices], 
                                     line=dict(color=c),
                                     name = label, 
                                     showlegend=False), 
                          row=2, 
                          col=2)

        elif d == None: 
            slopes_ = np.gradient(smoothed_msd, time)
            slopes.append(slopes_)
            dts.append(time)
            diff_co = np.array(slopes_)/6
            window_size = int(len(msd)*0.01)
            #diff_co = smooth_out(diff_co_, window_size, poly_order)
            fig.add_trace(go.Scatter(x=time*1e+9, #slop_time[skip_initial:len(slop_time)-skip_final]*1e+9, 
                                     y=diff_co, #[skip_initial:len(slop_time)-skip_final], 
                                     line=dict(color=c),
                                     name = label, 
                                     showlegend=False), 
                          row=2, 
                          col=2)
        

        

        #################################
        # Plot log(x) and log(y) in the second subplot
        #################################
        logx = np.log(np.abs(time))
        logy = np.log(np.abs(smoothed_msd))

        
        #smoothed_log_msd = smooth_out(np.log10(msd), n_smooth, window_size, poly_order)
        fig.add_trace(go.Scatter(x=logx, #time, 
                                 y=logy, #smoothed_msd, 
                                 line=dict(color=c), 
                                 showlegend=False), 
                      row=1, col=2)
    
        # Calculate the numerical derivative
        # Calculate the derivative with intervals of 10 values
        #dy_dx = smooth_out(np.gradient(logy, logx), window_size, poly_order)
        dy_dx_ = np.gradient(logy, logx)
        window_size = window_size * 2
        dy_dx = savgol_filter(dy_dx_, window_size, poly_order)
        betas.append(dy_dx)

        # Plot x_derivative and dy_dx in the first subplot
        fig.add_trace(go.Scatter(x=time[range(len(dy_dx))]*1e+9, 
                                 y= dy_dx, 
                                 line=dict(color=c), 
                                 showlegend=False), 
                      row=2, col=1)
        '''
        fig.add_trace(go.Scatter(x=x[-400:-100],                             
                                 y= dy_dx[-400:-100], 
                                 line=dict(color=c), 
                                 showlegend=False,
                                 xaxis='x4',
                                 yaxis='y4'
                                ))
        '''
        # Finding slope
        valid_indices = (dy_dx > 0.75) & (dy_dx < 1.2)
        if not np.any(valid_indices):
            valid_indices = (dy_dx > 0.55) & (dy_dx < 0.76)
        if not np.any(valid_indices):
            valid_indices = (dy_dx > 0.35) & (dy_dx < 0.76)
        try:
            if np.any(valid_indices):
                time_max = max(time[valid_indices])
                time_min = min(time[valid_indices])
                max_index = np.where(time == time_max)[0]
                min_index = np.where(time == time_min)[0]
                msd_max = smoothed_msd[max_index]
                msd_min = smoothed_msd[min_index]
                selected_msd_slop = (msd_max - msd_min) / (time_max - time_min)
                selected_msd_slop = selected_msd_slop[0]
            else:
                selected_msd_slop = None
        except Exception as e:
            selected_msd_slop = None
            print(f"An error occurred: {e}")

        selected_msd_slopes.append(selected_msd_slop)
    
    # Update layout
    font_size=15
    x_axis_details=dict(
            #title='Materials',
            #tickmode='array',
            #tickvals=np.arange(int(f_x_range[0]), int(f_x_range[1])),
            #ticktext=directories_,
            gridcolor='lightgray',  # Set the gridline color
            griddash='dash',
            showline=True,  # Show the border line
            linecolor='black',  # Set the border line color
            linewidth = 2,
            mirror = True,
            titlefont=dict(size = font_size+2, color = "black"),
            ticks="inside",
            tickwidth=2,
            ticklen=10,
            #dtick = 1,
            minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True),    
    )
        
    y_axis_details=dict(
            #tickmode='array',
            #tickvals=list(range(1, len(vac_) + 1)),
            gridcolor='lightgray',  # Set the gridline color
            griddash='dash',
            showline=True,  # Show the border line
            linecolor='black',  # Set the border line color
            linewidth = 2,
            #title='VF Energy (eV)',
            titlefont=dict(size = font_size+2, color = "black"),
            mirror = True,
            ticks="inside",
            tickwidth=2,
            ticklen=10,
            #dtick = 1,
            minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=False),
            #automargin=True,
            #anchor = 1,
            #tickformat=".0e",  # Use scientific notation       
    )   

    y_axis_details_exp=dict(
            #tickmode='array',
            #tickvals=list(range(1, len(vac_) + 1)),
            gridcolor='lightgray',  # Set the gridline color
            griddash='dash',
            showline=True,  # Show the border line
            linecolor='black',  # Set the border line color
            linewidth = 2,
            #title='VF Energy (eV)',
            titlefont=dict(size = font_size+2, color = "black"),
            mirror = True,
            ticks="inside",
            tickwidth=2,
            ticklen=10,
            #dtick = 1,
            minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=False),
            #automargin=True,
            #anchor = 1,
            tickformat=".1e",  # Use scientific notation       
    ) 
    
    fig.update_layout(
        #title_text='Subplots Example',
        xaxis=x_axis_details,
        xaxis2=x_axis_details,
        xaxis3=x_axis_details,
        xaxis4=x_axis_details,
        yaxis=y_axis_details_exp,
        yaxis2=y_axis_details,
        yaxis3=y_axis_details,
        yaxis4=y_axis_details_exp,
     
        showlegend=True,
        #legend=dict(orientation='h', yref = 'paper', y = 1.1, font_size=font_size),
        plot_bgcolor='white',  # Set the background color to white
        paper_bgcolor='white',  # Set the paper color to white
        height=600, 
        width=800,
        margin=dict(l=1),
        font_size=font_size,
        font = dict(size = font_size, color = "black"),
        
    
    )
    
    # Configure plot layout    
    fig.update_xaxes(title_text="Time(ns)", row=1, col=1, title_standoff=0)
    fig.update_yaxes(title_text="MSD (m<sup>2</sup>)", row=1, col=1)  # Å title_standoff=0
    fig.update_xaxes(title_text="log(Time) (s)", row=1, col=2, title_standoff=0) #type="log",
    fig.update_yaxes(title_text="log(MSD) (m<sup>2</sup>)", row=1, col=2, title_standoff=0) #title_standoff=0
    fig.update_xaxes(title_text="Time(ns)", row=2, col=1, title_standoff=7)
    fig.update_yaxes(title_text="β", row=2, col=1) #range=[0, 4.5] range=[-0.9, 4.9]
    fig.update_xaxes(title_text="Time(ns)", row=2, col=2, title_standoff=7)
    fig.update_yaxes(title_text="D (m<sup>2</sup>/s)", row=2, col=2, title_standoff=0) #range=[0, 4.5] range=[-0.9, 4.9]    


    fig.update_layout(height=600, 
                      width=900, 
                      #title_text='MSD',
                      legend=dict(font=dict(family="Courier", size=20, color="black"), 
                                yanchor='bottom',
                                y=1.01,  # Position at the top
                                xanchor="center",
                                x=0.5,  # Center horizontally
                                orientation="h"  # Horizontal orientation
                        ),
                     )
    # Assuming fig is your Plotly figure object
    fig.update_layout(legend= {'itemsizing': 'constant'},    
                      margin=dict(l=0, r=0, t=0, b=0),
                      plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
                      paper_bgcolor='rgba(0,0,0,0)'  # Set paper background to transparent
                     )
    

    # Save the plot to an HTML file
    fig.write_html(f"{output_name}_msd.html")
    fig.write_image(f"{output_name}_msd.png")

    # Display the plot
    return slopes, dts, betas, selected_msd_slopes #fig.show()


def ionic_conductivity(temperature, volume, n_particles, slop):
    # Constants
    q = 1.60217663*1e-19 # elementary charge in Coulombs
    kB = 1.380649e-23  # Boltzmann constant in J/K
    T = temperature  # temperature in Kelvin

    n = n_particles/(volume*1e-30) #Li/m^3


    #slop = msd/dt
    if slop is None:
        return 0.0
    else:
        D = np.array(slop)/6  #m/s
    sigma = (n * q**2 * D) / (kB * T) #S/m
    sigma = sigma  / 1e3  # in S/cm

    return sigma #'{:.2e}'.format(sigma*100) 

def plot_sigma(slopes, selected_msd_slopes, dts, betas, labels, temperatures, volumes, n_particles, output_name):  
    '''
    slopes: slopes of all msds under the label
    dts: time interval of slop foe all labels
    labels: each list
    temp...


    '''

    # Sample colors for different datasets
    #colors = ['#2E7F18', '#45731E', '#675E24', '#8D472B', '#B13433', '#C82538']
    colors =['#008000', '#FF0000', '#0000FF', '#FFA500', '#800080', '#00FFFF', '#FF00FF', '#FFFF00', '#000000', '#A52A2A']
    #labels = ['298',  '318',  '338',  '358', '378']
    fig = make_subplots(rows=1, 
                        cols=1, 
                        shared_xaxes=False, 
                        vertical_spacing=0.15, 
                        horizontal_spacing=0.15,
                       )
    sigmas = []
    for slop, selected_msd_slop, dt, beta, volume, temp, n_parti, c, label in zip(slopes, selected_msd_slopes, dts, betas, volumes, temperatures, n_particles, colors, labels):
        sigma = ionic_conductivity(temp, volume, n_parti, slop)
        sigma_total = ionic_conductivity(temp, volume, n_parti, selected_msd_slop)
        sigmas.append(sigma_total)
        if sigma_total is not None:
            sigma_total = '{:.1e}'.format(sigma_total)

        print(f"Calcualted sigma of {output_name} is {sigma_total} {label}")

        skip_initial = int(len(dt) * 0.001)
        skip_final = int(len(dt) * 0.001)

        x = dt   # in sec
        y = sigma           # in meter
        x = x[skip_initial:len(y)-skip_final]
        y = y[skip_initial:len(y)-skip_final]
        fig.add_trace(go.Scatter(x=x*1e+9, 
                                 y=y, 
                                 line=dict(color=c), 
                                 name = label, 
                                 showlegend=True), 
                      row=1, 
                      col=1)
    
    
    # Update layout
    font_size=15
    x_axis_details=dict(
            #title='Materials',
            #tickmode='array',
            #tickvals=np.arange(int(f_x_range[0]), int(f_x_range[1])),
            #ticktext=directories_,
            gridcolor='lightgray',  # Set the gridline color
            griddash='dash',
            showline=True,  # Show the border line
            linecolor='black',  # Set the border line color
            linewidth = 2,
            mirror = True,
            titlefont=dict(size = font_size+2, color = "black"),
            ticks="inside",
            tickwidth=2,
            ticklen=10,
            #tickformat=".e",
            #dtick = 1,
            minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=True),    
    )
        
    y_axis_details=dict(
            #tickmode='array',
            #tickvals=list(range(1, len(vac_) + 1)),
            gridcolor='lightgray',  # Set the gridline color
            griddash='dash',
            showline=True,  # Show the border line
            linecolor='black',  # Set the border line color
            linewidth = 2,
            #title='VF Energy (eV)',
            titlefont=dict(size = font_size+2, color = "black"),
            mirror = True,
            ticks="inside",
            tickwidth=2,
            ticklen=10,
            tickformat=".1e",
            #dtick = 1,
            minor=dict(ticks="inside", ticklen=5, tickwidth=1, tickcolor="black", showgrid=False),
            
            
            #automargin=True
            #anchor = 1
    )    
    fig.update_layout(
        #title_text='Subplots Example',
        xaxis=x_axis_details,
        yaxis=y_axis_details,
     
        showlegend=True,
        #legend=dict(orientation='h', yref = 'paper', y = 1.1, font_size=font_size),
        font_size=font_size,
        font = dict(size = font_size, color = "black"),
        height=400,
        width=500,
        legend=dict(font=dict(family="Courier", size=font_size, color="black"),
                  yanchor='bottom',
                  y=1.01,  # Position at the top
                  xanchor="center",
                  x=0.5,  # Center horizontally
                  orientation="h",  # Horizontal orientation
                  itemsizing="constant",
          ),


        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to transparent
        paper_bgcolor='rgba(0,0,0,0)'  # Set paper background to transparent

        
    
    )
    
    # Configure plot layout    
    fig.update_xaxes(title_text="Time(ns)", row=1, col=1, title_standoff=0)
    fig.update_yaxes(title_text="σ(S/cm)", row=1, col=1)  #title_standoff=0

    # Save the plot to an HTML file
    fig.write_html(f"{output_name}_sigma.html")
    fig.write_image(f"{output_name}_sigma.png")
    df = pd.DataFrame()
    df['Temperature'] = temperatures
    df['sigma'] = sigmas
    # Display the plot
    #print(f"{output_name} is DONE")
    return df #fig.show()



def plot_autopsy_msd_species_wise():
    # Define the folder name
    out_folder_name = "msd_plots"

    # Check if the folder exists
    if not os.path.exists(out_folder_name):
        # If it doesn't exist, create the folder
        os.makedirs(out_folder_name)
        print(f"Folder '{out_folder_name}' created.")

    folders = []
    for i in range(1, 10):
        try:
            folders.append(sys.argv[i])
        except:
            if i == 1:
                print("No autopsy folders are available HERE!!!")
                print("Usage: python plot_autopsy_msd.py folder1 folder2 folder3 ...")
                exit()
            else:
                pass

    species_ = grep_autopsy_species(folders)
    df_sigmas = pd.DataFrame()

    for folder in folders:
        print("#############")
        print(f"    {folder}")
        print("#############")

        for msd_type in ['self', 'distinct', 'total']:
            msds, ds, times, labels, volumes, n_particles, temperatures = grep_autopsy_data_species_wise(folder, msd_type)

            # Plot msds
            if labels:
                output_name = f"./{out_folder_name}/{folder}_{msd_type}"
                slopes, dts, betas, selected_msd_slopes = plot_msd(msds, ds, times, labels, output_name)

                if 'distinct' != msd_type:
                    df_sigma = plot_sigma(slopes, selected_msd_slopes, dts, betas, labels, temperatures, volumes, n_particles, output_name)
                    df_sigmas[f"{folder}_{msd_type}_temperature"] = df_sigma['Temperature']
                    df_sigmas[f"{folder}_{msd_type}_sigma"] = df_sigma['sigma']

    df_sigmas.to_csv('./msd_plots/ionic_conductivity_species_wise.txt', index=False)
    return


def grep_autopsy_data_species_wise(folder, msd_type):
    msds, diffusion_coeffic, times, temperatures, volumes, n_particles = [], [], [], [], [], []
    labels = []

    data_df = pd.read_csv(f"{folder}/msd_and_ngp.csv", delimiter=',').iloc[20:-2500]
    details_df = pd.read_csv(f"{folder}/details.csv", delimiter=',')

    for specie in details_df['species_name']:
        msd = (
            data_df[f'{specie}_xyz_{msd_type}_MSD']) * 1e-20

        msds.append(msd)
        labels.append(f"{specie}")
        time = None

        try:
            time = np.array(data_df['Time']) * 1e-12
        except:
            time = np.arange(0, len(msd)) * 100 * 1e-15
        times.append(time)

        a = np.array(details_df['a']).astype(float)[np.array(details_df['species_name']) == specie][0]
        b = np.array(details_df['b']).astype(float)[np.array(details_df['species_name']) == specie][0]
        c = np.array(details_df['c']).astype(float)[np.array(details_df['species_name']) == specie][0]
        zmin = np.array(details_df['zmin']).astype(float)[np.array(details_df['species_name']) == specie][0]
        zmax = np.array(details_df['zmax']).astype(float)[np.array(details_df['species_name']) == specie][0]
        c_ = zmax - zmin
        volumes.append(a * b * c_)
        n_particles.append(np.array(details_df['n_atoms']).astype(int)[np.array(details_df['species_name']) == specie][0])

        try:
            temperatures.append(np.array(details_df['temp_final']).astype(float)[np.array(details_df['species_name']) == specie][0])
        except:
            temperatures.append(300)

    ds = [None] * len(msds)
    return msds, ds, times, labels, volumes, n_particles, temperatures
