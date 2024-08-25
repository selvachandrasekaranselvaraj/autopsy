import warnings

# Filter and ignore specific warnings
warnings.filterwarnings('ignore')

from ase.io import read, write
from ase import Atoms

from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objs as go

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
import re, os, sys

def grep_lammps_log_species(filenames):
    final_df = process_lammps_log_files(filenames)
    return final_df, [column.split('_')[1] for column in final_df.columns if 'msd' in column]
    
def plot_lammps_log(filenames, skip_initial_indices, smooth_interval, n_smooth, slop_interval):
    final_df, atoms = grep_lammps_log_species(filenames)
    for i, atom in enumerate(atoms):
        n_items = len(final_df)
        natoms = int(final_df[f'{atom}_natoms'][0])
        msds = np.array([np.array(final_df[f'msd_{atom}'].iloc[i], dtype=float) for i in range(n_items)]) #Conver to m^2
        if f'D_{atom}' in list(final_df.columns):    
            ds = np.array([np.array(final_df[f'D_{atom}'].iloc[i], dtype=float) for i in range(n_items)]) #Conver to m^2
        else:
            ds = None
        times = np.array([np.array(final_df[f'Time'].iloc[i], dtype=float) for i in range(n_items)])  #Convert to Sec
        labels = [f"{atom}_{final_df['label'].iloc[i]}" for i in range(n_items)]
        poly_order = 1
        output_name = f"log_msd_{atom}"
        plot_msd(msds, ds, times, labels, output_name, skip_initial_indices, smooth_interval, poly_order, n_smooth, slop_interval)        
    return

def plot_autopsy_msd(skip_initial_indices, skip_final_indices):
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



    for msd_type in ['self', 'distinct', 'total']:        
        for specie in grep_autopsy_species(folders):
            msds, ds, times, labels, output_name = grep_autopsy_data(folders, msd_type, specie)
            # Plot msds        
            if labels:
                output_name = f"./{out_folder_name}/{specie}_{msd_type}"
                plot_msd(msds, ds, times, labels, output_name, skip_initial_indices, skip_final_indices)
    return
    

def grep_autopsy_species(folders):
    folder_labels = folders
    species = []
    for folder in folders:
        data_df = pd.read_csv(f"{folder}/msd_and_ngp.csv", delimiter=',')
        details_df = pd.read_csv(f"{folder}/details.csv", delimiter=',')
        species.extend(list(details_df['species_name']))
    species = list(set(species))
    return species

def grep_autopsy_data(folders, msd_type, specie):    
    msds, diffusion_coeffic, times = [], [], []
    labels = []
    for folder, label in zip(folders, folders):   
        label = label.split('/')[-1]
        data_df = pd.read_csv(f"{folder}/msd_and_ngp.csv", delimiter=',')
        details_df = pd.read_csv(f"{folder}/details.csv", delimiter=',')
        if specie in list(details_df['species_name']):   
            msd = (
                data_df[f'{specie}_xyz_{msd_type}_MSD']) * 1e-20

            msds.append(msd)
            labels.append(f"{specie}@{label}")

            time = np.array(data_df['Time']) * 1e-12
            times.append(time)


    ds = [None]*len(msds)
    output_name = specie
    return msds, ds, times, labels, output_name


def parse_lammps_log(log_content):
    data = {
        "units": None,
        "timestep": None,        
        "atom_style": None,
        "boundary": None,
        "box_abc": None,
        "box_tilt": None,
        "n_atoms": None,
        "i_velocity_T": None,
        "thermo": None,
    }
    lines = log_content.split('\n')
    
    for line in lines:
        #units_match = re.search(r'units\s+(\w+)', line)
        atom_style_match = re.search(r'atom_style\s+(\w+)', line)
        velocity_match = re.search(r'velocity\s+all\s+create\s+(\d+)\s+(\d+)\s+loop\s+(\w+)', line)
        thermo_match = re.search(r'thermo\s+(\d+)', line)
        #timestep_match = re.search(r'timestep\s+(\d+)', line)
        atoms_match = re.search(r'(\d+)\s+atoms\s+in\s+group\s+(\w+)', line)
        fix_match = re.search(r'fix\s+(\w+)\s+all\s+(\w+)\s+(\w+)\s+(\d+)\s+(\d+)\s+(\d+)', line)
        
        if line.startswith("units"):
            data["units"] = line.split()[1]
        elif atom_style_match:
            data["atom_style"] = atom_style_match.group(1)
        elif line.startswith("boundary"):
            data["boundary"] = " ".join(line.split()[1:])
        elif "triclinic box" in line:
            box_data = re.search(r'triclinic box = \((.*?)\) to \((.*?)\) with tilt \((.*?)\)', line)
            if box_data:
                data["box_abc"] = [round(float(i), 2) for i in box_data.group(2).split()]
                data["box_tilt"] = box_data.group(3)
        elif "reading atoms" in line:
            data["n_atoms"] = int(re.search(r'(\d+) atoms', lines[lines.index(line) + 1]).group(1))
        elif velocity_match:
            data["i_velocity_T"] = velocity_match.group(1)
        elif thermo_match:
            data["thermo"] = thermo_match.group(1)
        elif line.startswith("timestep"):
            data["timestep"] = float(line.split()[1])
        
        elif atoms_match:
            atom_count = atoms_match.group(1)
            group_name = atoms_match.group(2)
            data[f"{group_name}_natoms"] = atom_count
        elif fix_match:
            fix_name = fix_match.group(1)
            fix_type = fix_match.group(2)
            fix_param1 = fix_match.group(3)
            fix_param2 = fix_match.group(4)
            fix_param3 = fix_match.group(5)
            data[f"run_{fix_name}"] = {"type": fix_type, "i_temp": fix_param2, "f_temp": fix_param3}
            data['label'] = f"{fix_type}@{fix_param2}"

    if data["units"] == 'metal':
        data["timestep"] *= 1e-12
    elif data["units"] == 'real':
        data["timestep"] *= 1e-15

    return pd.DataFrame([data])

def find_word_in_file(file_path, word_to_find):
    line_numbers = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if word_to_find in line:
                line_numbers.append(line_number)
    return line_numbers, line_number

def process_lammps_log_files(filenames):
    dataframes = []
    for filename in filenames:
        with open(filename, 'r') as file_content:
            df = parse_lammps_log(file_content.read())
            file_path = filename
            starting_word = 'Per MPI rank memory'
            ending_word = 'Loop time of'
            begin_line_numbers, _ = find_word_in_file(file_path, starting_word)
            end_line_numbers, _ = find_word_in_file(file_path, ending_word)
            if begin_line_numbers:
                if len(begin_line_numbers) == len(end_line_numbers)+1:
                    end_line_numbers.append(_-5)
            df_1 = pd.DataFrame()
            with open(file_path, 'r') as file_data:
                data_ = file_data.readlines()
                for i_l, f_l in zip(begin_line_numbers, end_line_numbers):
                    data = [data_.split() for data_ in data_[i_l:f_l-1]]
                    df_ = pd.DataFrame(data[1:], columns=data[0]) 
                    if df['units'][0] == 'metal':
                        for col in data[0]:
                            if 'Time' == col:
                                df_[col] = np.array(df_[col], dtype=float) * 1e-12
                            elif 'c_msd' in col:
                                df_[col] = np.array(df_[col], dtype=float) * 1e-20
                            elif 'v_d' in col:
                                df_[col] = np.array(df_[col], dtype=float) * 1e-8                          
                    elif df['units'][0] == 'real':
                        for col in data[0]:
                            if 'Time' == col:
                                df_[col] = np.array(df_[col], dtype=float) * 1e-15
                            elif 'c_msd' in col:
                                df_[col] = np.array(df_[col], dtype=float) * 1e-20
                            elif 'v_d' in col:
                                df_[col] = np.array(df_[col], dtype=float) * 1e-5 
                                
                                
                    df_1 = pd.concat([df_1, df_])
                    y_labels = data[0]

            for column in y_labels:
                if 'c_msd' in column:
                    column1 = f"msd_{column[5:-3]}"
                elif 'v_d' in column:
                    column1 = f"D_{column[3:]}"  
                elif 'Temp' == column:
                    column1 = f"T (K)"  
                elif 'Press' == column:
                    if df['units'][0] == 'real':
                        column1 = f"P (atm)"  
                    elif df['units'][0] == 'metal':
                        column1 = f"P (bars)"       
                elif 'PotEng' == column:
                    column1 = f"E_pot (eV)"  
                elif 'TotEng' == column:
                    column1 = f"E_tot (eV)"  
                elif 'Step' == column:
                    column1 = f"n_steps"  
                elif 'Time' == column:
                    column1 = f"Time" 
             
                else:
                    column1 = column               
                df[column1] = [np.array(df_1[column])]
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)



def smooth_out(array, n_smooth, window_size, poly_order):
    smoothed_array = array
    for i_ in range(n_smooth):
        smoothed_array = savgol_filter(smoothed_array, window_size, poly_order)
        window_size += 1
    return smoothed_array
    

def interpolate_color(color1, color2, num_steps):
    rgb1 = np.array(color1)
    rgb2 = np.array(color2)
    step = 1 / (num_steps - 1)
    interpolated_colors = []
    for i in range(num_steps):
        interpolated_color = tuple((1 - i * step) * rgb1 + i * step * rgb2)
        interpolated_colors.append(interpolated_color)
    return interpolated_colors

def rgb_to_hex(rgb):
    # Convert RGB values to integers
    rgb_int = tuple(int(x) for x in rgb)
    # Format the RGB values into a hex string
    return '#{:02x}{:02x}{:02x}'.format(*rgb_int)
    
def generate_color_gradient():  
    n_colors = 20
    green = '#2E7F18'
    red = '#C82538'
    blue = '#26547C'
    green = (46, 127, 24)  # RGB values of green
    red = (200, 37, 56)    # RGB values of red
    blue = (38, 84, 124)   # RGB values of blue
    
    colors = [green, red, blue]
    n = len(colors)
    color_gradient = []
    for i in range(n - 1):
        gradient = interpolate_color(colors[i], colors[i + 1], n_colors)
        color_gradient.extend(gradient)        
    return [rgb_to_hex(rgb) for rgb in color_gradient]



def select_colors(colors, n):
    values = colors
    if n >= len(values):
        return values
        
    if n > 1:
        step_size = (len(values) - 1) / (n - 1)
        selected_colors = [values[int(i * step_size)] for i in range(n)]
    if n == 1:
        selected_colors = [colors[0]]
    return selected_colors


def plot_msd(msds, diff_co, times, labels, output_name, skip_initial_indices, skip_final_indices):  

    # Sample colors for different datasets
    colors = select_colors(generate_color_gradient(), len(labels)) 
    fig = make_subplots(rows=2, 
                        cols=2, 
                        shared_xaxes=False, 
                        vertical_spacing=0.15, 
                        horizontal_spacing=0.15,
                        insets=[dict(cell=(2,1), b=0.45, l = 0.45, w=0.45, h= 0.45)],
                        specs=[[{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]],
                       )
    
    legend_length = 0
    for msd, d, c, label, time_ in zip(msds, diff_co, colors, labels, times):
        window_size = int(len(msd)*0.05)
        poly_order = 1
        n_smooth = 2
        slop_interval = int(len(msd)*0.1)
        legend_length += len(label)
        # Apply Savitzky-Golay filter to smooth the noisy data
        msd_ = msd[skip_initial_indices:skip_final_indices]
        smoothed_msd_ = smooth_out(msd, n_smooth, window_size, poly_order)
        smoothed_msd = smoothed_msd_[skip_initial_indices:skip_final_indices]        
        time = np.array(time_)[skip_initial_indices:skip_final_indices]

        #################################
        #Original MSD 
        #################################
        c_ = tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) + (0.5,)  # Add 0.5 for 50% opacity
        fig.add_trace(go.Scatter(x=time, 
                                 y=msd_, 
                                 line=dict(color=f'rgba{c_}'),
                                 name = label, 
                                 showlegend=False), 
                      row=1, 
                      col=1)   

        #################################
        # Smoothed MSD       
        #################################
        fig.add_trace(go.Scatter(x=time, 
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
            fig.add_trace(go.Scatter(x=time, 
                                     y=d[skip_initial_indices:skip_final_indices], 
                                     line=dict(color=c),
                                     name = label, 
                                     showlegend=False), 
                          row=2, 
                          col=2)

        elif d == None: 
            interval = slop_interval         
            #slops = [(msd[i+interval] - msd[i])/(time[i+interval]-time[i]) for i in range(len(time)-interval)]   
            slops = [(smoothed_msd[i+interval] - smoothed_msd[i])/(time[i+interval]-time[i]) for i in range(len(time)-interval)]   
            diff_co = np.array(slops)/6
            s_diff_co = smooth_out(diff_co, n_smooth*3, window_size*2, poly_order)
            i_n= int(len(s_diff_co)*0.25)
            f_n= int(len(s_diff_co)*0.75)
    
            # Fit MSD to a linear function to get the slope
            coefficients = np.polyfit(time, smoothed_msd, 1)
            slope = coefficients[0]
            
            # Compute diffusion coefficient
            D = slope / 6
            print(f"{label} D: {D:.1e} m^2/s")
            fig.add_trace(go.Scatter(x=time, 
                                     y=diff_co[:-interval], 
                                     line=dict(color=c),
                                     name = label, 
                                     showlegend=False), 
                          row=2, 
                          col=2)
        

        

        #################################
        # Plot log(x) and log(y) in the second subplot
        #################################
        logy = np.log10(smoothed_msd)
        logx = np.log10(time)
        
        #smoothed_log_msd = smooth_out(np.log10(msd), n_smooth, window_size, poly_order)
        fig.add_trace(go.Scatter(x=logx, #time, 
                                 y=logy, #smoothed_msd, 
                                 line=dict(color=c), 
                                 showlegend=False), 
                      row=1, col=2)
    
        # Calculate the numerical derivative
        # Calculate the derivative with intervals of 10 values

        dy_dx = []
        interval = slop_interval
        for i in range(interval, len(logy)):
            dy_dx.append((logy[i]-logy[i-interval])/(logx[i]-logx[i-interval]) )

        #smoothed_dy_dx = smooth_out(np.array(dy_dx), n_smooth, window_size, poly_order)
        #interval = 1
        #dy_dx = derivative_with_interval(np.log(y), np.log(x*1e-15), interval)
        
        # Plot x_derivative and dy_dx in the first subplot
        fig.add_trace(go.Scatter(x=time[:-interval], 
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
        # Finding slop
        try:    
            time_sel = (max(x[(1.1>dy_dx)&(0.9<dy_dx)]) - min(x[(1.1>dy_dx)&(0.9<dy_dx)])) * 1e-9
            msd_sel = (max(y[(1.1>dy_dx)&(0.9<dy_dx)]) - min(y[(1.1>dy_dx)&(0.9<dy_dx)])) * 1e-16
            slop =  msd_sel/time_sel
            sigma = inoic_condictivity(slop)
        except:
            #print(" ")
            pass
    

    
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
    fig.update_xaxes(title_text="Time(s)", row=1, col=1, title_standoff=0)
    fig.update_yaxes(title_text="MSD (m<sup>2</sup>)", row=1, col=1)  # Å title_standoff=0
    fig.update_xaxes(title_text="Time(s)", row=1, col=2, title_standoff=0) #type="log",
    fig.update_yaxes(title_text="MSD (m<sup>2</sup>)", row=1, col=2, title_standoff=0) #title_standoff=0
    fig.update_xaxes(title_text="Time(s)", row=2, col=1, title_standoff=7)
    fig.update_yaxes(title_text="β", row=2, col=1) #range=[0, 4.5] range=[-0.9, 4.9]
    fig.update_xaxes(title_text="Time(s)", row=2, col=2, title_standoff=7)
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
    
    fig.update_layout(legend= {'itemsizing': 'constant'})
    # Assuming fig is your Plotly figure object
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    # Save the plot to an HTML file
    fig.write_html(f"{output_name}_msd.html")
    fig.write_image(f"{output_name}_msd.png")
    
    # Display the plot
    return fig.show()


