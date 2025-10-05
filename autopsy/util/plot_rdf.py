import warnings

# Filter and ignore specific warnings (e.g., DeprecationWarning)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import ase.io
from ase.build import molecule
from ase.geometry.analysis import Analysis
import os, sys
from os import listdir
from os.path import isfile, join
import re
import glob

from collections import Counter

from ase.build import make_supercell
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ase.data import atomic_numbers, atomic_names, atomic_masses, covalent_radii


def n_col_row(n_atoms_type):
    # Map number of atom types to (columns, rows, font_size)
    fs = 16
    layout_map = {
        1: (2, 1, fs), 2: (2, 2, fs), 3: (2, 2, fs),
        4: (3, 2, fs), 5: (3, 2, fs), 6: (4, 2, fs), 7: (4, 2, fs),
        8: (3, 3, fs), 9: (4, 3, fs), 10:(4, 3, fs), 11:(4, 3, fs),
        12:(3, 5, fs), 13:(3, 5, fs), 14:(3, 6, fs), 15:(3, 6, fs), 16: (3, 6, fs)
    }
    return layout_map.get(n_atoms_type, (2, 1, fs))

def axis_details(axis_range, axis_label, font_size, standoff):
    min_val, max_val = axis_range
    
    # Calculate the range and determine optimal tick spacing
    axis_range_val = max_val - min_val
    
    # Choose a nice tick step that gives 3-5 major ticks
    # Common nice steps: 1, 2, 5, 10, 20, 25, 50, etc.
    possible_steps = [1, 2, 5, 10, 20, 25, 50, 100]
    target_num_ticks = 4  # Aim for 4 major ticks (including start and end)
    
    # Find the best step size
    ideal_step = axis_range_val / (target_num_ticks - 1)
    tick_step = min(possible_steps, key=lambda x: abs(x - ideal_step))
    
    # Ensure tick_step is at least 1 for reasonable spacing
    tick_step = max(tick_step, 1)
    
    # Calculate major ticks
    first_tick = np.ceil(min_val / tick_step) * tick_step
    last_tick = np.floor(max_val / tick_step) * tick_step
    major_ticks = np.arange(first_tick, last_tick + tick_step, tick_step)
    
    # If we don't have enough ticks, adjust
    if len(major_ticks) < 3:
        # Use exactly 4 major ticks with equal spacing
        major_ticks = np.linspace(min_val, max_val, 4)
        major_ticks = np.round(major_ticks).astype(int)
    else:
        major_ticks = major_ticks.astype(int)
    
    dtick = tick_step
    minor_tick_step = dtick / 2
    first_minor = major_ticks[0] + minor_tick_step
    
    return dict(
        range=axis_range,
        title_text=axis_label,
        title_standoff=standoff,
        showline=True,
        linecolor='black',
        linewidth=2,
        mirror=True,
        ticks='inside',
        tickwidth=2,
        ticklen=10,
        tickcolor='black',
        tickfont=dict(size=font_size + 2, color='black', family="Times New Roman, serif"),
        gridcolor='lightgray',
        griddash='dash',
        tickmode='array',
        tickvals=major_ticks.tolist(),
        minor=dict(
            tickmode='linear',
            tick0=first_minor,
            dtick=minor_tick_step,
            ticks='inside',
            ticklen=5,
            tickwidth=1,
            tickcolor='black',
            showgrid=False
        )
    )

def update_figure_legends(fig, n_columns, n_rows, font_size):
    legend_x = 0.1 #1 - (1 / n_columns) * 0.01
    legend_y = 1.00 #(1 / n_rows) * 0.01
    fig.update_layout(
        legend=dict(
            orientation='h',
            xref='paper',
            yref='paper',
            xanchor='left',
            yanchor='bottom',
            x=legend_x,
            y=legend_y,
            font=dict(size=font_size + 3, color='black', family="Times New Roman, serif"),
            bgcolor='rgba(0,0,0,0)',
            itemsizing='constant',
            itemwidth=30,
            traceorder='normal'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, b=10, t=10),
        font=dict(size=font_size + 2, color='black', family="Times New Roman, serif"),
        height=250 * n_rows, width=250 * n_columns,
    )

def add_subplot_annotation(fig, n_row, n_col, sub_label, plot_range, font_size):
    fig.add_annotation(
        text=sub_label,
        yanchor="top",
        xanchor="left",
        xref="paper",
        yref="paper",
        x=plot_range[0],
        y=plot_range[1],
        showarrow=False,
        font=dict(size=font_size + 5, color='black', family="Times New Roman, serif"),
        row=n_row,
        col=n_col
    )

def convert_to_subscript(formula):
    """Convert chemical formula with numbers to subscript format"""
    import re
    
    # Use regex to find numbers and replace them with subscript
    subscript_formula = re.sub(r'(\d+)', r'<sub>\1</sub>', formula)
    return subscript_formula

def check_files():
    """Check for structure files in current directory"""
    directory_path = './'
    common_files = []
    for f in os.listdir(directory_path):
        if (f.endswith('.vasp') or f.endswith('.POSCAR') or f.endswith('.CONTCAR') or 
            f.endswith('.lmp') or f.endswith('.lammps') or f.endswith('.cif') or 
            f.endswith('.xyz') or f.endswith('.traj')):
            common_files.append(f)
    
    if not common_files:
        print("No structure files found!")
        print("Supported formats: .vasp, .POSCAR, .CONTCAR, .lmp, .lammps, .cif, .xyz, .traj")
        exit(1)
    
    return common_files

def read_structure_files(filenames):
    """Read structure files and return trajectory data"""
    traj_data, symbols, legends = [], [], []
    
    for trajectory_file in filenames:
        print(f"Reading {trajectory_file}...")
        try:
            # Try to read as trajectory first
            if trajectory_file.endswith('.lmp') or trajectory_file.endswith('.lammps'):
                traj = ase.io.read(trajectory_file, format="lammps-dump-text", index=":")
            else:
                traj = ase.io.read(trajectory_file, index=":")
            
            # If single structure, convert to list
            if not isinstance(traj, list):
                traj = [traj]
                
        except:
            # If trajectory reading fails, try as single structure
            try:
                traj = [ase.io.read(trajectory_file)]
            except Exception as e:
                print(f"Error reading {trajectory_file}: {e}")
                continue
        
        # Extract symbols
        all_symbols = [list(tra.symbols) for tra in traj]
        symbols.append(list(set(np.array(all_symbols).flatten())))
        traj_data.append(traj)
        legends.append(os.path.splitext(trajectory_file)[0])
    
    return traj_data, symbols, legends

def supercell(data, rMax):
    """Create supercell if the cell is too small for RDF calculation"""
    rMax_needed = rMax * 2  # Need larger cell for proper RDF
    cell = data.get_cell()
    lengths = cell.lengths()
    
    # Calculate required repetitions
    repetitions = np.ceil(rMax_needed / lengths).astype(int)
    
    # Ensure at least 2x2x2 for meaningful RDF
    repetitions = np.maximum(repetitions, [2, 2, 2])
    
    print(f"Creating {repetitions[0]}x{repetitions[1]}x{repetitions[2]} supercell")
    
    multiplier = np.diag(repetitions)
    data_super = make_supercell(data, multiplier)
    
    return data_super

def count_name(name_list, target_name):
    """Return the count of target_name in the list"""
    return name_list.count(target_name)

def calculate_rdf_data(traj_data, rMax=10, nBins=100):
    """Calculate RDF data for all element pairs"""
    all_rdf_data = []
    
    for legend_no, traj in enumerate(traj_data):
        print(f"No. of frames: {len(traj)}")
        
        # Create supercells if needed for proper RDF calculation
        for traj_i in range(len(traj)):
            cell = traj[traj_i].get_cell()
            if np.any(cell.lengths() < rMax * 2):
                traj[traj_i] = supercell(traj[traj_i], rMax)
       
        ana = Analysis(traj)
        
        # Get all unique symbols from this trajectory
        symbols = []
        for tra in traj:
            symbols.extend(list(tra.symbols))
        symbols = list(set(symbols))
        
        # Calculate RDF for all element pairs
        rdf_pairs = []
        for e_i, element1 in enumerate(symbols):
            for e_j, element2 in enumerate(symbols):
                if e_j >= e_i:
                    atomic_number1 = atomic_numbers[element1]
                    atomic_number2 = atomic_numbers[element2]

                    rdf = ana.get_rdf(rMax, nBins, imageIdx=None, 
                                    elements=(atomic_number1, atomic_number2))[0]

                    x = (np.arange(nBins) + 0.5) * rMax / nBins
                    
                    rdf_pairs.append({
                        'element1': element1,
                        'element2': element2,
                        'x': x,
                        'y': rdf,
                        #'legend': traj_data[1][legend_no] if len(traj_data) > 1 else "RDF"
                    })
        
        all_rdf_data.append({
            #'legend': f"Dataset_{legend_no+1}",
            'rdf_pairs': rdf_pairs,
            'symbols': symbols
        })
    
    return all_rdf_data

def plot_rdf():
    # Check for structure files
    filenames = []
    for i in range(1, 5):
        try:
            filenames.append(sys.argv[i])
        except:
            if i == 1:
                # No arguments provided, search for files
                filenames = check_files()
                print("Found these structure files:", filenames)
                break
            else:
                pass
    
    if not filenames:
        filenames = check_files()
    
    # Read structure files
    traj_data, symbols, legends = read_structure_files(filenames)
    
    if not traj_data:
        print("No valid structure files could be read!")
        return
    
    # Calculate RDF data
    all_rdf_data = calculate_rdf_data(traj_data)
    
    # Get all unique element pairs across all datasets
    all_element_pairs = set()
    for dataset in all_rdf_data:
        for pair in dataset['rdf_pairs']:
            all_element_pairs.add((pair['element1'], pair['element2']))
    
    all_element_pairs = sorted(list(all_element_pairs))
    no_of_subplots = len(all_element_pairs)
    
    print("No. of subplots: ", no_of_subplots)
    print("RDF pairs:", all_element_pairs)

    # Get layout configuration
    n_columns, n_rows, font_size = n_col_row(no_of_subplots)
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_columns,
        shared_xaxes=False,
        vertical_spacing=0.35 / n_rows,
        horizontal_spacing=0.35 / n_columns
    )
    # Subplot labels
    sub_labels = ['', '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
    
    # Colors for different datasets
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # Track which legends have been shown
    shown_legends = set()
    
    # Plot RDF data
    subplot_i = 1
    for row in range(1, n_rows + 1):
        for col in range(1, n_columns + 1):
            if subplot_i <= no_of_subplots:
                element1, element2 = all_element_pairs[subplot_i - 1]
                xaxis_key = f'xaxis{subplot_i}' if subplot_i > 1 else 'xaxis'
                yaxis_key = f'yaxis{subplot_i}' if subplot_i > 1 else 'yaxis'
    
                # Plot RDF for each dataset
                for dataset_idx, dataset in enumerate(all_rdf_data):
                    # Find the corresponding RDF pair in this dataset
                    rdf_data = None
                    for pair in dataset['rdf_pairs']:
                        if pair['element1'] == element1 and pair['element2'] == element2:
                            rdf_data = pair
                            break
                    
                    if rdf_data is not None:
                        legend_name = legends[dataset_idx] if dataset_idx < len(legends) else f"Dataset_{dataset_idx+1}"
                        
                        # Convert to subscript format
                        formatted_legend = convert_to_subscript(legend_name)
 
                        # Show legend only for the first occurrence of each dataset
                        showlegend = legend_name not in shown_legends
                        if showlegend:
                            shown_legends.add(legend_name)
                        
                        color = colors[dataset_idx % len(colors)]
                        fig.add_trace(
                            go.Scatter(
                                x=rdf_data['x'],
                                y=rdf_data['y'],
                                mode='lines',
                                name=formatted_legend,
                                line=dict(color=color, width=1),
                                showlegend=showlegend,
                            ),
                            row=row, col=col
                        )         
                # Set axis ranges and labels
                xaxis_range = [0.9, 7.2]  # RDF typically from 1Å to 10Å
                y_max = max([max(pair['y']) for dataset in all_rdf_data 
                           for pair in dataset['rdf_pairs'] 
                           if pair['element1'] == element1 and pair['element2'] == element2], default=3)
                yaxis_range = [0, y_max * 1.1]
                sub_annotate_range = [xaxis_range[0]*1.5, yaxis_range[1]]
                
                # Update x-axis
                fig.update_layout(**{
                    xaxis_key: axis_details(
                        xaxis_range, 
                        "Distance (Å)", 
                        font_size, 
                        1
                    )
                })
                
                # Update y-axis  
                fig.update_layout(**{
                    yaxis_key: axis_details(
                        yaxis_range,
                        f"RDF", # ({element1}-{element2})",
                        font_size,
                        1
                    )
                })
                
                # Add subplot annotation
                if subplot_i < len(sub_labels):
                    add_subplot_annotation(
                        fig, row, col, 
                        f"{sub_labels[subplot_i]} {element1}-{element2}", 
                        sub_annotate_range,  # Position in subplot
                        font_size
                    )
                
                subplot_i += 1

    # Update figure layout and legends
    update_figure_legends(fig, n_columns, n_rows, font_size)
    
    # Save the figure
    fig.write_image('rdf.png', scale=3)
    fig.write_html('rdf.html')
    print("RDF plot saved as 'rdf.png' and 'rdf.html'")
    
    return
