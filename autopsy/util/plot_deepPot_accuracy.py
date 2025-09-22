import numpy as np
import pandas as pd
import os, re, glob
from sklearn.metrics import r2_score
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def n_col_row(n_atoms_type):
    # Map number of atom types to (columns, rows, font_size)
    layout_map = {
        1: (2, 1, 18), 2: (3, 1, 20), 3: (2, 2, 18),
        4: (3, 2, 20), 5: (3, 2, 20), 6: (4, 2, 20), 7: (4, 2, 20),
        8: (3, 3, 20), 9: (4, 3, 20), 10: (4, 3, 20), 11: (4, 3, 20),
        12: (3, 5, 20), 13: (3, 5, 20), 14: (3, 6, 20), 15: (3, 6, 20), 16: (3, 6, 20)
    }
    return layout_map.get(n_atoms_type, (2, 1, 16))

def check_files():
    directory_path = './'
    file_force = os.path.join(directory_path, 'results.f.out')
    file_energy = os.path.join(directory_path, 'results.e_peratom.out')
    for file_ in (file_force, file_energy):
        if not os.path.exists(file_):
            print(f"Missing file: {file_}")
            exit(1)
    valid_folders = []
    for base_dir in ["./", "../00.data/validation_data"]:
        for d in [base_dir] if base_dir == "./" else sorted(glob.glob(f"{base_dir}/*")):
            atoms_file = os.path.join(d, 'type.raw')
            indices_file = os.path.join(d, 'type_map.raw')
            if os.path.isfile(atoms_file) and os.path.isfile(indices_file):
                valid_folders.append((indices_file, atoms_file))
    if not valid_folders:
        print("No valid folders with both 'type.raw' and 'type_map.raw' found.")
        exit(1)
    return file_force, file_energy, valid_folders

def parse_sectioned_file(filepath, num_cols):
    sections, current_key, current_data = {}, None, []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if current_key and current_data:
                    sections[current_key] = pd.DataFrame(current_data)
                    current_data = []
                match = re.search(r'#\s+(.*?):', line)
                current_key = os.path.basename(match.group(1)) if match else None
            elif line:
                values = line.split()
                for i in range(0, len(values), num_cols):
                    chunk = values[i:i + num_cols]
                    if len(chunk) == num_cols:
                        current_data.append(list(map(float, chunk)))
        if current_key and current_data:
            sections[current_key] = pd.DataFrame(current_data)
    return sections

def axis_details(axis_range, axis_label, font_size, standoff):
    min_val, max_val = axis_range
    major_ticks = np.array([
        round(min_val + 0.25 * (max_val - min_val)),
        round(min_val + 0.50 * (max_val - min_val)),
        round(min_val + 0.75 * (max_val - min_val))
    ])
    dtick = major_ticks[1] - major_ticks[0]
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
    legend_x = 1 - (1 / n_columns) * 0.01
    legend_y = (1 / n_rows) * 0.01
    fig.update_layout(
        legend=dict(
            orientation='v',
            xref='paper',
            yref='paper',
            xanchor='right',
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

def plot_deepPot_accuracy():
    file_force, file_energy, valid_folders = check_files()
    energy_sections = parse_sectioned_file(file_energy, num_cols=2)
    force_sections = parse_sectioned_file(file_force, num_cols=6)
    all_force_data, all_energy_data = [], []
    folder_lookup = {
        os.path.basename(os.path.dirname(path_map)): (path_map, path_type)
        for (path_map, path_type) in valid_folders
    }

    # Open the r2.txt file for writing
    r2_file =  open("r2.txt", "w")
    # Write header with the specific format
    r2_file.write("-" * 73 + "\n")
    r2_file.write(" " * 52 + "R²" + " " * 11 + "\n")
    r2_file.write("Dataset".center(30) + " " + "-" * 39 + "\n")
    r2_file.write(" " * 33 + "Energy".center(10))
    r2_file.write("Fx".center(10))
    r2_file.write("Fy".center(10))
    r2_file.write("Fz".center(10) + "\n")
    r2_file.write("-" * 73 + "\n")

    s_no = 1
    for dataset_name in energy_sections.keys():
        if dataset_name not in folder_lookup:
            print(f"⚠️  Warning: Dataset '{dataset_name}' not found in type files.")
            continue
        atomic_indices_file, atoms_file = folder_lookup[dataset_name]
        print(f"[{s_no:>2}] {dataset_name}")
        s_no += 1
        with open(atoms_file, 'r') as f_atoms, open(atomic_indices_file, 'r') as f_indices:
            indices = f_atoms.read().splitlines()
            atom_names = f_indices.read().splitlines()
            atoms_ = atom_names
        atoms = [atom_names[int(idx)] for idx in indices]
        df_e = energy_sections[dataset_name]
        df_f = force_sections[dataset_name]
        df_f.columns = ['DFT Fx', 'DFT Fy', 'DFT Fz', 'DLP Fx', 'DLP Fy', 'DLP Fz']
        df_e.columns = ['DFT E', 'DLP E']
        repeat_count = len(df_f) // len(atoms)
        remainder = len(df_f) % len(atoms)
        df_f['atoms'] = atoms * repeat_count + atoms[:remainder]
        df_f['dataset'] = dataset_name
        df_e['dataset'] = dataset_name

        # Calculate R² for energy
        r2_energy = r2_score(df_e['DFT E'], df_e['DLP E'])
        
        # Calculate R² for forces components
        r2_fx = r2_score(df_f['DFT Fx'], df_f['DLP Fx'])
        r2_fy = r2_score(df_f['DFT Fy'], df_f['DLP Fy'])
        r2_fz = r2_score(df_f['DFT Fz'], df_f['DLP Fz'])
        
        # Write all R² values in the specific format
        r2_file.write(f"{dataset_name:<30}")
        r2_file.write(f"{r2_energy:>10.2f}")
        r2_file.write(f"{r2_fx:>10.2f}")
        r2_file.write(f"{r2_fy:>10.2f}")
        r2_file.write(f"{r2_fz:>10.2f}\n")

        if r2_energy < 0.95:
            print(f"   ❌ Skipped due to lower R² = {r2_energy:.4f} than 0.95")
            continue
        all_force_data.append(df_f)
        all_energy_data.append(df_e)
  
    # Add final border line
    r2_file.write("-" * 73 + "\n")
    r2_file.close()
    force_all_full = pd.concat(all_force_data, ignore_index=True)
    data_e = pd.concat(all_energy_data, ignore_index=True)
    # Sample 10,000 per atom type
    sampled_force_data = [
        atom_df.sample(n=min(10000, len(atom_df)), random_state=42)
        for atom_type in sorted(force_all_full['atoms'].unique())
        for atom_df in [force_all_full[force_all_full['atoms'] == atom_type]]
    ]
    data_f = pd.concat(sampled_force_data, ignore_index=True)
    x_name = ['DFT Fx', 'DFT Fy','DFT Fz']
    y_name = ['DLP Fx', 'DLP Fy','DLP Fz']
    legends = [r"$F_x$", r"$F_y$", r"$F_z$"]
    data_f.columns = x_name + y_name + ['atoms', 'dataset']
    data_e.columns = ['DFT E', 'DLP E', 'dataset']
    n_atoms_type = len(atoms_)
    n_columns, n_rows, font_size = n_col_row(n_atoms_type)
    fig = make_subplots(rows=n_rows, cols=n_columns,
                        shared_xaxes=False,
                        vertical_spacing=0.35 / n_rows,
                        horizontal_spacing=0.35 / n_columns)
    colors = ['green', 'red', 'blue']
    subplot_i = 1
    sub_labels = ['', '(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)', '(l)', '(m)', '(n)']
    ex_name, ey_name = [r"$E_\mathrm{DFT}\ (\mathrm{eV})$", r"$E_\mathrm{DLP}\ (\mathrm{eV})$"]
    fx_name, fy_name = [r"$F_\mathrm{DFT}\ (\mathrm{eV/Å})$", r"$F_\mathrm{DLP}\ (\mathrm{eV}/\text{Å})$"]
    n_atoms_ = 0
    for n_row in range(1, n_rows + 1):
        for n_col in range(1, n_columns + 1):
            xaxis_key = f'xaxis{subplot_i}' if subplot_i > 1 else 'xaxis'
            yaxis_key = f'yaxis{subplot_i}' if subplot_i > 1 else 'yaxis'
            if n_row == 1 and n_col == 1:
                print("Plotting Energy...")
                e0 = min(np.array(data_e.drop(columns=['dataset']).min()))
                x_e = data_e['DFT E'] - e0
                y_e = data_e['DLP E'] - e0
                emax = max(np.array(data_e.drop(columns=['dataset']).max())) - e0
                erange = np.array([-0.000001, emax]) * 1.1
                fig.append_trace(
                    go.Scatter(
                        x=x_e,
                        y=y_e,
                        mode="markers",
                        marker=dict(color=colors[0], size=3),
                        showlegend=False,
                    ),
                    row=n_row,
                    col=n_col,
                )
                fig.append_trace(
                    go.Scatter(
                        x=erange,
                        y=erange,
                        mode="lines",
                        line=dict(dash='dot', width=1),
                        marker=dict(color='black', size=0),
                        showlegend=False,
                    ),
                    row=n_row,
                    col=n_col,
                )
                fig.update_layout(**{xaxis_key: axis_details(erange, ex_name, font_size, 1)})
                fig.update_layout(**{yaxis_key: axis_details(erange, ey_name, font_size, 1)})
                add_subplot_annotation(fig, n_row, n_col, sub_labels[subplot_i], erange, font_size)
                subplot_i += 1
            else:
                showlegend = (n_row == 1 and n_col == 2)
                try:
                    data_force = data_f[data_f['atoms'] == atoms_[n_atoms_]].drop(columns=['atoms', 'dataset'])
                    plot_this = True
                except:
                    plot_this = False
                    #data_force[x_name] = np.array([0.0]*10), np.array([0.0]*10), np.array([0.0]*10)
                    #data_force[y_name] = np.array([0.0]*10), np.array([0.0]*10), np.array([0.0]*10)
                if plot_this:
                    fmin = min(np.array(data_force.min()))
                    fmax = max(np.array(data_force.max()))
                    fran = max(-fmin, fmax)
                    frange = np.array([-fran, fran]) * 1.1
                    for column_i, (column_x, column_y) in enumerate(zip(x_name, y_name)):
                        y_true = data_force[column_x].astype(float)
                        y_pred = data_force[column_y].astype(float)
                        print(f"Plotting Forces of {atoms_[n_atoms_]}...")
                        fig.append_trace(
                            go.Scatter(
                                x=y_true,
                                y=y_pred,
                                mode="markers",
                                name=legends[column_i],
                                marker=dict(color=colors[column_i], size=2),
                                showlegend=showlegend,
                            ),
                            row=n_row,
                            col=n_col,
                        )
                        fig.append_trace(
                            go.Scatter(
                                x=frange,
                                y=frange,
                                mode="lines",
                                line=dict(dash='dot', width=1),
                                marker=dict(color='black', size=0),
                                showlegend=False,
                            ),
                            row=n_row,
                            col=n_col,
                        )
                    sub_label = f"{sub_labels[subplot_i]} {atoms_[n_atoms_]}"
                    fig.update_layout(**{xaxis_key: axis_details(frange, fx_name, font_size, 1)})
                    fig.update_layout(**{yaxis_key: axis_details(frange, fy_name, font_size, 1)})
                    add_subplot_annotation(fig, n_row, n_col, sub_label, frange, font_size)
                subplot_i += 1
                n_atoms_ += 1
    update_figure_legends(fig, n_columns, n_rows, font_size)

    # Write the plots to HTML, SVG, and PNG files
    #print("Writting HTML image")
    #fig.write_html('force_accuracy.html')
    #print("Writing SVG image")
    #fig.write_image('force_accuracy.svg',
    #                scale=2,
    #                width=width,
    #                height=height)
    #print("Writing PNG image")
    fig.write_image('force_accuracy.png', scale=3)

    return
