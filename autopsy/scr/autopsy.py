import argparse
import logging
import numpy as np
import pandas as pd

from time import time
import sys, os, re

from autopsy.util.sort_atomic_indices import sort_atomic_indices
from autopsy.util.select_center_atoms_indices import select_center_atoms_indices
from autopsy.ngp.calc_ngp import calc_ngp
from autopsy.msd.calc_msd_total import calc_msd_total
from autopsy.cross_msd.calc_msd_cross import calc_msd_cross
from autopsy.msd.calc_msd_self import calc_msd_self
from autopsy.vhf.calc_vhf import calc_vhf
from autopsy.util.find_n_proc import find_optimal_n_proc

class PropertyCalculator:
    '''
    PropertyCalculator class is responsible for calculating various properties based on trajectory data.
    The class includes methods for calculating ionic movements, VHF, NGP, MSD, cross MSD, and writing the results to CSV files.

    Example usage:
    from ase import Atoms
    from ase.calculators.emt import EMT

    # Create a simple trajectory with two frames
    atom1 = Atoms('Cu', positions=[(0, 0, 0)])
    atom2 = Atoms('Cu', positions=[(0.1, 0.2, 0.3)])
    data = [atom1, atom2]

    # Create an instance of PropertyCalculator
    calculator = PropertyCalculator(data)

    # Run the calculations
    calculator.run()

    # The calculated properties are now stored in 'properties.csv' and 'ionic_movement.csv' files
    '''

    def __init__(self, data, stime=None, zmin=None, zmax=None, atoms=None, out_dir=None):
        '''
        Constructor for the PropertyCalculator class.

        param data: list, trajectory data
            A list containing trajectory data, where each element represents a frame in the trajectory.
            Each frame is an instance of the ASE Atoms class, representing the atomic configuration at a specific time.
        n_frames: int
            The number of frames in the trajectory. It is calculated as the length of the provided trajectory data.
        symbols: numpy array
            An array containing the atomic symbols present in the first frame of the trajectory.
            This is used to identify unique atomic symbols in the trajectory.
        atoms_list: numpy array
            An array containing the unique atomic symbols present in the entire trajectory.
        cell: numpy array
            The norm of the cell vectors of the last frame in the trajectory.
            It represents the dimensions of the simulation cell.
        df: pandas DataFrame
            An empty pandas DataFrame that will be used to store calculated properties.
        axes: list of strings
            A list containing the Cartesian coordinates 'x', 'y', and 'z'.
            These are used to specify the axes for which properties are calculated.
        axis_indices: list of lists
            A list containing lists of indices corresponding to 'x', 'y', and 'z'.
            These indices are used to extract specific components of atomic positions.
        df_ionic_movement: pandas DataFrame
            An empty pandas DataFrame that will be used to store ionic movement data.
        '''
        self.data = data
        self.n_frames = len(data)
        self.stime = stime
        self.symbols = np.array(data[0].get_chemical_symbols())
        self.atoms_list = np.sort(list(set(self.symbols))) if atoms is None else np.array(atoms)
        self.cell = np.linalg.norm(data[-1].cell, axis=1)
        self.df = pd.DataFrame()
        self.df['Time'] = stime
        self.df_displacement = pd.DataFrame()
        self.df_displacement['Time'] = stime
        self.axes = ['xyz']  # ['x', 'y', 'z']
        self.axis_indices = np.array([[0, 1, 2]])  # [[0], [1], [2]]
        self.df_ionic_movement = pd.DataFrame()
        self.n_proc = int(find_optimal_n_proc())  # Number of processors for parallel calculations

        self.zmin = zmin
        self.zmax = zmax
        self.out_dir = out_dir

        # Create the directory if it doesn't exist
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)


    def calculate_ionic_movements(self, positions, cell, atom):
        '''
        Method to calculate ionic movements for a specific atom.

        :param positions: numpy array, atomic positions for the specified atom
        :param indices: numpy array, sorted indices for the specified atom
        :param atom: str, atomic symbol
        '''

        print(f"Writing ionic movements for {atom}...")
        indices = select_center_atoms_indices(positions, cell)
        df_ = pd.DataFrame()
        for a_i in indices[:4]:
            df_['x'] = positions[:, a_i, :].T[0]
            df_['y'] = positions[:, a_i, :].T[1]
            df_['z'] = positions[:, a_i, :].T[2]
            df_['atom'] = [atom] * len(positions[:, a_i, :].T[2])
            self.df_ionic_movement = pd.concat([self.df_ionic_movement, df_],
                                               axis=0)

        outfile = f"./{self.out_dir}/ionic_movement.xyz"
        self.df_ionic_movement.to_csv(outfile, index=False)
        print("Writing ionic movements is DONE")

    def calculate_vhf(self, positions, indices, atom):
        '''
        Method to calculate VHF (Van Hove Function) and displacement for a specific atom. This code performs parallelized calculations of Van Hove functions for a given trajectory using ThreadPoolExecutor. It also parallelizes the writing of the results to CSV files. The main function calc_vhf orchestrates these tasks and prints the time taken for parallel CSV writing. The code includes exception handling for better error management during parallel execution.

        :param positions: numpy array, atomic positions for the specified atom
        :param indices: numpy array, sorted indices for the specified atom
        :param atom: str, atomic symbol
        '''
        print(f"Calculating VHF and displacement of {atom}...")
        n_skipped_atoms = int(len(indices) * 0.1)
        displacement = calc_vhf(positions[:, indices[:-n_skipped_atoms], :],
                                atom, self.n_proc)

        for i in range(4):  # for center atoms
            self.df_displacement[f'{atom}{i}_xyz_displacement'] = np.insert(
                displacement[0][i], 0, 0)
            self.df_displacement[f'{atom}{i}_x_displacement'] = np.insert(
                displacement[1][i], 0, 0)
            self.df_displacement[f'{atom}{i}_y_displacement'] = np.insert(
                displacement[2][i], 0, 0)
            self.df_displacement[f'{atom}{i}_z_displacement'] = np.insert(
                displacement[3][i], 0, 0)
        print("VHF is DONE")

    def calculate_ngp(self, positions, indices, atom):
        '''
        Method to calculate NGP (Nearest Neighbor Geometry Parameter) for a specific atom.

        :param positions: numpy array, atomic positions for the specified atom
        :param indices: numpy array, sorted indices for the specified atom
        :param atom: str, atomic symbol
        '''
        print(f"Calculating NGP of {atom}")
        self.df[f'{atom}_xyz_ngp'] = calc_ngp(positions)
        print("NGP is DONE")

    def calculate_msd(self, positions, indices, atom):
        '''
        Method to calculate MSD (Mean Squared Displacement) for a specific atom.

        :param positions: numpy array, atomic positions for the specified atom
        :param indices: numpy array, sorted indices for the specified atom
        :param atom: str, atomic symbol
        '''
        print(f"Calculating MSD of {atom}")
        for axis, axis_i in zip(self.axes, self.axis_indices):
            ix = np.ix_(np.arange(positions.shape[0]), indices, axis_i)
            #pos_ = positions[:, indi_, axis_i].reshape(s_[0], len(indi_), len(axis_i))
            #pos_ = positions[:, :, axis_i].reshape(s_[0], s_[1], len(axis_i))
            pos_ = positions[ix]
            print(pos_.shape)
            self.df[f'{atom}_{axis}_total_MSD'] = calc_msd_total(pos_)
            self.df[f'{atom}_{axis}_self_MSD'] = calc_msd_self(
                pos_, self.n_proc)
            self.df[f'{atom}_{axis}_distinct_MSD'] = (
                self.df[f'{atom}_{axis}_total_MSD'] -
                self.df[f'{atom}_{axis}_self_MSD'])

        print("MSD Calculations are DONE")

    def calculate_cross_msd(self):
        '''
        Method to calculate cross MSD (Mean Squared Displacement) for pairs of atoms.
        '''
        print("Cross MSD")
        for a_i, atom1 in enumerate(self.atoms_list):
            for a_j, atom2 in enumerate(self.atoms_list):
                if a_i < a_j:
                    n_atoms1 = np.sum(self.symbols == atom1)
                    n_atoms2 = np.sum(self.symbols == atom2)
                    positions_atom1 = np.empty((len(self.data), n_atoms1, 3))
                    positions_atom2 = np.empty((len(self.data), n_atoms2, 3))
                    for i in range(0, len(self.data)):
                        positions_atom1[i] = self.data[i].positions[np.where(
                            np.array(list(self.data[i].symbols)) == atom1)[0]]
                        positions_atom2[i] = self.data[i].positions[np.where(
                            np.array(list(self.data[i].symbols)) == atom2)[0]]

                    for axis, axis_i in zip(self.axes, self.axis_indices):
                        print(f'{atom1}_{atom2}_{axis}')
                        s_a1 = positions_atom1.shape
                        pos_a1 = positions_atom1[:, :, axis_i].reshape(
                            s_a1[0], s_a1[1], len(axis_i))
                        s_a2 = positions_atom2.shape
                        pos_a2 = positions_atom2[:, :, axis_i].reshape(
                            s_a2[0], s_a2[1], len(axis_i))
                        times = np.arange(0, pos_a1.shape[0])
                        self.df[
                            f'{atom1}_{atom2}_{axis}_cross_MSD'] = calc_msd_cross(
                                pos_a1, pos_a2)

    def run(self):
        time_i = time()
        '''
        Method to perform calculations for various properties.
        '''
        pos_ = np.array([self.data[0].positions, self.data[-1].positions])
        indices = sort_atomic_indices(pos_, self.cell, self.zmin, self.zmax)
        atoms_list_ = np.array(list(self.data[0].symbols))[indices]
        self.symbols = np.array(list(self.data[0].symbols))

        #self.atoms_list = np.sort(list(set(atoms_list_)))

        #for atom in ['Li']: #self.atoms_list:

        # Create a list to store data for each atom
        data_list = []
        for atom in self.atoms_list:

            n_atoms = np.sum(self.symbols == atom)
            positions = np.empty((self.n_frames, n_atoms, 3))
            for i in range(0, len(self.data)):
                positions[i] = self.data[i].positions[np.where(
                    np.array(list(self.data[i].symbols)) == atom)[0]]
        
            
            indices = sort_atomic_indices(positions, self.cell, self.zmin, self.zmax)

            self.calculate_ionic_movements(positions, self.cell, atom)
            self.calculate_msd(positions, indices, atom)
            #self.calculate_vhf(positions, indices, atom)
            #self.calculate_ngp(positions, indices, atom)

            # Store data in a dictionary
            if self.zmin == None and self.zmax == None:
                zmin_ = 0.0
                zmax_ = round(self.cell[2], 3)
            else:
                zmin_ = self.zmin
                zmax_ = self.zmax
            atom_data = {
                'species_name': atom,
                'n_atoms_tot': n_atoms,
                'n_atoms': len(indices),
                'n_frames': self.n_frames,
                'a': round(self.cell[0], 3),  # Assuming self.cell is a list of a, b, c
                'b': round(self.cell[1], 3),  # Assuming self.cell is a list of a, b, c
                'c': round(self.cell[2], 3),  # Assuming self.cell is a list of a, b, c
                'zmin': zmin_,
                'zmax': zmax_,
                'n_proc': self.n_proc
                # Add more properties if needed
                }
            # Append data to the list
            data_list.append(atom_data)

        # Convert the list of dictionaries to a DataFrame
        self.details_df = pd.DataFrame(data_list)

        #self.calculate_cross_msd()
        print("DONE")
        print()
        #print("Writing outputs")
        self.write_outputs()
        print(f"Total time: {time()-time_i}")
        print("******DONE******")

    def read_in_lammps(self):
        # Path to the input file
        input_file_path = 'in.lammps'
        
        # Initialize an empty dictionary to store variables
        variables = {}
        
        # Read the input file
        with open(input_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('variable'):
                    parts = line.split()
                    var_name = parts[1]
                    #var_value = ' '.join(parts[3:])
                    var_value = parts[3].replace('"', '')
                    variables[var_name] = var_value
                elif "nvt temp" in line and 'temp_init' not in variables:
                    parts = line.split()
                    variables['temp_init'] = parts[5]
                    variables['temp_final'] = parts[6]
        
        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(list(variables.items()), columns=['Variable', 'Value']).T
        df.columns = df.iloc[0]
        df = df.drop(df.index[0]).reset_index(drop=True)
        return df

    def write_outputs(self):
        '''
        Method to write calculated properties to CSV files.
        '''
        # Save to CSV
        out_folder = f"./{self.out_dir}/"
        self.df.to_csv(out_folder+'msd_and_ngp.csv', index=False)

        df_in_lammps = self.read_in_lammps()
        # Concatenate DataFrames column-wise
        duplicates = pd.concat([df_in_lammps] * len(self.details_df), ignore_index=True)
        df_details_combined = pd.concat([self.details_df, duplicates], axis=1)

        df_details_combined.to_csv(out_folder+'details.csv',index=False)
        #self.df_ionic_movement.to_csv('ionic_movement.csv', index=False)
        #self.df_displacement.to_csv('ionic_displacement.csv', index=False)
