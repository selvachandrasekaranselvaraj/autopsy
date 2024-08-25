
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from autopsy.util.sort_atomic_indices import sort_atomic_indices
from autopsy.ngp.calc_ngp import calc_ngp
from autopsy.msd.calc_msd_total import calc_msd_total
from autopsy.cross_msd.calc_msd_cross import calc_msd_cross
from autopsy.msd.calc_msd_self import calc_msd_self
from autopsy.vhf.calc_vhf import calc_vhf

class PropertyCalculator:
    '''
    PropertyCalculator class is responsible for calculating various properties based on trajectory data. 
    PropertyCalculator calculates and stores various properties based on trajectory data. 
    The class includes methods for calculating ionic movements, VHF, NGP, MSD, cross MSD, and writing the results to CSV files.
    
    # Example usage:
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

    def __init__(self, data):
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
        self.symbols = np.array(list(data[0].symbols))        
        self.atoms_list = np.sort(list(set(self.symbols)))        
        self.cell = np.linalg.norm(data[-1].cell, axis=1)        
        self.df = pd.DataFrame()        
        self.axes = ['x', 'y', 'z']        
        self.axis_indices = [[0], [1], [2]]        
        self.df_ionic_movement = pd.DataFrame()


    def calculate_ionic_movements(self, positions, indices, atom):
        '''
        Method to calculate ionic movements for a specific atom.

        :param positions: numpy array, atomic positions for the specified atom
        :param indices: numpy array, sorted indices for the specified atom
        :param atom: str, atomic symbol
        '''
        
        print(f"Writing ionic movements for {atom}...")
        df_ = pd.DataFrame()
        for a_i in indices[:4]:
            df_['x'] = positions[:, a_i, :].T[0]
            df_['y'] = positions[:, a_i, :].T[1]
            df_['z'] = positions[:, a_i, :].T[2]
            df_['atom'] = [atom] * len(positions[:, a_i, :].T[2])
            self.df_ionic_movement = pd.concat([self.df_ionic_movement, df_], axis=0)
        print("Writing ionic movements is DONE")

    def calculate_vhf(self, positions, indices, atom):
        '''
        Method to calculate VHF (Van Hove Function) and displacement for a specific atom.

        :param positions: numpy array, atomic positions for the specified atom
        :param indices: numpy array, sorted indices for the specified atom
        :param atom: str, atomic symbol
        '''
        print(f"Calculating VHF and displacement of {atom}...")
        n_skipped_atoms = int(len(indices)*0.1)
        displacement = calc_vhf(positions[:, indices[:-n_skipped_atoms], :], atom)
        self.df[f'{atom}_xyz_displacement'] = np.insert(displacement[0], 0, 0)
        self.df[f'{atom}_x_displacement'] = np.insert(displacement[1], 0, 0)
        self.df[f'{atom}_y_displacement'] = np.insert(displacement[2], 0, 0)
        self.df[f'{atom}_z_displacement'] = np.insert(displacement[3], 0, 0)
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
            s_ = positions.shape
            indi_ = indices  # [:30]
            pos_ = positions[:, indi_, axis_i].reshape(s_[0], len(indi_), len(axis_i))
            self.df[f'{atom}_{axis}_total_MSD'] = calc_msd_total(pos_)
            self.df[f'{atom}_{axis}_self_MSD'] = calc_msd_self(pos_)
            self.df[f'{atom}_{axis}_distinct_MSD'] = (
                self.df[f'{atom}_{axis}_total_MSD'] - self.df[f'{atom}_{axis}_self_MSD']
            )

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
                        positions_atom1[i] = self.data[i].positions[
                            np.where(np.array(list(self.data[i].symbols)) == atom1)[0]
                        ]
                        positions_atom2[i] = self.data[i].positions[
                            np.where(np.array(list(self.data[i].symbols)) == atom2)[0]
                        ]

                    for axis, axis_i in zip(self.axes, self.axis_indices):
                        print(f'{atom1}_{atom2}_{axis}')
                        s_a1 = positions_atom1.shape
                        pos_a1 = positions_atom1[:, :, axis_i].reshape(s_a1[0], s_a1[1], len(axis_i))
                        s_a2 = positions_atom2.shape
                        pos_a2 = positions_atom2[:, :, axis_i].reshape(s_a2[0], s_a2[1], len(axis_i))
                        times = np.arange(0, pos_a1.shape[0])
                        self.df[f'{atom1}_{atom2}_{axis}_cross_MSD'] = calc_msd_cross(pos_a1, pos_a2)
                        
                        
    def run(self):
        '''
        Method to perform calculations for various properties.
        '''
        for atom in self.atoms_list:
            
            n_atoms = np.sum(self.symbols == atom)
            positions = np.empty((self.n_frames, n_atoms, 3))
            for i in range(0, len(self.data)):
                positions[i] = self.data[i].positions[np.where(np.array(list(self.data[i].symbols)) == atom)[0]]

            indices = sort_atomic_indices(positions, self.cell)            
                        
            self.calculate_ionic_movements(positions, indices, atom)
            self.calculate_vhf(positions, indices, atom)
            self.calculate_ngp(positions, indices, atom)
            self.calculate_msd(positions, indices, atom)
            
        self.calculate_cross_msd()
        print("DONE")
        print()
        print("Writing outputs")
        self.write_outputs()
        print("******DONE******")                        
                        

    def write_outputs(self):
        '''
        Method to write calculated properties to CSV files.
        '''
        # Save to CSV
        self.df.to_csv('properties.csv', index=False)
        self.df_ionic_movement.to_csv('ionic_movement.csv', index=False)
