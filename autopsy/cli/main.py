import argparse
import logging
import sys
import numpy as np
from autopsy.scr.autopsy import PropertyCalculator
from autopsy.util.read_trajectory import read_trajectory
from autopsy.util.unwrap import unwrap

def print_exit():
    logging.error("*****************************")
    logging.error("Use autopsy as follows:")
    logging.error("autopsy dump.lmp --zmin 1,2,3 --zmax 2,3,4")
    logging.error("Here 1,2,3 is a list of zmin values")
    logging.error("*****************************")
    print("*****************************")
    print("Use autopsy as follows:")
    print("autopsy dump.lmp --zmin 1,2,3 --zmax 2,3,4")
    print("Here 1,2,3 is a list of zmin values")
    print("*****************************")    
    sys.exit()

def is_int_or_float_list(lst):
    for item in lst:
        try:
            item = float(item)
            if not isinstance(item, (int, float)):
                print_exit()
        except:
            print_exit()

def main(args=None):
    parser = argparse.ArgumentParser(description="Run autopsy for postprocessing of LAMMPS trajectory")
    parser.add_argument('dump_file', help="The LAMMPS trajectory file to process")
    parser.add_argument('--zmin', type=str, default=None, help="Comma-separated (without space) list of zmin values: 1,2,3")
    parser.add_argument('--zmax', type=str, default=None, help="Comma-separated (without space) list of zmax values: 2,3,4")
    parser.add_argument('--atoms', type=str, help="Comma-separated (without space) list of atomic symbols: H,He,Li")

    args = parser.parse_args(args)

    atoms_list = args.atoms.split(',') if args.atoms else None
    zmin_list = args.zmin.split(',') if args.zmin else []
    zmax_list = args.zmax.split(',') if args.zmax else []

    if zmin_list:
        is_int_or_float_list(zmin_list)
    if zmax_list:
        is_int_or_float_list(zmax_list)

    if len(zmin_list) != len(zmax_list):
        print_exit()

    data, s_time = read_trajectory(args.dump_file)
    cell = np.linalg.norm(data[-1].cell, axis=1)

    if not zmin_list:
        zmin_list = [None]
        logging.info(f"z_min list is {zmin_list}")
    if not zmax_list:
        zmax_list = [None]
        logging.info(f"z_max list is {zmax_list}")

    for zmin, zmax in zip(zmin_list, zmax_list):
        zmin_ = str(0) if zmin is None else zmin
        zmax_ = str(int(cell[2])) if zmax is None else zmax
        out_dir = f"MSD_{zmin_}_{zmax_}"
        logging.info(out_dir)
        calculator = PropertyCalculator(data, stime=s_time, zmin=zmin, zmax=zmax, atoms=atoms_list, out_dir=out_dir)
        calculator.run()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)

