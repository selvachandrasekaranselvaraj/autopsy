#from autopsy import PropertyCalculator
from autopsy.util.read_trajectory import read_trajectory
from autopsy.scr.autopsy import PropertyCalculator
import sys, os
import numpy as np

def print_exit():
    print("*****************************")
    print("Use autopsy as follow:")
    print("autopsy dump.lmp 1,2,3  2,3,4")
    print("Here 1,2,3 is a list of zmin")
    print("*****************************")
    return sys.exit()

def is_int_or_float_list(lst):
    for item in lst:
        try:
            item = int(item)
            if not isinstance(item, (int, float)):
                print_exit()
        except:
            print_exit()

def main():
    #if isinstance(sys.argv[2], list):
    try:
        if sys.argv[2]:
           zmin_list = sys.argv[2].split(',')
           is_int_or_float_list(zmin_list)
    except:
        #print_exit()
        zmin_list = None
        print("There is no z_min input")
        pass
    try:
        if sys.argv[3]:
            zmax_list = sys.argv[3].split(',')
            is_int_or_float_list(zmax_list)
    except:
        #print_exit()
        zmax_list = None
        print("There is no z_max input")
        pass

    try:
        if sys.argv[2]:
            if len(zmin_list) != len(zmax_list):
                print_exit()
    except:
        #print_exit()
        pass

    data, s_time = read_trajectory()
    cell = np.linalg.norm(data[-1].cell, axis=1)

    if not zmin_list:
        zmin_list = [None] #[str(0)]
        print(f"z_min list is {zmin_list}")
    if not zmax_list:
        zmax_list = [None] #[str(int(cell[2]))]
        print(f"z_max list is {zmax_list}")



    for zmin, zmax in zip(zmin_list, zmax_list):
        if zmin == None:
            zmin_ = str(0)
        else:
            zmin_ = zmin
        if zmax == None:
            zmax_ = str(int(cell[2]))
        else:
            zmax_ = zmax
        out_dir = f"MSD_{zmin_}_{zmax_}"
        print(out_dir)
        calculator = PropertyCalculator(data, stime=s_time, zmin=zmin, zmax=zmax, out_dir=out_dir)

        # Run the calculations
        calculator.run()

# Other CLI-related functions or commands can be defined here
#if __name__ == "__main__":
#    main()

