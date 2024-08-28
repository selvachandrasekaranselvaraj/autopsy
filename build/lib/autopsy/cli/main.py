#from autopsy import PropertyCalculator
from autopsy.util.read_trajectory import read_trajectory
from autopsy.scr.autopsy import PropertyCalculator
def main():
    data = read_trajectory()
    calculator = PropertyCalculator(data)

    # Run the calculations
    calculator.run()

# Other CLI-related functions or commands can be defined here
if __name__ == "__main__":
    main()

