#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib.ticker as ticker

def read_file(file):
    """Return a numpy array.

    Reads file, seeks for 'fvalue' string in each line 
    and stores the cost function value computed at each 
    iteration of the factorization function which was
    used. These values are stored in a numpy array.

    :param file The path where the log file is located.
    """
    with open (file, 'rt') as logfile:
        fvalue = []
        for line in logfile:
            if(line.find("fvalue: ") != -1):
                splitted = line.split()
                fvalue_idx = splitted.index("fvalue:")
                fvalue.append(splitted[fvalue_idx+1:fvalue_idx+2])
        data = []
        for l in fvalue:
            for item in l:
                data.append(float(item))
    return np.array(data)

def matplot(fileName, title, y):
    """
    Prepares the plot where the data will be shown.
    The y-label shows the cost function value of the 
    fuction for each iteration specified in x-label. 
    Saves the plot in the current directory with name
    fileName.

    :param fileName The name of the file containing the plot.
    :param title    The title of the plot.
    :param y        Array with the values(cost function) for y-label. 
    """

    x  = [i for i in range(1,len(y)+1,1)]
    plt.title(title, fontweight='bold')
    plt.plot(x,y,'-o')
    plt.xlabel('iterations', fontweight='bold')
    plt.ylabel('cost function value', fontweight='bold')
    
    # Format xlabel to print integers instead of floats.
    locator = matplotlib.ticker.MultipleLocator(2)
    plt.gca().xaxis.set_major_locator(locator)
    formatter = matplotlib.ticker.StrMethodFormatter("{x:.0f}")
    plt.gca().xaxis.set_major_formatter(formatter)
    plt.savefig(fileName,bbox_inches='tight')

def main():
    """
    Reads file specified in variable 'fileName' and plots the 
    cost function value over iterations. 

    In case the path of the log file has changed, modify the 
    variable 'fileName'. 
    The final product of this function is a '.png' file in the
    current directory.

    NOTE: In order to produce a valid image the log file
          must be deleted or renamed before every run!
    """
    fileName = '../log/partensor.txt'    # Full Path to results file
    results  = read_file(fileName)
    sz       = results.size
    if results.size==0:
        print(f'There are no data in the file {fileName}!')
        sys.exit()

    fig        = plt.figure(1,constrained_layout=True)
    exportName = 'cost_function.png'
    title      = 'Results'
    matplot(exportName, title, results)

if __name__ == '__main__':
    main()
    help(main)
