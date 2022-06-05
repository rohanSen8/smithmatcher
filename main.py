# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd

from Tabulate import Tabulate
from smith import SmithChart

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    p1 = SmithChart("John", 36)
    p1.intializeZin(complex(0.01,0.5))
    #p1.showMoves()
    p1.ZA2AZ()
    #p1.moveTowardGenerator(0.125)
    #print(p1.gammatoZ(p1.gamma))
    #p1.addxl(-1)
    #p1.showMoves()
    #p1.addrin(1)
    y=p1.hitzyOne()
    p1.addxl(y.imag)
    p1.ZA2AZ()
    z=p1.hitzyMatch()
    p1.addxl(z.imag)
    #p1.addZ(1,1)
    p1.showMoves()
    #print(p1.gammatoZ(p1.gamma))

def mat(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    Tabulate('ro')
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    #Tabulate('ro')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
