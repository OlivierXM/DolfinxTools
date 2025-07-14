"""
    Custom Plotting Functionality from Matplotlib

    Provides a Matlab-esque interface
    Tasks:
    - Add support for arbitrary **kwargs for most functions
"""
import matplotlib.pyplot as plt
from matplotlib import font_manager as fontP
import math

class Font(fontP.FontProperties):
    """
    Inherited matplotlib.font_manager
    Create a custom font set
    """
    def __init__(self, **kwargs):
        """
        Initialize
        Args:
            **kwargs: (size = 12, weight = "bold", ....) 
        """
        super().__init__(**kwargs)

class PrebuiltPlot():
    """
        A Matlab inspired plot using matplotlib
    """
    def __init__(self, a:int, b:int, fontProp, shareX = True, **kwargs):
        """
            Instantiate the figure with subplots
            Args:
                a : Number of rows
                b : Number of columns
                fontProp : Default Font
                shareX : Do all subplots share the x axis
        """
        self._font = fontProp
        self._fig, (self._axes) = plt.subplots(a, b, sharex = shareX, **kwargs)
        if (a*b == 1):
            self._axes = [self._axes, self._axes]

        self._row = a
        self._col = b
        self._size = a*b    

    def plot(self, axisN:int, datax, datay, styleIn = '-', labelIn = None, lwIn = 2):
        """
            Plot data on the specified axis
            Args:
                axisN : Zero based axis index
                datax : X data to plot
                datay : Y data to plot
                styleIn : Line style to use
                labelIn : Legend label to assign
                lwIn : Line width to use
        """
        self.ConvertN(axisN).plot(datax, datay, styleIn, label = labelIn, lw = lwIn)

    def xlabel(self, axisN, strIn, fontIn = None):
        if (fontIn == None): fontIn = self._font

        if (axisN < 0):
            for i in range(self._size):
                self.ConvertN(i).set_xlabel(strIn, fontproperties = fontIn)
        else:
            self.ConvertN(axisN).set_xlabel(strIn, fontproperties = fontIn)
   
    def ylabel(self, axisN, strIn, fontIn = None):
        if (fontIn == None): fontIn = self._font
            
        if (axisN < 0):
            for i in range(self._size):
                self.ConvertN(i).set_ylabel(strIn, fontproperties = fontIn)
        else:
            self.ConvertN(axisN).set_ylabel(strIn, fontproperties = fontIn)

    def legend(self, axisN, locIn = 'lower center', fontIn = None, **kwargs):
        if not(isinstance(axisN, list)): axisN = [axisN]
        if (fontIn == None): fontIn = self._font
        for axisI in axisN:
            self.ConvertN(axisI).legend(loc = locIn, prop = fontIn, **kwargs)

    def axis(self, axisN, xl=None, xu=None, yl=None, yu=None):
        if (axisN < 0):
            for i in range(self._size):
                self.ConvertN(i).set(xlim=(xl, xu), ylim=(yl, yu))
        else:
            self.ConvertN(axisN).set(xlim=(xl, xu), ylim=(yl, yu))

    def grid(self, axisN, whichIn = 'major'):
        if not(isinstance(axisN, list)):
            axisN = [axisN]

        for axisI in axisN:
            self.ConvertN(axisI).grid(which = whichIn)
            
    def save(self, savePath:str, bbox_inches = "tight", **kwargs):
        self._fig.savefig(savePath, bbox_inches = bbox_inches, **kwargs)

    def title(self, strIn, fontIn = None):
        if (fontIn == None): fontIn = self._font
        self._fig.suptitle(strIn, fontproperties = fontIn)

    def show(self, blockIn = False):
        plt.show(block = blockIn)
        waitIt = input("Press enter to continue:.....")
        

    def ConvertN(self, N) -> plt.subplot:
        """
            Get the zero based nxn or nx1 or 1xn axis.
            2x2 => [0 2][1 3]
                
            Args:
                N: The zero-based index of the axis
            Returns:
                out: The matplotlib.pyplot.subplot axis handle
        """
        floorC = math.floor(N / self._row)
        floorR = (N - floorC * self._row)
        if (self._row == 1 or self._col== 1):
            return self._axes[N]
        else:
            return self._axes[floorR, floorC]