from mpi4py import MPI
import numpy as np
import math

## Partition a numpy array based on optimal divisions
def PartitionArray(arrIn:np.ndarray, comm:MPI.Intracomm, minDiv:int, index:int=0):
    # Get Comm Parameters #
    commSize = comm.Get_size()
    thisCore = comm.Get_rank()

    # Optimize Partition #
    numReq = min(math.ceil(arrIn.size/minDiv), commSize) # Optimal Processes
    partSize = math.ceil(arrIn.size/numReq) # Redefine partition size

    if (commSize == 1):
        return arrIn, 1, np.arange(arrIn.size)
    else:
        # Partition Array Elements #
        if (thisCore == partSize - 1):
            subArray = arrIn[(thisCore*partSize):,:]
            indices = np.arange(thisCore*partSize, arrIn.shape[index], dtype=np.int32)
        if (thisCore < partSize - 1):
            subArray = arrIn[thisCore*partSize:(thisCore+1)*partSize,:]
            indices = np.arange(thisCore*partSize, (thisCore+1)*partSize, dtype=np.int32)

        if (thisCore > partSize - 1):
            subArray = []
            indices = []

    return subArray, numReq, indices

## Concatenate arrays of varying size, with one equal dimension
def CompileReducedArray(arrIn:np.array, comm:MPI.Intracomm, numReq:int, index:int=0):
    if (comm.Get_size() == 1):
        return arrIn

    thisCore = comm.Get_rank()
    totalSize = comm.allreduce(arrIn.shape[0], op=MPI.SUM)
    colSize = 0
    if (arrIn.size > 0 and len(arrIn.shape) > 1):
        colSize = arrIn.shape[1]

    colSize = comm.bcast(colSize, root=0)
    if (len(arrIn.shape) > 1):
        shape = (totalSize, colSize)
    else:
        shape = (totalSize,)

    outArray = np.zeros(shape, dtype=arrIn.dtype)
    startIndex = 0
    for i in range(numReq):
        localArrayS = arrIn.shape[0]
        localArrayS = comm.bcast(localArrayS, root=i)
        if (len(arrIn.shape) > 1):
            localShape = (localArrayS, arrIn.shape[1])
        else:
            localShape = (localArrayS,)

        localArray = np.ndarray(localShape, dtype=arrIn.dtype)
        if (thisCore == i):
            localArray[:] = arrIn

        comm.Bcast(localArray, root=i)
        outArray[startIndex:startIndex+localArrayS] = localArray
        startIndex = startIndex + localArrayS

    return outArray

class MPIFile():
    """
    Text or csv style file generation in parallel
    """
    def __init__(self, comm:MPI.Intracomm, fileName:str, writeMode:str="w", rank:int=0):
        """
        Initialize
        Args:
            comm: MPI Communicator for World
            fileName: Filepath to write
            writeMode: Write/Read access permission ("w")
            rank: Process allowed to write to file (0)
        """
        self._comm = comm
        self._fileName = fileName
        self._writeMode = writeMode
        self._rank = rank
        if (self.isroot):
            self._file = open(fileName, writeMode)

    @property
    def isroot(self) -> bool:  
        """
        Get whether the MPIFile instance is root
        Returns:
            Whether file is root (bool)
        """ 
        return self._comm.Get_rank() == self._rank  

    def write(self, data:str, feed:str="\n"):
        """
        Write to file
        Args:
            data: String to write to file
            feed: Optionally specify linefeed ("\n")
        """
        if self.isroot:
            self._file.write(data + feed)

    def titleblock(self, data:str, newLine:str=""):
        """
        Write a title block to file
        Args:
            data: The string to write
            newLine: Optionally specify additional spacing prior to line ("")
        """
        if self.isroot:
            self._file.write(newLine + "------" + data + "------\n")

    def reopen(self, newMode:str="a+"):
        """
        Reopen a previously closed file
        Args:
            newMode: Optionally specify write mode ("a+")
        """
        if (self.isroot):
            self._file = open(self._fileName, newMode)

    def close(self):
        """
        Close the file
        """
        if self.isroot:
            self._file.close()

def MPIprint(a:int, b:int, c:str, end="\n"):
    """
    Ensure only info from a specific core is printed
    Args:
        a: The calling process number
        b: The target process to print
        c: Message to print to screen
        ter: The line terminator
    """
    if (a == b):
        print(c, end = end)