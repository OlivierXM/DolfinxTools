"""
    Run Tests on Tools
    OlivierXM

"""
import sys
import os
from mpi4py import MPI

sys.path.append("../src")
from TestModules import *
from dolfinxtools import mpitools

commWorld = MPI.COMM_WORLD
numCore = commWorld.Get_size() 
thisCore = commWorld.Get_rank()

if (numCore == 1):
    print("Testing run in serial.")

else:
    mpitools.MPIprint(thisCore, 0, f"Testing run with {numCore} cores")

testRunner = TestRunner(commWorld)

## Tests ##
testRunner.append(FacetIntegrals_2D())
testRunner.append(TestBox())
testRunner.append(Material_3D_Simple())

testRunner.RunTests()
## END SCRIPT ##