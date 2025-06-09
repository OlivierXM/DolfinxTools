"""
    Test Modules for DolfinxTools

    OlivierXM
"""

# Standard Imports #
from mpi4py import MPI
from dolfinx import mesh, fem, nls, io, cpp, log
import numpy as np
import ufl
import petsc4py.PETSc as pets
import os
import sys
import datetime
import time
import math

sys.path.append("Tools")
# Custom Imports #
import Notes
import cdfx_09 as cdfx
import mpitools

class TestRunner():
    tests = []
    totalCount = 0
    success = 0
    def __init__(self, comm:MPI.Intracomm):
        self.tests = []
        self.success = 0
        self.totalCount = 0
        self._comm = comm
        self._rank = comm.Get_rank()
    
    def append(self, newTest):
        self.tests.append(newTest)
    
    def RunTests(self):
        self.totalCount = len(self.tests)
        self.success = 0
        mpitools.MPIprint(self._rank, 0, f"Total tests : {self.totalCount}")
        mpitools.MPIprint(self._rank, 0, "--------------")
        for test in self.tests:
            mpitools.MPIprint(self._rank, 0, test.Name(), end =": ")
            result = test.Run()
            mpitools.MPIprint(self._rank, 0, result, end ="\n")
            if (result):
                self.success = self.success + 1
        
        mpitools.MPIprint(self._rank, 0, "--------------")
        mpitools.MPIprint(self._rank, 0, f"Total of {self.success} of {self.totalCount} tests succeeded")

class FacetIntegrals_2D():
    def Name(self):
        return "FacetIntegrals_2D"

    def Purpose(self):
        return "Test the usage of cdfx.SubDomain and cdfx.FacetFunction on simple geometry."

    def Run(self):
        commWorld = MPI.COMM_WORLD
        nx = 20
        ny = 20
        width = 1
        height = 2
        domain = mesh.create_rectangle(commWorld, [[0, 0],[width, height]], [nx, ny], ghost_mode=mesh.GhostMode.shared_facet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        num_facets = domain.topology.index_map(fdim).size_local + domain.topology.index_map(fdim).num_ghosts
        boxTree = cdfx.BoxTree(domain) 
        tol = 1e-12 # Tolerance for mesh tagging (not for computations) [m]
        scalar1 = fem.Constant(domain, pets.ScalarType(1.0))
        class Left(cdfx.SubDomain):
            def inside(self, x):
                return self.near(x[0], boxTree._box[0, 0], tol)

        class Right(cdfx.SubDomain):
            def inside(self, x):
                return self.near(x[0], boxTree._box[0, 1], tol)

        class Bottom(cdfx.SubDomain):
            def inside(self, x):
                return self.near(x[1], boxTree._box[1, 0], tol)

        class Top(cdfx.SubDomain):
            def inside(self, x):
                return self.near(x[1], boxTree._box[1, 1], tol)   

        facetFunction = cdfx.FacetFunction(domain, fdim, num_facets)
        Left().mark(facetFunction, 1)
        Right().mark(facetFunction, 2)
        Bottom().mark(facetFunction, 3)
        Top().mark(facetFunction, 4)
        facetTags = facetFunction.CreateMeshTag()

        ds = ufl.Measure("ds", domain = domain, subdomain_data = facetTags)

        sum1 = commWorld.allreduce(fem.assemble_scalar(fem.form(scalar1*ds(1)+scalar1*ds(2))) , op = MPI.SUM)
        sum2 = commWorld.allreduce(fem.assemble_scalar(fem.form(scalar1*ds(3)+scalar1*ds(4))) , op = MPI.SUM)

        return (abs(sum1-2*height) < 1e-2) and (abs(sum2-2*width) < 1e-2)

class TestBox():
    def Name(self):
        return "Test Box"

    def Purpose(self):
        return "Test the functionality of cdfx.BoxTree() on a simple 3D domain"

    def Run(self):
         # MPI #
        commWorld = MPI.COMM_WORLD
        nx = 20
        ny = 20
        nz = 20
        width = 1
        height = 2
        depth = 3
        domain = mesh.create_box(commWorld, [[-0.5 * width, -0.5 * height, -0.5 * depth],[0.5 * width, 0.5 * height, 0.5 * depth]], [nx, ny, nz], ghost_mode=mesh.GhostMode.shared_facet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)
        boxTree = cdfx.BoxTree(domain)
        testX =  (abs(boxTree._box[0,0]+0.5*width)<1e-2) and (abs(boxTree._box[0,1]-0.5*width)<1e-2)
        testY = (abs(boxTree._box[1,0]+0.5*height)<1e-2) and (abs(boxTree._box[1,1]-0.5*height)<1e-2)
        testZ = (abs(boxTree._box[2,0]+0.5*depth)<1e-2) and (abs(boxTree._box[2,1]-0.5*depth)<1e-2)
        return testX and testY and testZ

class Material_3D_Simple():
    def Name(self):
        return "Simple 3D Material"

    def Purpose(self):
        return "Use Material Property on a 3D geometry to verify application (DG) with SubDomain()"

    def Run(self):
        commWorld = MPI.COMM_WORLD
        nx = 40
        ny = 40
        nz = 40
        width = 1
        height = 1
        depth = 1
        domain = mesh.create_box(commWorld, [[-0.5 * width, -0.5 * height, -0.5 * depth],[0.5 * width, 0.5 * height, 0.5 * depth]], [nx, ny, nz], ghost_mode=mesh.GhostMode.shared_facet)
        tdim = domain.topology.dim
        fdim = tdim - 1
        domain.topology.create_connectivity(fdim, tdim)

        class LowerLevel(cdfx.SubDomain): # Lower Level
            def inside(self, x):
                x2 = x[2] - 0 # Loading is centered along length
                return x2 < (0.5-0.67) * depth

        Fs_DG = fem.functionspace(domain, ("DG", 0))
        spatialValue = cdfx.MaterialProperty(Fs_DG, default=2)
        LowerLevel().assign(spatialValue, 1) # override values at the supports
        dz = ufl.Measure('dx', domain = domain)
        averageValue = commWorld.allreduce(fem.assemble_scalar(fem.form(spatialValue * dz)), op = MPI.SUM)
        if (abs(averageValue-1.67) < 1e-2):
            return True    
        else:
            print(f"Value returned is {averageValue}, expected 1.67.\n")
            return False