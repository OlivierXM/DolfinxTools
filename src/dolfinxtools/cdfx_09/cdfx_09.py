"""
Classical DolfinX (Legacy FEniCS)

Used to adjust user from verbose used in dolfin to DolfinX

For example the class SubDomain is recreated here to provide ease of use
similary to that of legacy FEniCS.
"""
from dolfinx import fem
from dolfinx import mesh
from dolfinxtools import cdfx_09
from mpi4py import MPI
import petsc4py.PETSc as pets
import numpy as np
import ufl
import math
import typing

class SubDomain:
    """
        A dolfinx equivalent of dolfin.SubDomain
    """
    def __init__(self) -> None:
        """
            Default class constructor
        """
        return

    def on_boundary(self) -> bool:
        """
            Is object on the boundary? (Not Implemented)
        """
        return False

    def lor(self, *args) -> np.ndarray:
        """
            Shorthand for np.logical_or()
            Args:
                *args : Arbitrary series of boolean results
            Returns:
                res : np.ndarray of bool
        """
        res = np.logical_or(args[0], args[1])
        for i in args[2:]:
            res = np.logical_or(i, res)
        return res

    def land(self, *args) -> np.ndarray:
        """
            Shorthand for np.logical_and()
            Args:
                *args : Arbitrary series of boolean results
            Returns:
                res : np.array of bool
        """
        res = np.logical_and(args[0], args[1])
        for i in args[2:]:
            res = np.logical_and(i, res)
        return res

    def lnot(self, a) -> np.ndarray:
        """
            Shorthand for np.logical_not()
            Args:
                a : Boolean array
            Returns:
                res : np.ndarray of bool
        """
        return np.logical_not(a)

    def inside(self, x) -> np.ndarray:
        """
            A placeholder to be overriden. Called to determine entities satisfying conditions
            Args:
                x: The point coordinates [np.ndarray]
            Returns:
                Boolean array of logicals
        """
        return True

    def near(self, xIn, val, atol:typing.Union[int, float]) -> np.ndarray:
        """
            Functionality akin to dolfin.near()
            Return whether a given value is absolutely close to another point
            Args:
                xIn : The value to check
                val : The value to compare to
                atol : The absolute tolerance
            Returns:
                Array of logical [np.ndarray]
        """
        return np.isclose(xIn, val, rtol=0, atol=atol)

    def assign(self, matProp, value) -> None:
        """
            Provide a marker array to assign specified value
        """
        markedArray = mesh.locate_entities(matProp._domain, matProp._dim, self.inside)
        matProp.x.array[markedArray] = np.full(len(markedArray), value) 

    def mark(self, facetFunction, value:typing.Union[int, float]) -> None:
        """
            Given a facetFunction, mark appropriate facets with value as determined by SubDomain.inside()
            Args:
                facetFunction : The cdfx_09.FacetFunction
                value : The value to assign to matching facets
        """
        if (not(self.on_boundary())):
            markedArray = mesh.locate_entities(facetFunction._domain, facetFunction._fdim, self.inside)
        else:
            self._boundary = np.array(mesh.compute_boundary_facets(facetFunction._domain.topology))
            markedArray = mesh.locate_entities(facetFunction._domain, facetFunction._fdim, self.insideBoundary)

        facetFunction._indices.append(markedArray)
        facetFunction._markers[markedArray] = value
        return

def DirichletBCs(argList:dict, fs:fem.FunctionSpace, facetTags:mesh.MeshTags, type:bool=True) -> list[fem.bcs.DirichletBC]:
    """
        Generate dolfinx compatible dirichlet boundary conditions
        Args:
            argList : Dictionary of [BC Type, SurfaceTag, (optional)Direction]
            fs: fem.FunctionSpace associated with bcs
            facetTags: mesh.MeshTags for the corresponding bcs
            type: (optional) Search for tags topologically (True) or geometrically [bool]
        Returns:
            bcs: A list of fem.bcs.DirichletBC containing all relevant Dirichlet bcs
    """
    fdim = fs.mesh.topology.dim - 1
    bcs = []
    for i in argList:
        if 'Dirichlet' in argList[i]:
            if isinstance(argList[i]['Dirichlet'], fem.Constant):
                value = argList[i]['Dirichlet']
            else:
                value = fem.Constant(fs.mesh, pets.ScalarType(argList[i]['Dirichlet']))

            if 'Direction' in argList[i]:
                facets = np.array(facetTags.indices[facetTags.values==argList[i]['Surface']])
                if (type):
                    left_dofs = fem.locate_dofs_topological(fs.sub(argList[i]['Direction']), fdim, facets)
                    bcs.append(fem.dirichletbc(value, left_dofs, fs.sub(argList[i]['Direction'])))
                else:
                    left_dofs = fem.locate_dofs_geometrical(fs.sub(argList[i]['Direction']).collapse()[0], argList[i]['Marker'])
                    bcs.append(fem.dirichletbc(value, left_dofs, fs.sub(argList[i]['Direction'])))
                
            else:
                facets = np.array(facetTags.indices[facetTags.values==argList[i]['Surface']])
                if (type):
                    left_dofs = fem.locate_dofs_topological(fs, fdim, facets)
                else:
                    left_dofs = fem.locate_dofs_geometrical(fs, argList[i]['Marker'])

                bcs.append(fem.dirichletbc(value, left_dofs, fs))
    return bcs

def RobinBCs(argList:dict, ds:ufl.Measure, u:fem.Function, v:ufl.TestFunction) -> list:
    """
        Generate a compiled form of Robin boundary conditions
        Args:
            argList: Dictionary of [BC Type, SurfaceTag, tuple(Tinf, Convection Coefficient)]
            ds: ufl.Measure object to use for compiling form
            u: fem.Function to use with Tinf (or relevant external setpoint)
            v: ufl.TestFunction used in the form
    """ 
    bcs = []
    for i in argList:
        if 'Robin' in argList[i]:
            valT = argList[i]['Robin'][0]
            valH = argList[i]['Robin'][1]
            bcs.append(valH * ufl.inner(valT-u, v) * ds(argList[i]['Surface']))
    return sum(bcs)   

class MaterialProperty(fem.Function):
    ## MaterialProperty()
    # Construct a fem.Function with args
    # - fs: Provide the fem.FunctionSpace for this property
    # - default: Does the Material Property have an initial global value?
    def __init__(self, fs, default = None):
        super().__init__(fs)
        self._domain = fs.mesh
        self._dim = self._domain.topology.dim
        if not(default == None):
            self.assignAll(default)

    def assign(self):
        print("assign() is not directly supported by MaterialProperty, use assign from cdfx.SubDomain")        

    def CellTagMark(self, region, marker:int, value):
        """
            Given a MeshTags object defined over the domain, use tags to assign values
            Args:
                region : MeshTags object
                marker : Marker to find
                value : Value to assign to function
        """
        markedArray = region.find(marker)
        self.x.array[markedArray] = np.full(len(markedArray), value)      
        
    def assignAll(self, value):
        """
            Assign every cell in this collection the same value
            Args:
                value : The default value to assign
        """
        self.x.array[:] = np.full(len(self.x.array), value)


class FacetFunction:
    """
        A utility function for marking mesh facets
    """
    def __init__(self, domain:mesh.Mesh, fdim:int, numEntities, initVal:int=0):
        """
            Default Constructor
            Args:
                domain : The dolfinx.mesh.Mesh
                fdim : The facet dimension
                numEntities : 
                initVal : Initial facet value
        """
        self._domain = domain
        self._fdim = fdim
        self._indices = []
        self._numEntities = numEntities
        # self._markers = []
        self._markers = np.zeros(self._numEntities, dtype=np.int32)
        self._markers[:] = initVal


    def MarkAll(self, value:int):
        """
            Assign all values the same integer
            Args:
                value : The new integer to assign
        """
        self._markers[:] = value

    def CellTagMark(self, region, marker, value):
        markedArray = region.find(marker)
        self._indices.append(markedArray)
        self._markers[markedArray] = value   

    def CreateMeshTag(self, name:str="meshtags") -> mesh.MeshTags:
        """
            Create a mesh tag from the current function
            Args:
                name : Optionally specify the mesh tag name (meshtags)
            Returns:
                out : The mesh.MeshTags object
        """
        out = mesh.meshtags(self._domain, self._fdim, np.arange(self._numEntities, dtype=np.int32), self._markers)
        out.name = name
        return out

class BoxTree(object):
    """
        A general tool used for getting mesh boundary dimensions
    """
    def __init__(self, domain:mesh.Mesh):
        """
            The default constructor with args
            Args:
                domain : The dolfinx.mesh.Mesh object
        """
        self._domain = domain
        self._dim = domain.topology.dim
        self._box = np.ndarray(shape = (self._dim,2), dtype = float)
        for i in range(self._dim):
            self._box[i,0] = domain.comm.allreduce(np.min(domain.geometry.x[:, i]), op=MPI.MIN)
            self._box[i,1] = domain.comm.allreduce(np.max(domain.geometry.x[:, i]), op=MPI.MAX)
        
    def __getitem__(self, item):
        return self._box[item[0], item[1]]
    @property 
    def center(self) -> np.ndarray:
        """
            Get the center of the domain
        """
        x0 = self._box[0, 0] + 0.5 * self.length
        y0 = self._box[1, 0] + 0.5 * self.width
        z0 = self._box[2, 0] + 0.5 * self.height
        return np.ndarray([x0, y0, z0], dtype=np.float64)

    @property
    def length(self):
        """
            Get the x extents
        """
        return self._box[0, 1] - self._box[0, 0]
    
    @property
    def width(self):
        """
            Get the y extents
        """
        return self._box[1, 1] - self._box[1, 0]
    
    @property
    def height(self):
        """
            Get the z extents
        """
        return self._box[2, 1] - self._box[2, 0]

def errornorm(comm:MPI.Intracomm, formArg):
    """
        Calculate the L2 error
        Args:
            formArg : The ufl form to compute
        Returns:
            The norm
    """
    error_local = fem.assemble_scalar(fem.form(formArg))
    return np.sqrt(comm.allreduce(error_local, op=MPI.SUM))

class FunctionX(fem.Function):
    """
        Shorthand for dolfinx.fem.Function
    """
    def __init__(self, fs:fem.FunctionSpace):
        """
            Default constructor
            Args:
                fs : The fem.Functionspace to create the function
        """
        super().__init__(fs)
        self._fs = fs

    @property
    def data(self):
        """
            Shorthand for fem.Function._cpp_object
        """
        return self._cpp_object

    def interp(self, expressIn, fs:fem.FunctionSpace = None) -> None:
        """
            Interpolate a ufl.form onto the function
            Args:
                expressIn : The ufl.form to map onto the function
                fs : Optionally specify the functionspace to use
        """
        if (fs == None):
            fs = self._fs
        self.interpolate(fem.Expression(expressIn, fs.element.interpolation_points()))
        self.x.scatter_forward()

def Rotate(angX:float, angY:float, angZ:float) -> np.ndarray:
    """
        Return a rotate matrix that rotates in x, then y, then z
        Args:
            angX : Rotation angle about x [deg]
            angY : Rotation angle about y [deg]
            angZ : Rotation angle about z [deg]
        Returns:
            out : The 3x3 rotation matrix
    """
    _rotZ = cdfx_09.RotationMatrix(angZ, np.array([0, 0, 1]))
    _rotY = cdfx_09.RotationMatrix(angY, np.array([0, 1, 0]))
    _rotX = cdfx_09.RotationMatrix(angX, np.array([1, 0, 0]))
    return np.matmul(_rotZ, np.matmul(_rotY, _rotX))


def RotationMatrix(angle:float, axis:np.ndarray) -> np.ndarray:
    """
        Return the rotation matrix for rotating an object about a specific axis
        https://en.wikipedia.org/wiki/Rotation_matrix
        Args:
            angle : The angle to rotate about axis
            axis : The axis to rotate about
        Returns:
            A 3x3 np.array used to rotate a tensor
    """
    angle = math.radians(angle)
    cosang = math.cos(angle)
    sinang = math.sin(angle)
    magN = math.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    ux = axis[0]/magN
    uy = axis[1]/magN
    uz = axis[2]/magN

    r11 = cosang + ux * ux * (1-cosang)
    r12 = ux * uy * (1-cosang) - uz * sinang
    r13 = ux * uz * (1-cosang) + uy * sinang
    
    r21 = uy * ux * (1-cosang) + uz * sinang
    r22 = cosang + uy * uy * (1-cosang)
    r23 = uy * uz * (1-cosang) - ux * sinang

    r31 = uz * ux * (1-cosang) - uy * sinang
    r32 = uz * uy * (1-cosang) + ux * sinang
    r33 = cosang + uz * uz * (1-cosang)
    return np.array([[r11, r12, r13],
                     [r21, r22, r23],
                     [r31, r32, r33]])

def VoigtRotation(angle:float, axis:np.array, rotateAxis:bool = False) -> np.ndarray:
    """
        Return the Voigt rotation matrix for transforming a compliance or stiffness matrix
        https://scicomp.stackexchange.com/questions/35600/4th-order-tensor-rotation-sources-to-refer
        Args:
            angle : The rotation matrix in degrees
            axis : The axis to rotate 
            rotateAxis : Whether to treat the axis as x,y,z angles to rotate a cartesian vector
        Returns:
            A 6x6 array for rotating the 6x6 voigt tensor
    """
    if (rotateAxis):
        rotMat = Rotate(axis[0], axis[1], axis[2])
    else:
        rotMat = RotationMatrix(angle, axis)
    
    q11 = rotMat[0, 0] ** 2
    q21 = rotMat[1, 0] ** 2
    q31 = rotMat[2, 0] ** 2
    q41 = rotMat[1, 0] * rotMat[2, 0]
    q51 = rotMat[0, 0] * rotMat[2, 0]
    q61 = rotMat[0, 0] * rotMat[1, 0]

    q12 = rotMat[0, 1] ** 2
    q22 = rotMat[1, 1] ** 2
    q32 = rotMat[2, 1] ** 2
    q42 = rotMat[1, 1] * rotMat[2, 1]
    q52 = rotMat[0, 1] * rotMat[2, 1]
    q62 = rotMat[0, 1] * rotMat[1, 1]

    q13 = rotMat[0, 2] ** 2
    q23 = rotMat[1, 2] ** 2
    q33 = rotMat[2, 2] ** 2
    q43 = rotMat[1, 2] * rotMat[2, 2]
    q53 = rotMat[0, 2] * rotMat[2, 2]
    q63 = rotMat[0, 2] * rotMat[1, 2]

    q14 = 2 * rotMat[0, 1] * rotMat[0, 2]
    q24 = 2 * rotMat[1, 1] * rotMat[1, 2]
    q34 = 2 * rotMat[2, 1] * rotMat[2, 2]
    q44 = rotMat[1, 1] * rotMat[2, 2] + rotMat[1, 2] * rotMat[2, 1]
    q54 = rotMat[0, 1] * rotMat[2, 2] + rotMat[0, 2] * rotMat[2, 1]
    q64 = rotMat[0, 1] * rotMat[1, 2] + rotMat[0, 2] * rotMat[1, 1]

    q15 = 2 * rotMat[0, 0] * rotMat[0, 2]
    q25 = 2 * rotMat[1, 0] * rotMat[1, 2]
    q35 = 2 * rotMat[2, 0] * rotMat[2, 2]
    q45 = rotMat[1, 0] * rotMat[2, 2] + rotMat[1, 2] * rotMat[2, 0]
    q55 = rotMat[1, 1] * rotMat[2, 2] + rotMat[0, 2] * rotMat[2, 0]
    q65 = rotMat[1, 1] * rotMat[1, 0] + rotMat[0, 2] * rotMat[1, 0]

    q16 = 2 * rotMat[0, 0] * rotMat[0, 1]
    q26 = 2 * rotMat[1, 0] * rotMat[1, 1]
    q36 = 2 * rotMat[2, 0] * rotMat[2, 1]
    q46 = rotMat[1, 0] * rotMat[2, 1] + rotMat[1, 1] * rotMat[2, 0]
    q56 = rotMat[1, 1] * rotMat[2, 1] + rotMat[0, 1] * rotMat[2, 0]
    q66 = rotMat[1, 1] * rotMat[2, 2] + rotMat[0, 1] * rotMat[0, 1]

    return np.array([[q11,q12,q13,q14,q15,q16],
                     [q21,q22,q23,q24,q25,q26],
                     [q31,q32,q33,q34,q35,q36],
                     [q41,q42,q43,q44,q45,q46],
                     [q51,q52,q53,q54,q55,q56],
                     [q61,q62,q63,q64,q65,q66]])


def MapSubspace(fs:fem.FunctionSpace):
    num_subs = fs.num_sub_spaces
    spaces = []
    maps = []
    for i in range(num_subs):
        space_i, map_i = fs.sub(i).collapse()
        spaces.append(space_i)
        maps.append(map_i)
    return spaces, maps


def transfer_submesh_data(u_parent: fem.Function, u_sub: fem.Function,
                          sub_to_parent_cells: np.ndarray, inverse: bool = False) -> None:
    """
    Transfer data between a function from the parent mesh and a function from the sub mesh.
    Both functions has to share the same element dof layout
    https://gist.github.com/jorgensd/9170f86a9e47d22b73f1f0598f038773
    Args:
        u_parent: Function on parent mesh
        u_sub: Function on sub mesh
        sub_to_parent_cells: Map from sub mesh (local index) to parent mesh (local index)
        inverse: If true map from u_sub->u_parent else u_parent->u_sub
    """
    V_parent = u_parent.function_space
    V_sub = u_sub.function_space
    # FIXME: In C++ check elementlayout for equality
    if inverse:
        for i, cell in enumerate(sub_to_parent_cells):
            bs = V_parent.dofmap.bs
            bs_sub = V_sub.dofmap.bs
            assert(bs == bs_sub)
            parent_dofs = V_parent.dofmap.cell_dofs(cell)
            sub_dofs = V_sub.dofmap.cell_dofs(i)
            for p_dof, s_dof in zip(parent_dofs, sub_dofs):
                for j in range(bs):
                    u_parent.x.array[p_dof * bs + j] = u_sub.x.array[s_dof * bs + j]
    else:
        for i, cell in enumerate(sub_to_parent_cells):
            bs = V_parent.dofmap.bs
            bs_sub = V_sub.dofmap.bs
            assert(bs == bs_sub)
            parent_dofs = V_parent.dofmap.cell_dofs(cell)
            sub_dofs = V_sub.dofmap.cell_dofs(i)
            for p_dof, s_dof in zip(parent_dofs, sub_dofs):
                for j in range(bs):
                    u_sub.x.array[s_dof * bs + j] = u_parent.x.array[p_dof * bs + j]

def ComputeEntities(domain:mesh.Mesh, dim:int) -> int:
    """
        Return the total number of entities including ghosts
        Args:
            domain: The Dolfinx mesh
            dim: The topology dim
        Returns:
            out: The total count
    """
    indexMap = domain.topology.index_map(dim)
    return indexMap.size_local + indexMap.num_ghosts

def Create_pcw_field(domain: mesh.Mesh, cell_markers:mesh.MeshTags, property_dict:dict, name:typing.Optional[str] = None, default_value:typing.Optional[typing.Union[int, float]] = 0) -> fem.Function:
    """
    Create a piecewise constant field with different values per subdomain.
    Adapted from https://bleyerj.github.io/comet-fenicsx/tours/interfaces/intrinsic_czm/intrinsic_czm.html
    Args:
        domain: Mesh
        cell_markers: Meshtags of topology
        property_dict: A dictionary mapping region tags to physical values
        name: Function name (optional)
        default_value: Default value for function (optional)

    Returns:
        k: A DG-0 function existing on domain
    """
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    k.x.array[:] = default_value
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k

def project(function, space:fem.FunctionSpace, dz:ufl.Measure, petscOpt = None) ->fem.Function:
    """
        Project an expression onto another space
        Args:
            function: The function to project
            space: The target function space
            dz: The ufl measure object
        Return:
            out: The projected solution
    """
    p = ufl.TrialFunction(space)
    q = ufl.TestFunction(space)
    a = ufl.inner(p, q) * dz
    L = ufl.inner(function, q) * dz

    if (petscOpt == None):
        problem = fem.petsc.LinearProblem(a, L, bcs = [])
    else:
        problem = fem.petsc.LinearProblem(a, L, bcs = [], petsc_options = petscOpt)
    return problem.solve()

def Assemble_Scalar(comm:MPI.Intracomm, form_arg, op:MPI.Op = MPI.SUM) -> float:
    """
        Return the assembled scalar for the form args
        Args:
            comm : The MPI communicator to use
            form_arg : The form args to assemble
            op : The MPI operation to execute
        Returns:
            out : The MPI assembled arg
    """
    return comm.allreduce(fem.assemble_scalar(fem.form(form_arg)), op = op)

## End SCRIPT ##
