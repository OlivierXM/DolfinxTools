import ufl
from dolfinx import fem, mesh
import petsc4py.PETSc as pets
import typing
import numpy as np

def epsilon(u:fem.Function) -> ufl.tensoralgebra.Sym:
    """
        Return symmetric part of the gradient of u
        Args:
            u : The displacement function
    """
    return ufl.sym(ufl.grad(u))

def voigtstress(vec:fem.Function, tdim:int = None) -> ufl.tensors.ListTensor:
    """
        Ambiguous call for returning a vector as a symmetric tensor using Voigt notation
        Args:
            vec : The ufl object to rearrange
            tdim : The topological dimension
        Returns:
            out : The ufl tensor (2x2) or (3x3)
    """
    if (tdim == None):
        tdim = vec.function_space.mesh.topology.dim

    if (tdim == 2):
        return voigt2stress(vec)
    else:
        return voigt3stress(vec)

def voigt2stress(vec:fem.Function) -> ufl.tensors.ListTensor:
    """
        Rewrite a vector into a symmetric tensor using Voigt notation
        Args:
            vec : The ufl object to rearrange (xx, yy, xy)
        Returns:
            out : The ufl tensor 2x2
    """
    return ufl.as_tensor([[vec[0], vec[2]],
                      [vec[2], vec[1]]])

def voigt3stress(vec:fem.Function) -> ufl.tensors.ListTensor:
    """
        Rewrite a 6x1 vector into a symmetric tensor using Voigt notation
        Args:
            vec : The ufl object to rearrange (xx, yy, zz, yz, xz, xy)
        Returns:
            out : The ufl tensor 3x3
    """
    return ufl.as_tensor([[vec[0], vec[5], vec[4]],
                      [vec[5], vec[1], vec[3]],
                      [vec[4], vec[3], vec[2]]])

def SigmaVec(sigmaIn, tdim:int = None):
    """
        Ambiguous call to return a symmetric stress tensor as vector
        Args:
            sigmaIn : The (nxn) stress matrix
        Returns:
            out : The (3*n-3)x1 stress vector
    """
    if (tdim == None):
        tdim = sigmaIn.function_space.mesh.topology.dim

    if (tdim == 2):
        return SigmaVec2(sigmaIn)
    else:
        return SigmaVec3(sigmaIn)

def SigmaVec2(sigmaIn):
    """
        Return the symmetric 2x2 sigma vector as a ufl.as_vector (xx, yy, xy)
        Args:
            sigmaIn : The (nxn) stress matrix
        Returns:
            out : The (3*n-3)x1 stress vector
    """
    return ufl.as_vector([sigmaIn[0,0],sigmaIn[1,1],sigmaIn[0,1]])

def SigmaVec3(sigmaIn):
    """
        Return the symmetric 3x3 sigma vector as a ufl.as_vector (xx, yy, zz, yz, xz, xy)
        Args:
            sigmaIn : The (nxn) stress matrix
        Returns:
            out : The (3*n-3)x1 stress vector
    """
    return ufl.as_vector([sigmaIn[0,0],sigmaIn[1,1],sigmaIn[2,2], sigmaIn[1,2], sigmaIn[0,2], sigmaIn[0,1]])

def Macaulay(arg1, arg2):
    """
        Return arg1 if sign matches arg2
        Args:
            arg1 : The value
            arg2 : The sign (1/-1), returns 0 if opposite
        Return:
            out : The summed result
    """
    return 0.5*(arg1 + arg2*abs(arg1))

def DevStress(stressIn:fem.Function, tdim:int = None):
    """
        Return the deviatoric part of the Cauchy stress
        Args:
            stressIn : The (3*n-3)x1 stress vector
            tdim : The topological dimension (2D/3D)
        Returns:
            out : The nxn symmetric stress matrix
    """
    if (tdim == None):
        tdim = stressIn.function_space.mesh.topology.dim
    
    if (tdim == 2):
        vS = voigt2stress(stressIn)
    else:
        vS = voigt3stress(stressIn)

    return vS - (1./3) * ufl.tr(vS) * ufl.Identity(tdim)

def DevStrain(u:fem.Function, tdim:int = None):
    """
        Return the deviatoric part of the symmetric gradient of u
        Args:
            u : The displacement function nx1
            tdim : The 
        Returns:
            out : The deviatoric strain matrix
    """
    if (tdim == None):
        tdim = u.function_space.mesh.topology.dim
    localEps = epsilon(u)
    return localEps - (1./3) * ufl.tr(localEps)*ufl.Identity(tdim)

def ZeroTens(domain:mesh.Mesh, isTens:bool) -> fem.Constant:
    """
        Return the zero tensor for domain's topological space
        Arg:
            domain : The topological domain of dim = n
            isTens : Return the (3*n-3) or n matrix (false)
        Returns:
            out : The fem Constant tensor
    """
    tDim = domain.topology.dim
    if (isTens):
        k = 3 * (tDim - 1)
        return fem.Constant(domain, pets.ScalarType(np.zeros((k, k), dtype = np.float64)))
    else:
        return fem.Constant(domain, pets.ScalarType(np.zeros((tDim, tDim), dtype = np.float64)))

def StrainVector(tens:ufl.tensors.ListTensor, tdim:int):
    """
        Return an nx1 strain vector with 2x correction for shear strains
        Args:
            tens : The ufl strain tensor e_ij
            tdim : The topological dimension
        Returns:
            out : The nx1 strain vector (exx, eyy, 2 * exy)
    """
    if (tdim == 2):
        return ufl.as_vector([tens[0,0], tens[1,1], 2*tens[0,1]])
    else:
        return ufl.as_vector([tens[0,0], tens[1,1], tens[2, 2], 2*tens[1, 2], 2*tens[0,2], 2*tens[0,1]])