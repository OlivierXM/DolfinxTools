# Dolfinx Tools
A repository of Dolfinx case studies

# Scripts
DolfinxTools contains the following utilities

## cdfx.py
- Provides a legacy dolfin-esque interface to dolfinx
  - i.e. a SubDomain class that can be used to mark facets

## CreateMesh.py
- Generate a dolfinx compatible xdmf/h5 mesh file from GMSH .geo extension
  - Supports mesh tags, facet marking, cell tags


# Test Cases
A growing repository of test cases is also provided using the provided tools