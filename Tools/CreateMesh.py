"""
    Generate a dolfinx compatible xdmf/h5 from GMSH .geo file
    Currently only support 2D geometries
"""
import meshio
import os
import sys

## Offer default if no args passed
if (len(sys.argv) == 1):
    print(f"You have not provided any argument, exiting")
    quit()
else:
    fileName = sys.argv[1]

## Mesh with gmsh (2D)
dim = 2
if (len(sys.argv) == 3):
    dim = sys.argv[2]
    os.system(f'gmsh {fileName}.geo -{dim} -format msh2')
else:
    os.system(f'gmsh {fileName}.geo -{dim} -format msh2')


## Read in mesh
msh = meshio.read(f'{fileName}.msh')

## Parse for 2D and 3D topology
for cell in msh.cells:
    if cell.type == "triangle":
        triangle_cells = cell.data
    elif  cell.type == "tetra":
        tetra_cells = cell.data

## Scrub for physical markers
for key in msh.cell_data_dict["gmsh:physical"].keys():
    if key == "triangle":
        triangle_data = msh.cell_data_dict["gmsh:physical"][key]
    elif key == "tetra":
        tetra_data = msh.cell_data_dict["gmsh:physical"][key]

## Write to file
if (dim == 2):
    triangle_mesh = meshio.Mesh(points=msh.points[:,:2], cells=[("triangle", triangle_cells)])
    meshio.write(f"{fileName}.xdmf", triangle_mesh)
    print(f"Recommended HPC processes is : {int(round(len(triangle_cells)/40000,0))}!")

if (dim == 3):
    tet_mesh = meshio.Mesh(points=msh.point, cells=[("tetra", tetra_cells)])
    meshio.write(f"{fileName}.xdmf", tet_mesh)

print(f"Output written to {fileName}.xdmf")