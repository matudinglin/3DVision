import openmesh as om
import numpy as np

mesh = om.TriMesh()

cube = om.read_trimesh('assets/cube.obj')

# collapse one edge
vh0 = cube.vertex_handle(0)
vh1 = cube.vertex_handle(1)


om.write_mesh('assets/modified_cube.obj', cube)