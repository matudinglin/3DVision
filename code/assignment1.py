import openmesh as om
import numpy as np
import heapq
import cProfile

def subdivision_loop(mesh, iterations=1):
    """
    Apply Loop subdivision to the input mesh for the specified number of iterations.
    :param mesh: input mesh
    :param iterations: number of iterations
    :return: mesh after subdivision
    """
    
    print(f"------> Original Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    
    if iterations < 1:
        print("Invalid number of iterations")
        return mesh
    
    for it in range(iterations):
        # Create a new mesh to store the subdivided mesh
        new_mesh = om.TriMesh()
        
        new_vertex_handles = {}
        # Calculate new positions for existing vertices
        for vh in mesh.vertices():
            # Compute beta
            n = mesh.valence(vh)
            beta = (40.0 - (2.0 * np.cos(2 * np.pi / n) + 3) ** 2) / (64 * n)
            # Compute new position
            new_pos = (1 - n * beta) * np.array(mesh.point(vh))
            for neighbor_vh in mesh.vv(vh):
                new_pos += beta * np.array(mesh.point(neighbor_vh))
            # Store new position and vertex handle
            new_vertex_handles[vh.idx()] = new_mesh.add_vertex(new_pos)
        
        edge_to_new_vertex = {}
        # Split every edge and create new vertices at midpoints
        for eh in mesh.edges():
            # Retrieve halfedge handles and vertices
            heh = mesh.halfedge_handle(eh, 0)
            opp_heh = mesh.opposite_halfedge_handle(heh)
            from_vh = mesh.from_vertex_handle(heh)
            to_vh = mesh.to_vertex_handle(heh)
            is_boundary = mesh.is_boundary(eh)
            # Interior edge
            if not is_boundary:
                c_vh = mesh.to_vertex_handle(mesh.next_halfedge_handle(opp_heh))
                d_vh = mesh.to_vertex_handle(mesh.next_halfedge_handle(heh))
                midpoint = 3.0 / 8.0 * (np.array(mesh.point(from_vh)) + np.array(mesh.point(to_vh))) + \
                        1.0 / 8.0 * (np.array(mesh.point(c_vh)) + np.array(mesh.point(d_vh)))
            # Boundary edge
            else:
                midpoint = 0.5 * (np.array(mesh.point(from_vh)) + np.array(mesh.point(to_vh)))
            # Add the new vertex to the new mesh and store its handle in the mapping
            new_vh = new_mesh.add_vertex(midpoint)
            edge_to_new_vertex[eh.idx()] = new_vh
        
        # Reconnect vertices to form new faces
        for fh in mesh.faces():
            # Retrieve original vertex indices of the face
            face_vhs = [vh.idx() for vh in mesh.fv(fh)]
            # Calculate edge indices for edges of the face
            edge_indices = [mesh.edge_handle(mesh.find_halfedge(mesh.vertex_handle(face_vhs[i]), 
                                                                mesh.vertex_handle(face_vhs[(i+1) % 3]))).idx() 
                            for i in range(3)]
            # Retrieve new vertex handles
            face_new_vhs = [edge_to_new_vertex[edge_idx] for edge_idx in edge_indices]
            # Retrieve original vertex handles
            face_ori_vhs = [new_vertex_handles[vh_idx] for vh_idx in face_vhs]
            # Add four new faces
            new_mesh.add_face(face_ori_vhs[0], face_new_vhs[0], face_new_vhs[2])
            new_mesh.add_face(face_ori_vhs[1], face_new_vhs[1], face_new_vhs[0])
            new_mesh.add_face(face_ori_vhs[2], face_new_vhs[2], face_new_vhs[1])
            new_mesh.add_face(face_new_vhs[0], face_new_vhs[1], face_new_vhs[2])
        
        mesh = new_mesh
        print(f"- Iteration {it + 1} - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
        
    print(f"------> New Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")

    return mesh

def simplify_quadric_error(mesh, face_count=1, subset=False):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    def calculate_edge_cost(mesh, eh):
        heh = mesh.halfedge_handle(eh, 0)
        vh0 = mesh.to_vertex_handle(heh)
        vh1 = mesh.from_vertex_handle(heh)
        K0 = mesh.vertex_property("quad", vh0)
        K1 = mesh.vertex_property("quad", vh1)
        Kij = K0 + K1
        B = np.concatenate([Kij[:3,:], np.array([0,0,0,1]).reshape(1,4)], axis=0)
        # Compute optimal collapse point x
        if np.linalg.det(B) != 0:
            x = np.matmul(np.linalg.inv(B), np.array([0, 0, 0, 1])).reshape(4, 1)
            cost = np.matmul(np.matmul(x.T, Kij), x)
            x = x.reshape(4)
        else:
            x = 0.5 * (np.array(mesh.point(vh0)) + np.array(mesh.point(vh1)))
            cost = np.matmul(np.matmul(x.T, Kij), x)
            x = np.append(x, 1)
        return cost, x

    def compute_Ki_per_vertex(mesh, vh):
        Ki = np.zeros((4, 4))
        for fh in mesh.vf(vh):
            v1 = mesh.point(mesh.to_vertex_handle(mesh.halfedge_handle(fh)))
            v2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(mesh.halfedge_handle(fh))))
            v3 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(mesh.next_halfedge_handle(mesh.halfedge_handle(fh)))))
            v_matrix = np.array([v1, v2, v3])
            temp = np.matmul(np.linalg.inv(v_matrix), np.array([[1],[1],[1]]))
            plane = np.concatenate([temp.T, np.array(-1).reshape(1, 1)], axis = 1) / (np.sum(temp**2)**0.5)
            Ki += np.matmul(plane.T, plane)
        mesh.set_vertex_property("quad", vh, Ki)
    
    print(f"------> Original Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    
    if face_count < 1 or face_count > mesh.n_faces():
        print("Invalid face count")
        return mesh
    
    # Compute Ki for each vertex
    for vh in mesh.vertices():
        compute_Ki_per_vertex(mesh, vh)
    
    # Compute edge costs
    for eh in mesh.edges():
        cost, x = calculate_edge_cost(mesh, eh)
        mesh.set_edge_property("cost", eh, cost)
        mesh.set_edge_property("x", eh, x)
    
    # Collapse edges until the target face count is reached
    while mesh.n_faces() > face_count:
        # Find the edge whose collapse minimizes the total quadric error
        min_cost = float("inf")
        min_x = None
        min_eh = None
        for eh in mesh.edges():
            cost = mesh.edge_property("cost", eh)
            if cost < min_cost:
                min_cost = cost
                min_eh = eh
                min_x = mesh.edge_property("x", eh)
        # Retrieve halfedge handle and vertices
        min_heh = mesh.halfedge_handle(min_eh, 0)
        vh0 = mesh.to_vertex_handle(min_heh)
        vh1 = mesh.from_vertex_handle(min_heh)
        # Collapse the edge
        mesh.collapse(min_heh)
        if subset:
            mesh.set_point(vh0, mesh.point(vh0))
        else:
            mesh.set_point(vh0, min_x[:3])  
        # Update quadrics for the vertices
        quad0 = mesh.vertex_property("quad", vh0)
        quad1 = mesh.vertex_property("quad", vh1)
        mesh.set_vertex_property("quad", vh0, quad0 + quad1)
        # Update the edge costs
        for eh in mesh.ve(vh0):
            cost, x = calculate_edge_cost(mesh, eh)
            mesh.set_edge_property("cost", eh, cost)
            mesh.set_edge_property("x", eh, x)
        # Remove collapsed elements
        mesh.garbage_collection()
        print(f"- Collapsed Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
        
    print(f"------> New Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    return mesh

if __name__ == '__main__':
    '''Load mesh and print information'''
    mesh = om.read_trimesh("assets/bunny.obj")
    
    # '''Apply loop subdivision over the loaded mesh'''
    # with cProfile.Profile() as pr:
    #     mesh_subdivided = subdivision_loop(mesh, iterations=3)
    #     pr.print_stats(sort='cumtime')
    # om.write_mesh("assets/assignment1/subdivision_cube_3.obj", mesh_subdivided)

    '''Apply quadratic error mesh decimation over the loaded mesh'''
    with cProfile.Profile() as pr:
        mesh_decimated = simplify_quadric_error(mesh, face_count=2000, subset=True)
        pr.print_stats(sort='cumtime')
    om.write_mesh("assets/assignment1/decimation_bunny_2000_subset.obj", mesh_decimated)