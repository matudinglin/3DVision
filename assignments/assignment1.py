import openmesh as om
import numpy as np
import heapq

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
        # Step 1: Calculate new positions for existing vertices
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
        # Step 2: Split every edge and create new vertices at midpoints
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
        
        # Step 3: Reconnect vertices to form new faces
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

def simplify_quadric_error(mesh, face_count=1):
    """
    Apply quadratic error mesh decimation to the input mesh until the target face count is reached.
    :param mesh: input mesh
    :param face_count: number of faces desired in the resulting mesh.
    :return: mesh after decimation
    """
    def create_edge_costs(edge_costs, quadrics, mesh):
        edge_costs.clear()
        for eh in mesh.edges():
            # Set Kij = Ki+ Kj
            heh = mesh.halfedge_handle(eh, 0)
            vh0 = mesh.to_vertex_handle(heh)
            vh1 = mesh.from_vertex_handle(heh)
            Kij = quadrics[vh0.idx()] + quadrics[vh1.idx()]
            # TODO: Compute optimal collapse point x and cost
            # Compute optimal collapse point x and cost
            # If Kij is not invertible, set x as midpoint of the edge
            # if np.linalg.det(Kij) == 0:
            #     x = 0.5 * (np.array(mesh.point(vh0)) + np.array(mesh.point(vh1)))
            #     x = np.append(x, 1)
            # else:
            #     w = np.array([0, 0, 0, 1]).T
            #     x = np.dot(np.linalg.inv(Kij), w)
            x = 0.5 * (np.array(mesh.point(vh0)) + np.array(mesh.point(vh1)))
            x = np.append(x, 1)
            cost = np.dot(x.T, np.dot(Kij, x))
            edge_costs.append((cost, heh, x, vh0, vh1))
        
    def compute_ki(mesh, quadrics):
        for fh in mesh.faces():
            norm = np.array(mesh.calc_face_normal(fh))
            norm = np.append(norm, 0)
            K = np.outer(norm, norm)
            for vh in mesh.fv(fh):
                quadrics[vh.idx()] += K
    
    print(f"------> Original Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    
    if face_count < 1 or face_count > mesh.n_faces():
        print("Invalid face count")
        return mesh
    
    # Compute Ki for each vertex
    quadrics = {vh.idx(): np.zeros((4, 4)) for vh in mesh.vertices()}
    compute_ki(mesh, quadrics)
    
    # Create a priority queue of edges based on the quadric error
    edge_costs = []
    heapq.heapify(edge_costs)
    create_edge_costs(edge_costs, quadrics, mesh)
    
    # Collapse edges until the target face count is reached
    while mesh.n_faces() > face_count:
        # Find the edge whose collapse minimizes the total quadric error
        _, heh, x, vh0, vh1 = heapq.heappop(edge_costs)
        # Collapse the edge
        if not mesh.is_collapse_ok(heh):
            heapq.heappop(edge_costs)
            continue
        mesh.collapse(heh)
        mesh.garbage_collection()
        mesh.set_point(vh0, x[:3])
        print(f"- Collapsed Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
        # Update the quadric for the new vertex
        quadrics[vh0.idx()] = quadrics[vh0.idx()] + quadrics[vh1.idx()]
        quadrics.pop(vh1.idx())
        # Recreate the edge costs
        # TODO: Update edge costs
        quadrics = {vh.idx(): np.zeros((4, 4)) for vh in mesh.vertices()}
        compute_ki(mesh, quadrics)
        create_edge_costs(edge_costs, quadrics, mesh)
        
    print(f"------> New Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    return mesh

if __name__ == '__main__':
    '''Load mesh and print information'''
    # mesh = trimesh.load_mesh('assets/cube.obj')
    # mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh = om.read_trimesh("assets/bunny.obj")
    
    '''Apply loop subdivision over the loaded mesh'''
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)
    #mesh_subdivided = subdivision_loop(mesh, iterations=2)
    
    '''Save the subdivided mesh'''
    # mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    #om.write_mesh("assets/assignment1/cube_subdivided.obj", mesh_subdivided)

    '''Apply quadratic error mesh decimation over the loaded mesh'''
    # mesh_decimated = mesh.simplify_quadric_decimation(4)
    mesh_decimated = simplify_quadric_error(mesh, face_count=500)
    
    '''Save the decimated mesh'''
    # mesh_decimated.export('assets/assignment1/cube_decimated.obj')
    om.write_mesh("assets/assignment1/cube_decimated.obj", mesh_decimated)