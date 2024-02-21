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

def simplify_quadric_error(mesh, face_count=1):
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
        # if np.linalg.det(B) > 0:
        #     x = np.matmul(np.linalg.inv(B), np.array([0, 0, 0, 1])).reshape(4, 1)
        #     cost = np.matmul(np.matmul(x.T, Kij), x)
        #     x = x.reshape(4)
        # else:
        #     # x = 0.5 * (np.array(mesh.point(vh0)) + np.array(mesh.point(vh1)))
        #     # x = np.append(x, 1)
        #     # cost = np.matmul(np.matmul(x.T, Kij), x)
        #     v_1 = np.append(mesh.point(vh0), 1).reshape(4, 1)
        #     v_2 = np.append(mesh.point(vh1), 1).reshape(4, 1)
        #     v_mid = (v_1 + v_2) / 2
            
        #     delta_v_1 = np.matmul(np.matmul(v_1.T, Kij), v_1)
        #     delta_v_2 = np.matmul(np.matmul(v_2.T, Kij), v_2)
        #     delta_v_mid = np.matmul(np.matmul(v_mid.T, Kij), v_mid)
            
        #     errors = np.array([delta_v_1, delta_v_2, delta_v_mid]).flatten()  # Flatten for argmin
        #     min_error_index = np.argmin(errors)
            
        #     positions = np.hstack([v_1, v_2, v_mid])  # Stack positions side by side
        #     current_v_opt = positions[:, min_error_index].reshape(4)
        #     x = current_v_opt[:3] 
        #     x = np.append(x, 1)
            
        #     v_1=np.append(mesh.point(vh0),1).reshape(4,1)
        #     v_2=np.append(mesh.point(vh1),1).reshape(4,1)
        #     v_mid=(v_1+v_2)/2
        #     delta_v_1=np.matmul(np.matmul(v_1.T, Kij), v_1)
        #     delta_v_2=np.matmul(np.matmul(v_2.T, Kij), v_2)
        #     delta_v_mid=np.matmul(np.matmul(v_mid.T, Kij), v_mid)
        #     cost=np.min(np.array([delta_v_1, delta_v_2, delta_v_mid]))
        #     min_delta_loc=np.argmin(np.array([delta_v_1, delta_v_2, delta_v_mid]))
        #     x=np.concatenate([v_1,v_2,v_mid],axis=1)[:,min_delta_loc].reshape(4)
        x = 0.5 * (np.array(mesh.point(vh0)) + np.array(mesh.point(vh1)))
        x = np.append(x, 1)
        cost = np.matmul(np.matmul(x.T, Kij), x)
        return cost, x

    def update_edge_costs(edge_costs, mesh, vh0, vh1):
        for i, (cost, eh, x, valid) in enumerate(edge_costs):
            if mesh.from_vertex_handle(mesh.halfedge_handle(eh, 0)) == vh1 or \
               mesh.to_vertex_handle(mesh.halfedge_handle(eh, 0)) == vh1 or \
               mesh.from_vertex_handle(mesh.halfedge_handle(eh, 0)) == vh0 or \
               mesh.to_vertex_handle(mesh.halfedge_handle(eh, 0)) == vh0:
                edge_costs[i] = (cost, eh, x, False)
            
        for eh in mesh.ve(vh0):
            cost, x = calculate_edge_cost(mesh, eh)
            heapq.heappush(edge_costs, (cost, eh, x, True))
        

    def compute_Ki_per_vertex(mesh, vh):
        Ki = np.zeros((4, 4))
        for fh in mesh.vf(vh):
            # norm = np.array(mesh.calc_face_normal(fh))
            # norm = norm / np.linalg.norm(norm)
            # d = -np.dot(norm, np.array(mesh.point(vh)))
            # plane = np.append(norm, d)
            point_1 = mesh.point(mesh.to_vertex_handle(mesh.halfedge_handle(fh)))
            point_2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(mesh.halfedge_handle(fh))))
            point_3 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(mesh.next_halfedge_handle(mesh.halfedge_handle(fh)))))
            point_mat=np.array([point_1, point_2, point_3])
            abc=np.matmul(np.linalg.inv(point_mat), np.array([[1],[1],[1]]))
            plane = np.concatenate([abc.T, np.array(-1).reshape(1, 1)], axis=1)/(np.sum(abc**2)**0.5)
            Ki += np.matmul(plane.T, plane)
        mesh.set_vertex_property("quad", vh, Ki)  
    
    print(f"------> Original Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    
    if face_count < 1 or face_count > mesh.n_faces():
        print("Invalid face count")
        return mesh
    
    # Compute Ki for each vertex
    for vh in mesh.vertices():
        compute_Ki_per_vertex(mesh, vh)
    
    # Create a priority queue of edges based on the quadric error
    edge_costs = []
    heapq.heapify(edge_costs)
    for eh in mesh.edges():
        cost, x = calculate_edge_cost(mesh, eh)
        heapq.heappush(edge_costs, (cost, eh, x, True))
    
    # Collapse edges until the target face count is reached
    while mesh.n_faces() > face_count:
        # Find the edge whose collapse minimizes the total quadric error
        _, eh, x, valid = heapq.heappop(edge_costs)
        heh = mesh.halfedge_handle(eh, 0)
        vh0 = mesh.to_vertex_handle(heh)
        vh1 = mesh.from_vertex_handle(heh)
        if not mesh.is_collapse_ok(heh) or not valid:
            print(f"Invalid collapse: {vh0.idx()} -> {vh1.idx()}")
            continue
        # Collapse the edge
        mesh.collapse(heh)
        mesh.set_point(vh0, x[:3])
        # Update quadrics for the vertices
        quad0 = mesh.vertex_property("quad", vh0)
        quad1 = mesh.vertex_property("quad", vh1)
        mesh.set_vertex_property("quad", vh0, quad0 + quad1)
        # Update the edge costs
        update_edge_costs(edge_costs, mesh, vh0, vh1)
        # Remove collapsed elements
        mesh.garbage_collection()
        print(f"- Collapsed Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
        
    print(f"------> New Mesh - Vertices: {mesh.n_vertices()}, Faces: {mesh.n_faces()}, Edges: {mesh.n_edges()}")
    return mesh

if __name__ == '__main__':
    '''Load mesh and print information'''
    # mesh = trimesh.load_mesh('assets/cube.obj')
    # mesh = trimesh.creation.box(extents=[1, 1, 1])
    mesh = om.read_trimesh("assets/bunny.obj")
    
    '''Apply loop subdivision over the loaded mesh'''
    # mesh_subdivided = mesh.subdivide_loop(iterations=1)
    # mesh_subdivided = subdivision_loop(mesh, iterations=1)
    
    '''Save the subdivided mesh'''
    # mesh_subdivided.export('assets/assignment1/cube_subdivided.obj')
    #om.write_mesh("assets/assignment1/cube_subdivided.obj", mesh_subdivided)

    '''Apply quadratic error mesh decimation over the loaded mesh'''
    # mesh_decimated = mesh.simplify_quadric_decimation(4)
    # with cProfile.Profile() as pr:
    mesh_decimated = simplify_quadric_error(mesh, face_count=1000)
    #     pr.print_stats()
    
    '''Save the decimated mesh'''
    # mesh_decimated.export('assets/assignment1/cube_decimated.obj')
    om.write_mesh("assets/cube_decimated.obj", mesh_decimated)