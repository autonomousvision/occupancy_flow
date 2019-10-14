import os
from plyfile import PlyElement, PlyData
import numpy as np
from trimesh.util import array_to_string
import trimesh

def export_pointcloud(vertices, out_file, as_text=True):
    assert(vertices.shape[1] == 3)
    vertices = vertices.astype(np.float32)
    vertices = np.ascontiguousarray(vertices)
    vector_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertices = vertices.view(dtype=vector_dtype).flatten()
    plyel = PlyElement.describe(vertices, 'vertex')
    plydata = PlyData([plyel], text=as_text)
    plydata.write(out_file)


def load_pointcloud(in_file):
    plydata = PlyData.read(in_file)
    vertices = np.stack([
        plydata['vertex']['x'],
        plydata['vertex']['y'],
        plydata['vertex']['z']
    ], axis=1)
    return vertices


def read_off(file):
    """
    Reads vertices and faces from an off file.

    :param file: path to file to read
    :type file: str
    :return: vertices and faces as lists of tuples
    :rtype: [(float)], [(int)]
    """

    assert os.path.exists(file), 'file %s not found' % file

    with open(file, 'r') as fp:
        lines = fp.readlines()
        lines = [line.strip() for line in lines]

        # Fix for ModelNet bug were 'OFF' and the number of vertices and faces
        # are  all in the first line.
        if len(lines[0]) > 3:
            assert lines[0][:3] == 'OFF' or lines[0][:3] == 'off', \
                   'invalid OFF file %s' % file

            parts = lines[0][3:].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 1
        # This is the regular case!
        else:
            assert lines[0] == 'OFF' or lines[0] == 'off', \
                   'invalid OFF file %s' % file

            parts = lines[1].split(' ')
            assert len(parts) == 3

            num_vertices = int(parts[0])
            assert num_vertices > 0

            num_faces = int(parts[1])
            assert num_faces > 0

            start_index = 2

        vertices = []
        for i in range(num_vertices):
            vertex = lines[start_index + i].split(' ')
            vertex = [float(point.strip()) for point in vertex if point != '']
            assert len(vertex) == 3

            vertices.append(vertex)

        faces = []
        for i in range(num_faces):
            face = lines[start_index + num_vertices + i].split(' ')
            face = [index.strip() for index in face if index != '']

            # check to be sure
            for index in face:
                assert index != '', \
                      'found empty vertex index: %s (%s)' \
                      % (lines[start_index + num_vertices + i], file)

            face = [int(index) for index in face]

            assert face[0] == len(face) - 1, \
                'face should have %d vertices but as %d (%s)' \
                % (face[0], len(face) - 1, file)
            assert face[0] == 3, \
                'only triangular meshes supported (%s)' % file
            for index in face:
                assert index >= 0 and index < num_vertices, \
                    'vertex %d (of %d vertices) does not exist (%s)' \
                    % (index, num_vertices, file)

            assert len(face) > 1

            faces.append(face)

        return vertices, faces

    assert False, 'could not open %s' % file


def save_mesh(mesh, out_file, digits=10, face_colors=None):
    digits = int(digits)
    # prepend a 3 (face count) to each face
    if face_colors is None:
        faces_stacked = np.column_stack((
            np.ones(len(mesh.faces)) * 3, mesh.faces)).astype(np.int64)
    else:
        mesh.visual.face_colors = face_colors
        assert(mesh.visual.face_colors.shape[0] == mesh.faces.shape[0])
        faces_stacked = np.column_stack((
            np.ones(len(mesh.faces)) * 3, mesh.faces,
            mesh.visual.face_colors[:, :3])).astype(np.int64)
    export = 'OFF\n'
    # the header is vertex count, face count, edge number
    export += str(len(mesh.vertices)) + ' ' + str(len(mesh.faces)) + ' 0\n'
    export += array_to_string(
        mesh.vertices, col_delim=' ', row_delim='\n', digits=digits) + '\n'
    export += array_to_string(faces_stacked, col_delim=' ', row_delim='\n')
    
    with open(out_file, 'w') as f:
        f.write(export)

    return mesh


def load_mesh(mesh_file):
    with open(mesh_file, 'r') as f:
        str_file = f.read().split('\n')
        n_vertices, n_faces, _ = list(
            map(lambda x: int(x), str_file[1].split(' ')))
        str_file = str_file[2:]  # Remove first 2 lines

        v = [l.split(' ') for l in str_file[:n_vertices]]
        f = [l.split(' ') for l in str_file[n_vertices:]]

    v = np.array(v).astype(np.float32)
    f = np.array(f).astype(np.uint64)[:, 1:4]

    mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)

    return mesh