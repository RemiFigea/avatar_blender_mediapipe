import bpy
import json
import mathutils
import mediapipe as mp

# ---- CONSTANTS ---- #
# ------------------- #

LANDMARKS_FILEPATH = '/home/remifigea/code/avatar/oct24/test_lm_00632.json'

# ---- FUNCIONS ---- #
# ------------------ #

def convert_landmarks_to_blender(filepath=LANDMARKS_FILEPATH):
    '''
    Convert landmarks coordinates extracted with Mediapipe to the Blender coordinate system.
    
    In the Blender coordinate system:
        - The Z-axis goes from bottom to top,
        - The Y-axis goes from front to back.
    
    In the Mediapipe coordinate system:
        - The Y-axis goes from top to bottom,
        - The Z-axis goes from front to back.
    
    A specific scale factor for body subdivsision is also applied to the Z coordinate from the Mediapipe system.

    Parameters:
    -----------
    filepath: str
        The path to mediapipe landmarks JSON file.
    Returns:
    --------
    landmarks_dict: dictionnary
        Dictionnary of converted landmarks.
        Each landmark is a mathutils.Vector representing the converted 3D coordinates in Blender's coordinate system.
    '''        
    # ---- Load original dictionary of landmarks
    #
    with open(filepath, 'r') as json_file:
        landmarks_dict = json.load(json_file)

    # ---- Get width of the frame from which were extracted the landmarks and remove it
    #
    width = landmarks_dict.pop('width')

    # ---- We adapt y scale depending on the subsection, we adujst this rate after various trial
    y_scale_factor = {
        'pose': 0.2,
        'hand_L': 2,
        'hand_R': 2,
        'face': 1,
        }

    # ---- Remove height information from the original dictionary of landmarks
    #
    del landmarks_dict['height']

    # ---- Modify each landmark's coordinates
    #
    for body_subdivision, body_subdivision_dict in landmarks_dict.items():
        for frame_dict in body_subdivision_dict.values():
            if frame_dict:  # Only proceed if frame_dict is not empty
                for key, landmark in frame_dict.items():
                    xc = landmark[0]
                    yc = landmark[2] * y_scale_factor.get(body_subdivision) * width
                    zc = -landmark[1]
                    vector = mathutils.Vector((xc, yc, zc))

                    # ---- Assign the modified vector back to the dictionary
                    #
                    frame_dict[key] = vector

    return landmarks_dict

def get_faces(edges):
    """
    Build a set of triangles from a set of edges.

    Parameters:
    ----------
    edges : set of tuple
        Each tuple contains two integers, representing the IDs of face keypoints.

    Returns:
    -------
    faces_set : set of frozenset
        Each frozenset contains three integers, representing the IDs of face keypoints that form a triangle.
    """
    
    # ---- Initialize a dictionary to collect neighbors for each keypoint
    #
    neighbors = {}

    # ---- Populate the dictionary with neighboring keypoints
    #
    for edge in edges:
        p1, p2 = edge
        
        # ---- Add p1 and p2 as keys with empty sets if not already present
        #
        if p1 not in neighbors:
            neighbors[p1] = set()
        if p2 not in neighbors:
            neighbors[p2] = set()
        
        # ---- Add each point as a neighbor of the other
        #
        neighbors[p1].add(p2)
        neighbors[p2].add(p1)
    
    # ---- Initialize a set to store triangles
    #
    faces_set = set()
    
    # ---- Iterate over each keypoint to find triangles
    #
    for p1 in neighbors:
        for p2 in neighbors[p1]:
            for p3 in neighbors[p2]:
                # ---- Ensure p3 is not the same as p1 and is a neighbor of p1
                #
                if p3 != p1 and p3 in neighbors[p1]:
                    # ---- Create a triangle and add it to the set
                    #
                    triangle = frozenset([p1, p2, p3])
                    faces_set.add(triangle)

    return faces_set


# ---- EXECUTIONS ---- #
# -------------------- #

# ---- Load landmark data from JSON file and convert it to Blender coordinate reference
#
landmarks = convert_landmarks_to_blender(LANDMARKS_FILEPATH)
face_landmarks = landmarks.get('face')

# ---- Retrieve predefined edges from Mediapipe for facial tessellation
#      FACEMESH_TESSELATION provides a list of edges connecting facial keypoints.
#
edges = mp.solutions.face_mesh.FACEMESH_TESSELATION

# ---- Generate a list of triplets of integers representing triangles
#      Each triplet corresponds to indices of facial keypoints that form the triangular faces of the 3D mesh.
#
faces_set = get_faces(edges)

# ---- Convert the set of face indices into a list format
#      This prepares the triangles (triplets of indices) for Blender's mesh structure.
#
faces_list = [list(face) for face in faces_set]

# ---- Inititialize a the mesh and the associated object
#
mesh = bpy.data.meshes.new(name="Face_Mesh")
obj = bpy.data.objects.new("Face", mesh)

# ---- Link object to the scene
#
bpy.context.collection.objects.link(obj)


# ---- Determine face keypoints for the first frame
#
face_frame_idx = list(map(int, face_landmarks.keys()))
min_face_frame_ind = min(face_frame_idx)
face_keypoints_first_frame = list(list(face_landmarks.get(str(min_face_frame_ind)).values()))

# ---- Create the mesh from the keypoints corresponding to the first frame
#
mesh.from_pydata(face_keypoints_first_frame, [], faces_list)

# ---- Update the mesh
#
mesh.update()