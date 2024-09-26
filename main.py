import bpy
import json

# ---- CONSTANTS ---- #
#

# ---- Filepath to landmarks .json file
#
FILEPATH  = './data/world_lm_00632.json'

# ---- Chains of keypoints to connect corresponding to bones
#
CHAIN_DICT = {
    'left_arm' : [[12, 14], [14, 16]],
    'right_arm' : [[11, 13], [13, 15]],
}

# ---- FUNCIONS ---- #
#

def convert_landmark_to_blender(X_lm, width=1):
    '''
    Convert landmarks coordinates extracted with Mediapipe to the Blender coordinate system.
    
    In the Blender coordinate system:
        - The Z-axis goes from bottom to top,
        - The Y-axis goes from front to back.
    
    In the Mediapipe coordinate system:
        - The Y-axis goes from top to bottom,
        - The Z-axis goes from front to back.
    
    A ratio is also applied to the Z coordinate from the Mediapipe system.

    Parameters:
    -----------
    X_lm: tuple of float
        The 3D coordinates in the Mediapipe system of the point to convert.
    width: float, optional
        Optional ratio to apply to the Z coordinate from the Mediapipe system. Default is 1.
    
    Returns:
    --------
    xc, yc, zc: tuple of float
        The coordinates of the given point converted into the Blender coordinate system.
    '''        
    xc = X_lm[0]
    yc = X_lm[2] * width
    zc = - X_lm[1]

    return  xc, yc, zc


def get_landmarks(keypoints_set, filepath):
    '''
    Load, filter, and convert landmark coordinates from a JSON file to the Blender coordinate system.
    
    Parameters:
    -----------
    keypoints_set: set of int
        Set of selected keypoint indices to convert.
    filepath: str
        Path to the JSON file containing landmark data.
    
    Returns:
    --------
    converted_landmarks: dict
        A dictionary of converted landmarks for each frame.
    '''
    # ---- Load original dictionnary of landmarks
    #
    with open(filepath, 'r') as json_file:
        landmarks = json.load(json_file)
    
    # ---- Initialize an empty dict to collected converted landmarks
    #
    converted_landmarks = {}

    # ---- Get width of the frame from which where extracted the landmarks
    # ---- Remove width and heigh informations of the original dictionnary of landmarks
    #
    width = landmarks.pop('width')
    del landmarks['height']

    # ---- Loop over each frame
    #
    for frame_indice, frame_lm_dict in landmarks.items():

        # ---- Convert landmarks for each selected keypoint
        #
        converted_frame = { key : convert_landmark_to_blender(value, width=1.5) for key, value in frame_lm_dict.items() if int(key) in keypoints_set}
        converted_landmarks[frame_indice] = converted_frame

    return converted_landmarks


def create_bone(edit_bones, name, head, tail):
    '''
    Create and return a new bone with the specified name, head, and tail positions.
    
    Parameters:
    -----------
    edit_bones: bpy.types.ArmatureBones
        Collection of bones in edit mode.
    name: str
        Name of the new bone.
    head: tuple of float
        Coordinates of the bone's head.
    tail: tuple of float
        Coordinates of the bone's tail.
    
    Returns:
    --------
    bone: bpy.types.Bone
        The created bone.
    '''
    bone = edit_bones.new(name)
    bone.head = head
    bone.tail = tail

    return bone

def map_landmarks_with_chain(chain_dict, filepath=FILEPATH):
    '''
    Maps body chain landmarks from a file.

    Parameters:
    -----------
    chain_dict: dict
        Keys are names of body chains; values are lists of pairs of keypoint IDs representing articulation endpoints.
    filepath: str
        Path to the JSON file containing landmarks.

    Returns:
    --------
    lm_dict: dict
        Dictionary mapping chain names to their respective landmarks.
    '''
    lm_dict ={}

    # ---- Loop over each chain
    #
    for chain_name, connections_list in chain_dict.items():

        # ---- Determine the set of keypoints represented in the current chain
        #
        keypoints_set = set()
        for connection in connections_list:
            for keypoint in connection:
                keypoints_set.add(keypoint)

        # ---- Build the dictionnary of all frames with the keypoints of the current chain
        #
        lm_dict[chain_name] = get_landmarks(keypoints_set, filepath)
    
    return lm_dict


def generate_keyframe(frame_ind, lm_dict, chain_dict, chain_name):
    '''
    Generates a keyframe for Blender animation for a specified body chain.

    Parameters:
    -----------
    frame_ind: int
        Index of the frame to generate.
    lm_dict: dict
        Mapping of chain names to their landmarks.
    chain_dict: dict
        Mapping of body chain names to pairs of keypoint IDs for articulations.
    chain_name: str
        Name of the body chain to animate.
    '''
    # ---- Retrieve landmarks of the chain
    #
    lm_chain_dict = lm_dict[chain_name]

    # ---- Loop over each bone of the chain:
    #
    for i in range(len(chain_dict[chain_name])) :

        # ---- Retrieve bone in pose mode
        #
        bone_name = f'{chain_name}_{i}'
        pose_bone = armature.pose.bones.get(bone_name)

        # ---- Retrieve head for the current position
        #
        head_0=pose_bone.head

        # ---- Specify head for the final position
        #
        bone_keypoints = chain_dict.get(chain_name)[i]
        head = lm_chain_dict.get(frame_ind).get(str(bone_keypoints[0]))
        
        # ---- Translate the bone
        #
        pose_bone.location = (
            head[0] - head_0[0],
            head[1] - head_0[1],
            head[2] - head_0[2]
            )
        pose_bone.keyframe_insert(data_path="location", frame=int(frame_ind))

    # ---- Translate the extra bone
    #
    extra_bone_name = f'{chain_name}_extra'
    pose_extra_bone = armature.pose.bones.get(extra_bone_name)

    # ---- Retrieve head for the current position
    #
    head_extra_0 = pose_extra_bone.head

    # ---- Specify head for the fianl position
    # ---- Head of extra bone correspond to tail of the last bone of the chain
    #
    last_bone_keypoint = bone_keypoints
    tail_of_last_bone = lm_chain_dict.get(frame_ind).get(str(last_bone_keypoint[1]))
    head_extra = tail_of_last_bone
    
    # ---- Translate the bone
    #
    pose_extra_bone.location = (
        head_extra[0] - head_extra_0[0],
        head_extra[1] - head_extra_0[1],
        head_extra[2] - head_extra_0[2]
        )
    pose_extra_bone.keyframe_insert(data_path="location", frame=int(frame_ind))

# ---- BUILD THE SCENE ---- #
#

# ---- Initialize a new scene
#
new_scene = bpy.data.scenes.new("MyNewScene")
bpy.context.window.scene = new_scene

# ---- Initialize a armature
#
bpy.ops.object.armature_add(enter_editmode=True)
armature = bpy.context.object
armature.name = "AvatarArmature"

# ---- Switch to edit mode
#
bpy.ops.object.mode_set(mode='EDIT')

# ---- Remove the default initial bone
#
armature.data.edit_bones.remove(armature.data.edit_bones[0])

# ---- Get the lamndmarks dictionnary adapted to chain
#
lm_dict = map_landmarks_with_chain(CHAIN_DICT, FILEPATH)



# ---- INITIALIZE EACH BONE OF EACH CHAIN IN EDIT MODE ---- #
#

# ---- Loop over the chain
#
for chain_name, connections_list in CHAIN_DICT.items():

    # ---- Get the landmark dict of the current chain
    #
    lm_chain_dict = lm_dict[chain_name]

    # ---- Get indice of the first frame
    #
    frame_idx = list(lm_dict.get(chain_name).keys())
    frame_idx = list(map(int, frame_idx))
    first_frame_ind = min(frame_idx)

    #    ---- Create the bones of the current chain of the first frame
    #
    edit_bones = armature.data.edit_bones

    # ---- Loop over each connection of keypoint to realize
    #
    for i, connection in enumerate(connections_list):

        # ---- Specify head and tail of the bone and create it
        #
        head = lm_chain_dict[str(first_frame_ind)].get(str(connection[0]))
        tail = lm_chain_dict[str(first_frame_ind)].get(str(connection[1]))

        create_bone(edit_bones, f'{chain_name}_{i}', head, tail)

        # ---- Add a tiny extra bone for the extremtity of the connection.
        # ---- It's useful to control the position of the extremety of the chain
        #
        if i == len(connections_list) - 1:

            # ---- Specify head and tail of the bone and create it
            #
            head_extra = tail
            tail_extra = tuple(coordinate + 0.00001 for coordinate in head_extra)

            create_bone(edit_bones, f'{chain_name}_extra', head_extra, tail_extra)


# ---- Switch to pose mode to animate the bones in the keyframes
#
bpy.ops.object.mode_set(mode='POSE')



# ---- APPLY A STRETCH_TO CONTRAINT TO BONES OF THE SAME CHAIN ---- #
#

# ---- Loop over each chain
#
for chain_name, connections in CHAIN_DICT.items():

    # ---- Specify a strech_to contraint to force tail of previous bone of a chain to follow head of the following bone
    #
    for i, connection in enumerate(connections):

        # ---- Specify parent and target bone name
        #
        parent_bone_name = f'{chain_name}_{i}'

        if i + 1 < len(connections):
            target_bone_name = f'{chain_name}_{i+1}'
        else:
            target_bone_name = f'{chain_name}_extra'

        # ---- Retrieve the bones in pose mode
        #
        parent_pose_bone = armature.pose.bones.get(parent_bone_name)
        target_pose_bone = armature.pose.bones.get(target_bone_name)

        # ---- Apply contraint
        #
        stretch_constraint =  parent_pose_bone.constraints.new('STRETCH_TO')
        stretch_constraint.target = armature
        stretch_constraint.subtarget = target_pose_bone.name


# ---- GENERATE THE TRANSFORMATION FOR EACH FRAME ---- #
#

# --- Loop over each chain
#
for chain_name in CHAIN_DICT.keys():

    # ---- Loop over each frame
    #
    for frame_ind in lm_dict.get(chain_name).keys():

        generate_keyframe(
            frame_ind,
            lm_dict=lm_dict,
            chain_dict=CHAIN_DICT,
            chain_name=chain_name
            )


# ---- Configurate the time line to see the animation ----
#

# ---- Get the maximum value of the indice of the frame
#
max_frame_ind = 1
for chain_name in CHAIN_DICT.keys():
    frame_idx = list(lm_dict.get(chain_name).keys())
    frame_idx = list(map(int, frame_idx))
    max_ind = max(frame_idx)
    if max_ind > max_frame_ind:
        max_frame_ind = max_ind

new_scene.frame_start = 1
new_scene.frame_end = max_frame_ind
new_scene.frame_current = 1
