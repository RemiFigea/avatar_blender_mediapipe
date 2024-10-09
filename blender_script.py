import bpy
import json
import mathutils


# ---- CONSTANTS ---- #
# ------------------- #


# ---- Filepath to landmarks .json file
#
#PROJECT_DIRPATH = #path to the local directory where the repository is cloned
LANDMARKS_FILEPATH = PROJECT_DIRPATH + '//data//test_lm_00632.json'
MODEL_FILEPATH = PROJECT_DIRPATH + '//data//original_model.blend'

# ---- Chains of keypoints to connect to build a bone
# ---- Keypoint Ids reference is Mediapipe
#
CHAIN_DICT = {
    'pose' : {
        'arm_L' : [[12, 14], [14, 16]],
        'arm_R' : [[11, 13], [13, 15]],
        # 'pinky_L': [[16 , 18]],
        # 'index_L': [[16, 20]],
        # 'thumb_L': [[16 , 22]],
        # 'pinky_R':[[15, 17]],
        # 'index_R':[[15 , 19]],
        # 'thumb_R':[[15, 21]],
    },
    'hand_L' : {
        'thumb_hd_L' : [[0, 1], [1, 2], [2, 3], [3, 4]],
        'index_hd_L' : [[0, 5], [5, 6], [6, 7], [7, 8]],
        'middle_hd_L' : [[0, 9], [9, 10], [10, 11], [11, 12]],
        'ring_hd_L': [[0, 13], [13, 14], [14, 15], [15, 16]],
        'pinky_hd_L': [[0, 17], [17, 18], [18, 19], [19, 20]],
    },
    'hand_R' : {
        'thumb_hd_R' : [[0, 1], [1, 2], [2, 3], [3, 4]],
        'index_hd_R' : [[0, 5], [5, 6], [6, 7], [7, 8]],
        'middle_hd_R' : [[0, 9], [9, 10], [10, 11], [11, 12]],
        'ring_hd_R': [[0, 13], [13, 14], [14, 15], [15, 16]],
        'pinky_hd_R': [[0, 17], [17, 18], [18, 19], [19, 20]],
    }
}

# ---- Dictionary containing face accessory information.
# ---- Each accessory (right eye, left eye, hair) is associated with a target vertex group.
# ---- These vertex groups ('Eye_R_Target', 'Eye_L_Target', 'Hair_Target') were defined on the original model.
# ---- They serve as reference points for calculating the relative offset of the accessories from the face.
#
FACE_ACCESSORIES_DICT = {
    'eye_R': {'target_vertex_group': 'Eye_R_Target'},
    'eye_L': {'target_vertex_group': 'Eye_L_Target'},
    'hair': {'target_vertex_group':'Hair_Target'},
}



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


def adjust_hand_to_arm_z_position(landmarks):
    '''
    Align the left and right hand landmarks with the corresponding arm extremities.
    
    Parameters:
    -----------
    landmarks: dict
        Dictionary containing sub-dictionaries for pose, left hand, and right hand landmarks. 
        Each sub-dictionary maps Mediapipe keypoint IDs to their respective vector of coordinates in Blender reference.
    
    Returns:
    --------
    landmarks: dict
         Updated 'landmarks' dictionary with adjusted hand positions (left and right hands).
    '''
    # ---- Adapt coordinates of the hands to match arm position
    #
    for hand, pose_keypoint_ref in [('hand_L', '16'), ('hand_R', '15')]:

        for frame_ind, frame_lm_dict in landmarks[hand].items():
            # ---- Set landmarks of hand keypoint with ID 0 to the value of landmarks of pose keypoint with ID 16
            #
            if landmarks['pose'][frame_ind].get(pose_keypoint_ref) is not None and landmarks[hand][frame_ind].get('0') is not None:
                translation_vector = landmarks['pose'][frame_ind].get(pose_keypoint_ref) - landmarks[hand][frame_ind].get('0')
                landmarks[hand][frame_ind]['0'] = landmarks['pose'][frame_ind].get(pose_keypoint_ref)
            
                # ---- Adjust each hand landmarks of the frame
                #
                for keypoint in frame_lm_dict.keys():
                    if keypoint != '0' and landmarks[hand][frame_ind].get(keypoint) is not None:
                        landmarks[hand][frame_ind][keypoint] = landmarks[hand][frame_ind].get(keypoint) + translation_vector
            else:
                landmarks[hand][frame_ind] = {}
            
    return landmarks


def get_scale_factor_and_translation_vector(armature, pose_landmarks, reference_bone_name):
    '''
    Calculate the scale factor and translation vector to align a specified bone 
    in the given armature with the corresponding bone in the pose_landmarks.

    Parameters:
    -----------
    armature: bpy.types.Armature
        The armature of the model to map landmarks on.
    pose_landmarks: dict
        The landmarks converted to Blender coordinates system, used to generate the new armature.
        Landmarks type should be mathutils.Vector.
    bone_name: str
        The name of the reference bone to match between the two armatures.

    Returns:
    --------
    scale_rate: float
        The factor to scale the generated bone.
    translation_vector: mathutils.Vector
        The vector to translate the generated bone to align with the original.
    '''
    # ---- Retrieve information about the bone named bone_name of the given original model armature
    #
    armature_bone = armature.data.bones.get(reference_bone_name)
    
    if armature_bone:
        # ---- Get the length of the bone
        #
        original_model_bone_length = armature_bone.length
    
    # ---- From landmarks converted to blender system of coordinate, calculate the bone length of the same bone in the first frame configuration ---- #
    
    # ---- Retrieve the frame 
    # --- Reach the first frame with landmarks
    #
    frame_idx_sorted = sorted(list(map(int, pose_landmarks.keys())))

    # ---- Loop over each frame
    #
    for frame_ind in frame_idx_sorted:

        # ---- Check if the frame as value for the given bone
        #
        if pose_landmarks.get(str(frame_ind)) != {}:

            # ---- Calculate bone lenght
            #
            head = pose_landmarks[str(frame_ind)].get('12')
            tail = pose_landmarks[str(frame_ind)].get('14')
            landmarks_bone_length = (head-tail).length

            break
    
    # ---- Determine the rate to apply to scale the given bone created by landmarks to the size of the same bone in the original model
    #
    scale_rate = original_model_bone_length/landmarks_bone_length


    # ---- Determine the translation vector to apply to the scaled bone generated with landmarks to match the head of the generated bone with the head of the same bone of th original model
    #
    scaled_bone_head = scale_rate * head
    armature_bone_head = armature_bone.head

    translation_vector = armature_bone_head - scaled_bone_head + mathutils.Vector((0, - 0.02, 0)) # ----we had a little offset after checking the animation

    return scale_rate, translation_vector


def get_mapped_landmarks(armature, filepath=LANDMARKS_FILEPATH, chain_dict=CHAIN_DICT):
    '''
    Maps body chain landmarks from a file.

    Parameters:
    -----------
    armature: bpy.types.Armature
        The armature of the model to map landmarks on.
    filepath: str
        Path to the JSON file containing landmarks.
    chain_dict: dict
        Keys are names of body chains; values are lists of pairs of keypoint IDs representing articulation endpoints.

    Returns:
    --------
    lm_dict: dict
        Dictionary mapping chain names to their respective landmarks.
    '''
    # ---- Initialize an empty dictionnary to collect reorganized and resized landmarks
    #
    lm_dict = {}

    # ---- Convert the landmarks to Blender system of coordinates
    #
    converted_landmarks = convert_landmarks_to_blender(filepath)

    # ---- Adjust hand landmarks position to pose landmarks position
    #
    adjusted_landmarks = adjust_hand_to_arm_z_position(converted_landmarks)

    # ---- Build a dictionnary with subdictionnary specific to each chain ---- #

    # ---- At the same time, rescale and apply translation to landmarks to match with size and position of the original model ---- #
    
    pose_landmarks = adjusted_landmarks.get('pose')

    # ---- Set the reference bone name to arm_L_0 bone is used to determine scale factor and translation vector
    #
    reference_bone_name = 'arm_L_0'
    # ---- Calculate scale factor and translation vector to adjust landmarks to original model armature size and position
    #
    scale_factor, translation_vector = get_scale_factor_and_translation_vector(armature, pose_landmarks, reference_bone_name)
    # ----  Loop over each subsection
    #
    for body_subdivision, chain_sub_dict in chain_dict.items():

        # ---- Loop over each chain
        #
        for chain_name, connections_list in chain_sub_dict.items():

            # ---- Initialize an empty subdictionnary
            #
            lm_dict[chain_name] = {}

            # ---- Determine the set of keypoints represented in the current chain
            #
            keypoints_set = set()
            for connection in connections_list:
                for keypoint in connection:
                    keypoints_set.add(str(keypoint))

            # ---- Buil the subdictionnary for the chain with adjust landmarked to the size and location of original model armature
            #
            for frame, lm_frame_dict in adjusted_landmarks.get(body_subdivision).items():

                
                if lm_frame_dict == {}: # Only process if not empty
                    continue
                lm_dict[chain_name][frame] = {}

                for keypoint, landmarks in lm_frame_dict.items():
                    # ---- Check if the keypoint correpond to the current chain
                    #
                    if keypoint in keypoints_set:
                        lm_dict[chain_name][frame][keypoint] = landmarks * scale_factor + translation_vector
                
                if lm_dict[chain_name][frame] == {}:
                    del lm_dict[chain_name][frame]

    # ---- We will use shape key to animated the face, therefore we don't need to rescale face landmarks
    #
    face_landmarks = adjusted_landmarks.get('face')
    lm_dict['face'] = {}
    
    for frame, lm_frame_dict in face_landmarks.items():
        if lm_frame_dict == {}:
            continue
        lm_dict['face'][frame] = lm_frame_dict

    return lm_dict


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


def set_to_invisible(chain_dict, armature, chain_name, choosen_frame_ind):
    '''
    Sets bones in a specified chain to invisible in the given armature from frame 1 to `choosen_frame_ind` (included). 
    Bones containing 'extra' in their name are hidden for all frames, as they are used only for animation support.

    Parameters:
    -----------
    chain_dict: dict
        Keys are names of body chains; values are lists of pairs of keypoint IDs representing articulation endpoints.
    armature: bpy.types.Armature
        The armature where visibility modifications will be applied.
    chain_name: str
        The name of the bone chain to process.
    frame_ind: str or int
        Frame number from which bones (except 'extra' bones) will become visible.

    Returns:
    --------
    None
    '''
    for i in range(len(chain_dict[chain_name])) :

        # ---- Retrieve bone in pose mode
        #
        bone_name = f'{chain_name}_{i}'
        pose_bone = armature.pose.bones.get(bone_name)
        
        # ---- Set bone to invisible from frame 1
        #
        pose_bone.bone.hide = True
        pose_bone.bone.keyframe_insert(data_path="hide", frame=1)

        # ---- Set bone to visible from frame frame_ind + 1
        #
        pose_bone.bone.hide = False
        pose_bone.bone.keyframe_insert(data_path="hide", frame=int(choosen_frame_ind)+1)

    # ---- Set to invisible extra bone for all frame
    #
    if (chain_name != 'arm_L') and (chain_name != 'arm_R'):
        # ---- Retrieve extra bone in pose bone
        #
        extra_bone_name = f'{chain_name}_extra'
        pose_extra_bone = armature.pose.bones.get(extra_bone_name)

        # ---- Set extra bone to invisible from frame 1 
        #
        pose_extra_bone.bone.hide = True
        pose_extra_bone.bone.keyframe_insert(data_path="hide", frame=1)


def generate_armature_keyframe(frame_ind, lm_dict, chain_dict, chain_name):
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

    bpy.context.scene.frame_set(int(frame_ind))

    # ---- Retrieve landmarks of the chain
    #
    lm_chain_dict = lm_dict[chain_name]

    # ---- Set bones to invisible and apply no transition from all previous frame ---- #

    # ---- Loop over each bone of the chain:
    #
    for i in range(len(chain_dict[chain_name])) :

        # ---- Retrieve bone in pose mode
        #
        bone_name = f'{chain_name}_{i}'
        pose_bone = armature.pose.bones.get(bone_name)
       

        # ---- Specify head for the final position
        #
        bone_keypoints = chain_dict.get(chain_name)[i]
        head_final = lm_chain_dict.get(frame_ind).get(str(bone_keypoints[0]))
       
        
        expected_final_position_arm_ref = head_final
        
        
        pose_bone.matrix.translation = expected_final_position_arm_ref


        pose_bone.keyframe_insert(data_path="location", frame=int(frame_ind))

         

    if (chain_name != 'arm_L') and (chain_name != 'arm_R'):

        # ---- Translate the extra bone
        #
        extra_bone_name = f'{chain_name}_extra'
        pose_extra_bone = armature.pose.bones.get(extra_bone_name)

        # ---- Specify head for the final position
        # ---- Head of extra bone correspond to tail of the last bone of the chain
        #
        last_bone_keypoint = bone_keypoints
        tail_of_last_bone = lm_chain_dict.get(frame_ind).get(str(last_bone_keypoint[1]))
        head_extra = tail_of_last_bone

        expected_head_extra_final_arm_ref = head_extra

        pose_extra_bone.matrix.translation = expected_head_extra_final_arm_ref

        pose_extra_bone.keyframe_insert(data_path="location", frame=int(frame_ind))


def get_vertex_index(mesh_obj, vertex_group_name):
    '''
    Returns the index of the single vertex in the specified vertex group for the given mesh object.
    
    Parameters:
    -----------
    mesh_obj : bpy.types.Object
        The mesh object containing the vertex group.
    vertex_group_name : str
        The name of the vertex group, which should contain only one vertex.
    
    Returns:
    --------
    vertex_ind : int
        The index of the vertex in the vertex group. If the group is not found or empty, returns None.
    '''
    # ---- Retrieve vertex group
    #
    vertex_group = mesh_obj.vertex_groups.get(vertex_group_name)
    
    # ----  Check if vertex group exist
    #
    if vertex_group:
        # ----  Loop over each vertex of the mesh object
        #
        for vertex in mesh_obj.data.vertices:
            # ---- Loop over each group the vertex belongs to ----
            #
            for group in vertex.groups:
                # ---- Select vertex index if the group is in vertex_group
                #
                if group.group == vertex_group.index:
                    vertex_ind = vertex.index
                    break
    else:
        print("Target vertex group does not exist.")
    
    return vertex_ind


def get_vertex_coords(obj, vertex_ind):
    '''
    Returns the world coordinates of a vertex from a mesh object, taking into account
    modifiers and shape keys.

    Parameters:
    ----------
    obj: bpy.types.Object
        The Blender object containing the mesh.
    vertex_ind: int
        The index of the vertex whose coordinates are to be retrieved.

    Returns:
    -------
    co_world: mathutils.Vector
        The world coordinates of the specified vertex.
    '''
    # ---- Get the Dependency Graph to evaluate the object with applied modifiers
    #
    depsgraph = bpy.context.evaluated_depsgraph_get()

    # ---- Get the evaluated object (with modifiers and shape keys applied)
    #
    obj_eval = obj.evaluated_get(depsgraph)

    # ---- Retrieve the vertex from the evaluated object's data
    #
    vertex = obj_eval.data.vertices[vertex_ind]

    # ---- Convert local coordinates to world coordinates using the object's transformation matrix
    #
    co_world = obj_eval.matrix_world @ vertex.co
    
    return co_world


def apply_stretch_to_contraints(armature, chain_dict=CHAIN_DICT):
    '''
    Apply a "Stretch To" constraint to bones in a specified chain, forcing the tail of each bone to follow the head of the next bone in the chain.
    If the bone name is `arm_L_1` or `arm_R_1`, the constraint will force the bone to follow the corresponding hand (left or right).

    Parameters:
    -----------
    armature : bpy.types.Object
        The armature containing the bones to which the constraints will be applied.
    chain_dict: dict
        Keys are names of body chains; values are lists of pairs of keypoint IDs representing articulation endpoints.
    Returns
    -------
    None
    '''
    # ---- Switch to pose mode to apply contraint the each bone
    #
    bpy.ops.object.mode_set(mode='POSE')

    # ---- Apply strech_to contraint to bones of the same chain
    #
    for chain_sub_dict in chain_dict.values():

        # ---- Loop over each chain
        #
        for chain_name, connections in chain_sub_dict.items():

            # ---- Loop over each bone of a chain
            #
            for i in range(len(connections)):

                # ---- If bone is the last of the left arm, it should follow the left hand
                #
                if chain_name == 'arm_L' and i == 1:
                    parent_bone_name = f'{chain_name}_{i}'
                    target_bone_name = 'thumb_hd_L_0'

                # ---- If bone is the last of the right arm, it should follow the right hand
                #
                elif  chain_name == 'arm_R' and i == 1:
                    parent_bone_name = f'{chain_name}_{i}'
                    target_bone_name = 'thumb_hd_R_0'

                else:
                    # ---- Specify parent and target bone name
                    #
                    parent_bone_name = f'{chain_name}_{i}'

                    # --- Check if the bone is the last one of the current chain
                    #
                    if i + 1 < len(connections):
                        target_bone_name = f'{chain_name}_{i+1}'

                    # --- The last bone of a chain follow an extra bone we will set to invisible but useful to manage tail position of the last bone
                    #
                    else:
                        target_bone_name = f'{chain_name}_extra'

                # ---- Retrieve the bones in pose mode
                #
                parent_pose_bone = armature.pose.bones.get(parent_bone_name)
                target_pose_bone = armature.pose.bones.get(target_bone_name)

                # ---- Apply contraint
                #
                stretch_constraint =  parent_pose_bone.constraints.new('STRETCH_TO')
                stretch_constraint.volume = 'NO_VOLUME'
                stretch_constraint.target = armature
                stretch_constraint.subtarget = target_pose_bone.name


def adapt_face_landmarks_to_model(face_mesh_obj, face_landmarks):
    '''
    Adjust face landmarks to align the 'Chin_Target' vertex in both the landmarks and the given mesh object.

    Parameters:
    -----------
    face_mesh_obj: bpy.types.Object
        The Blender mesh object used to align the face landmarks.
    face_landmarks: dict
        Dictionary of face landmarks with each keypoint as a vector.

    Returns:
    --------
    None
    '''
    chin_vertex_id = get_vertex_index(face_mesh_obj, 'Chin_Target')

    # ---- Retrieve world and local location of the face in the original model
    #
    world_face_original_location = get_vertex_coords(face_mesh_obj, chin_vertex_id)
    local_face_original_location = face_mesh_obj.matrix_world.inverted() @ world_face_original_location

    # ---- Get the first frame indice
    #
    face_frame_idx = list(map(int, face_landmarks.keys()))
    first_face_lm_frame_ind = min(face_frame_idx)
    first_face_lm_frame_dict = face_landmarks.get(str(first_face_lm_frame_ind))

    # ---- Get location of the face for the first frame indice (reference vertex is the bottom ot the chin)
    #
    first_frame_face_location = first_face_lm_frame_dict.get(str(chin_vertex_id)).copy()

    # ---- Calculate the offset between face position for the first frame and face position of the original model
    #
    offset = first_frame_face_location - local_face_original_location #local reference is used for shapekey

    # ---- Apply the offset to face each landmarks of each frame
    # 
    for face_lm_frame_dict in  face_landmarks.values():
        for keypoint in face_lm_frame_dict.keys():
            face_lm_frame_dict[keypoint] = face_lm_frame_dict[keypoint] - offset


def generate_face_shape_keys(face_mesh_obj, face_landmarks):
    '''
    Generate and animate shape keys for a face mesh based on face landmarks data.
    The function creates a basis shape key from the first frame's landmarks, then generates and animates specific shape keys for each frame of the provided landmarks.

    Parameters:
    -----------
    face_mesh_obj: bpy.types.Object
        The Blender face mesh object to which shape keys will be added.
    face_landmarks: dict
        Dictionary where each key is a frame index, and each value is a sub-dictionary mapping keypoint IDs to their vector coordinates.
    Returns:
    --------
    None
    '''
    bpy.ops.object.mode_set(mode='OBJECT')
    face_frame_idx = list(map(int, face_landmarks.keys()))
  
    # ---- Create the basis shape key ---- #

    face_mesh_obj.shape_key_add(name="Basis", from_mix=False)
    basis_shape_key = face_mesh_obj.data.shape_keys.key_blocks['Basis']

    # ---- Assign the values of first frame landmarks to the basis shape key
    #
    basis_shape_key.data.foreach_set("co", [lm for landmarks in face_landmarks.get(str(min(face_frame_idx))).values() for lm in landmarks])

    # ---- Generate a specific shape key for each frame ---- #

    # ---- Loop aver each frame
    #
    for frame_ind, face_lm_frame_dict in face_landmarks.items():

        # ---- Set timeline to the current frame
        #
        bpy.context.scene.frame_set(int(frame_ind))

        # ---- Add a shape key corresponding to the face position of the current frame
        #
        frame_shape_key = face_mesh_obj.shape_key_add(name=f'Frame_{frame_ind}_Shape', from_mix=False)
        frame_shape_key.data.foreach_set("co", [lm for landmarks in face_lm_frame_dict.values() for lm in landmarks])

        # ---- Set to 1 the value of the shape key and generate the keyframe for the current frame
        #
        face_mesh_obj.data.shape_keys.key_blocks[f'Frame_{frame_ind}_Shape'].value = 1
        face_mesh_obj.data.shape_keys.key_blocks[f'Frame_{frame_ind}_Shape'].keyframe_insert(data_path="value", frame=int(frame_ind))

        # ---- Set to 0 the value of the shape key and generate the keyframe for the previous and following frame (to deal with interpolation issue)
        #
        face_mesh_obj.data.shape_keys.key_blocks[f'Frame_{frame_ind}_Shape'].value = 0

        if int(frame_ind) < max(face_frame_idx):
            face_mesh_obj.data.shape_keys.key_blocks[f'Frame_{frame_ind}_Shape'].keyframe_insert(data_path="value", frame=int(frame_ind)+1)

        if int(frame_ind) > min(face_frame_idx):
            face_mesh_obj.data.shape_keys.key_blocks[f'Frame_{frame_ind}_Shape'].keyframe_insert(data_path="value", frame=int(frame_ind)-1)

    # ---- Go back to the first frame and set to 0 the value of all shape keys except the basis shape and generate the keyframes
    #
    bpy.context.scene.frame_set(1)

    for key_block in face_mesh_obj.data.shape_keys.key_blocks:
        if key_block.name != "Basis":
            key_block.value = 0
            key_block.keyframe_insert(data_path="value", frame=1)


def update_face_accessories_dict(face_mesh_obj, face_accessories_dict=FACE_ACCESSORIES_DICT):
    '''
    Compute and update the offset of each face accessory relative to its target vertex in the given face mesh.

    Parameters:
    -----------
    face_mesh_obj: bpy.types.Object
        The Blender face mesh object used as reference.
    face_accessories_dict: dict
        Dictionary mapping each face accessory to its target vertex group in the face mesh.

    Returns:
    --------
    dict
        Updated face_accessories_dict with 'original_offset' and 'target_vertex_ind' for each accessory.
    '''
    updated_face_accessories_dict = face_accessories_dict.copy()
    # ---- Loop over each parent object name and associated target vertex group
    #
    for object_name in updated_face_accessories_dict.keys():
        target_vertex_group_name = updated_face_accessories_dict[object_name].get('target_vertex_group')
    
        parent_obj = None
        # ---- Select the object
        #
        for obj in bpy.context.view_layer.objects:
            if (obj and obj.type == 'MESH'): 
                if object_name in obj.name:
                    parent_obj = obj
                    break
        if parent_obj:
            # ---- Determine the original location of the current parent objet
            #
            world_parent_original_location = parent_obj.location.copy()

            # ---- Retrieve the associated orginal location of the associated target vertex
            #
            target_vertex_ind = get_vertex_index(face_mesh_obj, target_vertex_group_name)
            world_target_vertex_group_original_location = get_vertex_coords(face_mesh_obj, target_vertex_ind)

            # ---- Calculate the offset
            #
            original_offset = world_parent_original_location - world_target_vertex_group_original_location

            # ---- Add a new section to the dictionnary corresponding to the original offset
            #
            updated_face_accessories_dict[object_name]['original_offset'] = original_offset

            # ---- Add a new section to the dictionnary corresponding to the index of the vertex of the target vertex group
            #
            updated_face_accessories_dict[object_name]['target_vertex_ind'] = target_vertex_ind

        else:
            print(f'{parent_obj} not found.')
    return updated_face_accessories_dict


def generate_face_accessories_keyframes(face_mesh_obj, update_face_accessories_dict):
    '''
    Animate face accessories by applying their offset relative to the target vertex for each frame.

    Parameters:
    -----------
    face_mesh_obj: bpy.types.Object
        The Blender face mesh object used as reference for calculating accessory positions.
    update_face_accessories_dict: dict
        Dictionary containing face accessories information, including target vertex indices and offsets.

    Returns:
    --------
    None
    '''
    # ---- Each face accessories follow face move by keeping the original offset of the original model

    # ---- Loop over each frame
    #
    for frame_ind in face_landmarks.keys():

        # ---- Set the scene to the frame ind
        #
        bpy.context.scene.frame_set(int(frame_ind))
    
        # --- Loop over each object
        #
        for object_name in update_face_accessories_dict.keys():

            # ---- Select the current object
            #
            accessory_obj = None

            for obj in bpy.context.view_layer.objects:
                if obj and obj.type == 'MESH': 
                    if object_name in obj.name:
                        accessory_obj = obj
            if accessory_obj:

                # ---- Get the target vertex position for the current frame
                #
                target_vertex_ind = update_face_accessories_dict[object_name]['target_vertex_ind']
                world_target_vertex_coord = get_vertex_coords(face_mesh_obj, vertex_ind=target_vertex_ind)

                # ---- Specify the position to the current parent object for the current frame
                #
                original_offset = update_face_accessories_dict[object_name]['original_offset']
                accessory_obj.location = world_target_vertex_coord + original_offset
                accessory_obj.keyframe_insert(data_path="location", frame=int(frame_ind))

            else:
                print(f'{accessory_obj} not found.')


# ---- LOAD DATA ---- #
# ------------------- #

#  ---- Create the scene
#
new_scene_name = "ImportedScene"
new_scene = bpy.data.scenes.new(new_scene_name)
bpy.context.window.scene = new_scene

# ---- Load and import all object of the original model files .blend
#
with bpy.data.libraries.load(filepath=MODEL_FILEPATH) as (data_from, data_to):
    data_to.objects = data_from.objects

for obj in data_to.objects:
    if obj is not None:
        new_scene.collection.objects.link(obj)

bpy.context.view_layer.update()

# ---- Select the face object
#
for obj in bpy.context.view_layer.objects:
    if obj and obj.type == 'MESH' and 'face' in obj.name:
        face_mesh_obj = obj

if not face_mesh_obj:
    print('face object not found.')

#---- Retrieve the initial offset of both eyes and hair relative to the face. We called this object auxilaries object
# ---- Later, we will ensure that the eyes and hair maintain this offset relative to the face throughout the animation
#
updated_auxiliaries_obj_dict = update_face_accessories_dict(face_mesh_obj, face_accessories_dict=FACE_ACCESSORIES_DICT)



# ---- ADAPT LANDMARKS TO ORIGINAL MODEL ---- #
# ------------------------------------------- #

# ---- Select the armature in the scene
#
armature = None

for obj in bpy.context.view_layer.objects:
    if obj and obj.type == 'ARMATURE':
        armature = obj
        break

if armature:
    bpy.context.view_layer.objects.active = armature
    print(f'Armature name selected: {armature.name}')
else:
    print('No armature found')

# ---- Adapt armature landmarks to the original model armature size and position and specified chains in CHAIN DICT
#
lm_dict = get_mapped_landmarks(armature, filepath=LANDMARKS_FILEPATH, chain_dict=CHAIN_DICT)

# ---- Adapt face landmarks to original model position and size
#
face_landmarks = lm_dict['face']
adapt_face_landmarks_to_model(face_mesh_obj, face_landmarks)



# ---- GENERATE THE ANIMATION ---- #
# -------------------------------- #


# ---- Apply contraints to bones ---- #

# ---- Apply strech_to contraint to the bones of the armature to force the tail of a previous bone to follow the head of a following bone in a chain.
# ---- In addition the last bone of the arm channel will follow the corresponding hand (left or right).
#
apply_stretch_to_contraints(armature, chain_dict=CHAIN_DICT)


# ---- Generate the keyframes for the armature ---- #

# ---- Loop over each subdictionnary of each subsection
#
for chain_sub_dict in CHAIN_DICT.values():

    # --- Loop over each chain
    #
    for chain_name in chain_sub_dict.keys():

        # ---- Retrieve the first frame index of the chain
        #
        frame_idx = list(map(int, lm_dict.get(chain_name).keys()))
        min_frame_ind = min(frame_idx)
        max_frame_ind = max(frame_idx)

        # ---- Set the chain to invisible from frame 1 to the given frame_ind
        #
        choosen_frame_ind = max_frame_ind
        set_to_invisible(chain_sub_dict, armature, chain_name, choosen_frame_ind)

        # ---- Loop over each frame
        #
        for frame_ind in lm_dict.get(chain_name).keys():

            generate_armature_keyframe(
                frame_ind,
                lm_dict=lm_dict,
                chain_dict=chain_sub_dict,
                chain_name=chain_name
                )

# ---- Generate shape keys to anime the the face ---- #

generate_face_shape_keys(face_mesh_obj, face_landmarks)

# ---- Generate the keyframes for the face accessories object ---- #

generate_face_accessories_keyframes(face_mesh_obj, updated_auxiliaries_obj_dict)
        


# ---- CONFIGURATE THE TIMELINE TO SEE THE ANIMATION ---- #
# ------------------------------------------------------- #


max_frame_ind = max(
    max(map(int, lm_dict[chain_name].keys()))
    for chain_name in lm_dict
)

new_scene.frame_start = 1
new_scene.frame_end = max_frame_ind 
new_scene.frame_current = 1
