import bpy
import json
import mathutils
import os
import socketio
import sys


# ---- CONSTANTS ---- #
# ------------------- #

MODEL_FILEPATH = '/data/original_model.blend'
VIDEO_OUTPUT_PATH = '/static/generated_video.mp4'

# Defines keypoint chains for building bones, using Mediapipe keypoint IDs
# Each chain is a list of keypoint pairs [start_id, end_id], representing bone connections
CHAIN_DICT = {
    'pose' : {
        'arm_L' : [[12, 14], [14, 16]],
        'arm_R' : [[11, 13], [13, 15]],
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

# Dictionary of face auxiliaries object with associated target vertex groups ('Eye_R_Target', 'Eye_L_Target', 'Hair_Target') 
# for calculating their offsets from the face.
FACE_AUXILIARIES_OBJ_DICT = {
    'eye_R': {'target_vertex_group_on_face': 'Eye_R_Target'},
    'eye_L': {'target_vertex_group_on_face': 'Eye_L_Target'},
    'hair': {'target_vertex_group_on_face':'Hair_Target'},
}


# ---- FUNCIONS ---- #
# ------------------ #

def convert_landmarks_to_blender(filepath):
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

    with open(filepath, 'r') as json_file:
        landmarks_dict = json.load(json_file)


    # Scaling factor for y-coordinate adjustments, tuned after trials
    y_scale_factor = {
        'pose': 0.2,
        'hand_L': 2,
        'hand_R': 2,
        'face': 1,
        }

    width = landmarks_dict.pop('width')
    del landmarks_dict['height']

    # ---- Adjust each landmark's coordinates
    for body_subdivision, body_subdivision_dict in landmarks_dict.items():
        for frame_dict in body_subdivision_dict.values():
            if frame_dict:
                for key, landmark in frame_dict.items():
                    xc = landmark[0]
                    yc = landmark[2] * y_scale_factor.get(body_subdivision) * width
                    zc = -landmark[1]
                    vector = mathutils.Vector((xc, yc, zc))

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
    for hand, pose_keypoint_ref in [('hand_L', '16'), ('hand_R', '15')]:

        for frame_ind, frame_lm_dict in landmarks[hand].items():

            # ---- Set hand landmark ID 0 to the associated pose reference
            if landmarks['pose'][frame_ind].get(pose_keypoint_ref) is not None and landmarks[hand][frame_ind].get('0') is not None:
                translation_vector = landmarks['pose'][frame_ind].get(pose_keypoint_ref) - landmarks[hand][frame_ind].get('0')
                landmarks[hand][frame_ind]['0'] = landmarks['pose'][frame_ind].get(pose_keypoint_ref)
            
                # ---- Adjust remaining hand landmarks
                for keypoint in frame_lm_dict.keys():
                    if keypoint != '0' and landmarks[hand][frame_ind].get(keypoint) is not None:
                        landmarks[hand][frame_ind][keypoint] = landmarks[hand][frame_ind].get(keypoint) + translation_vector
            else:
                landmarks[hand][frame_ind] = {}
            
    return landmarks

def get_armature_scale_factor_and_translation_vector(original_model_armature, pose_landmarks, reference_bone_name):
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
    reference_bone_name: str
        The name of the reference bone to match between the two armatures.

    Returns:
    --------
    scale_rate: float
        The factor to scale the generated bone.
    translation_vector: mathutils.Vector
        The vector to translate the generated bone to align with the original.
    '''
    # ---- Get the length of the reference bone in the original model armature
    original_model_armature_bone = original_model_armature.data.bones.get(reference_bone_name)
    
    if original_model_armature:
        original_model_bone_length = original_model_armature_bone.length
    
    # ---- Retrieve the length of the reference bone based on landmarks in the first frame that have values.
    frame_idx_sorted = sorted(list(map(int, pose_landmarks.keys())))

    for frame_ind in frame_idx_sorted:

        if pose_landmarks.get(str(frame_ind)) != {}:

            head = pose_landmarks[str(frame_ind)].get('12')
            tail = pose_landmarks[str(frame_ind)].get('14')
            landmarks_bone_length = (head-tail).length
            break
    

    scale_rate = original_model_bone_length/landmarks_bone_length

    scaled_bone_head = scale_rate * head
    armature_bone_head = original_model_armature_bone.head

    translation_vector = armature_bone_head - scaled_bone_head + mathutils.Vector((0, - 0.02, 0)) # we had a small offset after checking the animation

    return scale_rate, translation_vector

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

def get_list_of_pairs_of_keypoints(chain_dict, chain_name):
    '''
    Extracts the list of keypoint ID pairs for the specified body chain from the provided chain dictionary.

    Parameters:
    -----------
    chain_dict: dict
        Dictionary containing sub-dictionaries. Each sub-dictionary maps body chain names to lists of keypoint ID pairs.
    chain_name: str
        The name of the body chain to retrieve the associated list of keypoint ID pairs.

    Returns:
    --------
    list of list of int
        List of pairs of keypoint ID pairs (articulations) for the specified chain name.
    '''
    # Select the sub_dictionary associated with the chain
    chain_sub_dict = None
    for sub_dict in chain_dict.values():
        if chain_name in sub_dict:
            chain_sub_dict = sub_dict
            break

    # Ensure we found the correct sub_dictionary
    if chain_sub_dict is None:
        raise ValueError(f"Chain name {chain_name} not found in any sub_dictionary.")
    
    pairs_of_keypoints_list = chain_sub_dict.get(chain_name)

    return pairs_of_keypoints_list

def generate_a_specific_armature_keyframe(armature, frame_ind, lm_dict, chain_sub_dict, chain_name):
    '''
    Generates a keyframe for Blender animation for a specified body chain.

    Parameters:
    -----------
    armature: bpy.types.Armature
        The armature to which keyframe will be added.
    frame_ind: int
        Index of the frame to generate.
    lm_dict: dict
        Mapping of chain names to their landmarks.
    chain_sub_dict: dict
        Mapping of body chain names to pairs of keypoint IDs for articulations.
    chain_name: str
        Name of the body chain to animate.
    '''
    bpy.context.scene.frame_set(int(frame_ind))

    lm_chain_dict = lm_dict[chain_name]

    # ---- Loop through each bone in the chain to set its position
    for i in range(len(chain_sub_dict[chain_name])) :

        bone_name = f'{chain_name}_{i}'
        pose_bone = armature.pose.bones.get(bone_name)
       
        # ---- Retrieve landmarks for the head of the bone to determine its position
        bone_keypoints = chain_sub_dict.get(chain_name)[i]
        expected_head_position = lm_chain_dict.get(frame_ind).get(str(bone_keypoints[0]))      
        
        pose_bone.matrix.translation = expected_head_position
        pose_bone.keyframe_insert(data_path="location", frame=int(frame_ind))

         
    # ---- Translate the extra bone of the chain (excluding 'arm_L' and 'arm_R') to manage the tail position
    if (chain_name != 'arm_L') and (chain_name != 'arm_R'):

        extra_bone_name = f'{chain_name}_extra'
        pose_extra_bone = armature.pose.bones.get(extra_bone_name)

         # ---- Retrieve landmarks for the tail of the bone at the chain's extremity
        chain_extremity_bone_keypoint = bone_keypoints
        tail_of_chain_extremity_bone = lm_chain_dict.get(frame_ind).get(str(chain_extremity_bone_keypoint[1]))

        expected_head_extra_bone = tail_of_chain_extremity_bone
        pose_extra_bone.matrix.translation = expected_head_extra_bone
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
    vertex_group = mesh_obj.vertex_groups.get(vertex_group_name)
    
    if vertex_group:
        # Loop over each vertex of the mesh object
        for vertex in mesh_obj.data.vertices:
            # Loop over each group the vertex belongs to
            for group in vertex.groups:
                # elect vertex index if the group is in vertex_group
                if group.group == vertex_group.index:
                    vertex_ind = vertex.index
                    break
    else:
        print("Target vertex group does not exist.")
    
    return vertex_ind

def get_vertex_world_coords(obj, vertex_ind):
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
    # Get the Dependency Graph to evaluate the object with applied modifiers
    dependancy_graph = bpy.context.evaluated_depsgraph_get()

    evaluated_object = obj.evaluated_get(dependancy_graph)
    vertex = evaluated_object.data.vertices[vertex_ind]
    co_world = evaluated_object.matrix_world @ vertex.co
    
    return co_world

def render_callback(scene, context):
    '''
    Emit the rendering progress of a scene as a percentage.

    This function calculates the current rendering progress based on the scene's 
    frame range and emits the progress to a connected client via a socket. 
    If no client is connected, it prints a notification to the console.

    Parameters
    ----------
    scene : bpy.types.Scene
        The Blender scene being rendered.
    context : bpy.types.Context
        The current Blender context.

    Returns
    -------
    None
    '''
    try:
        progress = int(100 * (scene.frame_current - scene.frame_start)  / (scene.frame_end - scene.frame_start))
        if sio.connected:
            sio.emit('message', f'Building the video: {progress}%')
        else:
            print("Client not connected, impossible to emit progression message.")
    except:
        print('Progress not available.')

class FaceAuxiliaries:
    '''
    Class to handle face auxiliaries object such as eyes and hair.

    Attributes
    ----------
    name : str
        The name of the face auxiliary.
    target_vertex_group_on_face : str
        The vertex group associated with this auxiliary on the face mesh.
    obj : bpy.types.Object or None
        The Blender object associated with the face auxiliary, if found.
    target_vertex_ind : int or None
        The index of the target vertex on the face mesh.
    original_offset_with_face : mathutils.Vector or None
        The offset of the auxiliary object from its target vertex on the face mesh.
    '''
    def __init__(self, name, target_vertex_group_on_face):
        '''
        Parameters:
        -----------
        name : str
            The name of the face auxiliary.
        target_vertex_group_on_face : str
            The vertex group associated with this auxiliary on the face mesh.
        '''
        self.name = name
        self.target_vertex_group_on_face = target_vertex_group_on_face
        self.obj = None
        self.target_vertex_ind = None
        self.original_offset_with_face = None
    
    def get_associated_obj_in_the_scene(self):
        '''
        Finds and returns the associated object in the Blender scene that matches the auxiliary's name.
        '''
        for obj in bpy.context.view_layer.objects:
            if (obj and obj.type == 'MESH' and self.name in obj.name): 
                self.obj = obj
                return self.obj
        return None

class  SceneManager:
    '''
    Class to manage all transformation applyed to object in blender scene object.
    
    Attributes
    ----------
    scene : bpy.types.Scene
        The Blender scene object being managed.
    chain_dict : dict
        A dictionary mapping body subdivision names to sub-dictionaries, 
        where each sub-dictionary maps a chain name to a list of pairs of Mediapipe keypoint IDs 
        [start_id, end_id], representing bone connections.
    face_auxiliaries_list : list
        A list of face auxiliary objects, each an instance of the FaceAuxiliaries class.
    '''

    def __init__(self, scene_name, chain_dict):
        '''
        Parameters:
        -----------
        scene_name: str
            Name of the Blender scene object to create.
        chain_dict : dict
            A dictionary mapping body subdivision names to sub-dictionaries, 
            where each sub-dictionary maps a chain name to a list of pairs of Mediapipe keypoint IDs 
            [start_id, end_id], representing bone connections.
        '''
        self.scene = bpy.data.scenes.new(scene_name)
        self.chain_dict = chain_dict
        self.face_auxiliaries_list = []
        bpy.context.window.scene = self.scene
    
    def load_original_3D_model(self, model_filepath):
        '''
        Load a 3D model from a specified file path into the current scene.

        Parameters:
        -----------
        model_filepath: str
            The file path to the 3D model to be loaded.

        Returns:
        --------
        None
        '''    
        scene = self.scene

        try:
            with bpy.data.libraries.load(model_filepath) as (data_from, data_to):
                data_to.objects = data_from.objects

            for obj in data_to.objects:
                if obj is not None:
                    scene.collection.objects.link(obj)
                    bpy.context.view_layer.objects.active = obj
                    obj.select_set(True)
                    # Ensure the original reference scaling rate is 1, usefull for futur resizing
                    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
                    obj.select_set(False)  # S'assurer que l'objet est actif

            bpy.context.view_layer.update()
        
        except Exception as e:
            print(f"Error loading model: {e}")

    def get_armature(self):
        '''
        Return the armature object of the scene
        '''
        armature = None

        for obj in bpy.context.view_layer.objects:
            if obj and obj.type == 'ARMATURE':
                armature = obj
                break

        if armature:
            bpy.context.view_layer.objects.active = armature
        else:
            print('No armature found')
        
        return armature
    
    def get_face_mesh(self):
        '''
        Return the face mesh of the scene
        The object is expected to be named 'face', but Blender might append an index to the name upon import.
        '''
        face_mesh_obj = None

        for obj in bpy.context.view_layer.objects:
            if obj and obj.type == 'MESH' and 'face' in obj.name:
                face_mesh_obj = obj

        if not face_mesh_obj:
            print('face object not found.')
        
        return face_mesh_obj   

    def initialize_face_auxiliaries_list(self, face_auxiliaries_obj_dict):
        '''
        Parameters:
        -----------
        face_auxiliaries_obj_dict: dict
            A dictionary where keys are names of face auxiliaries 
                                           and values are dictionaries containing properties 
                                           such as 'target_vertex_group_on_face'.
        Returns:
        self.face_auxiliaries_list: list
             A list of face auxiliary objects, each an instance of the FaceAuxiliaries class.
        '''

        for name, subdict in face_auxiliaries_obj_dict.items():
            target_vertex_group_on_face = subdict.get('target_vertex_group_on_face')
            face_auxiliary = FaceAuxiliaries(name, target_vertex_group_on_face)
     
            self.face_auxiliaries_list.append(face_auxiliary)
        
        return self.face_auxiliaries_list
    
    def update_face_auxiliaries_list(self):
        '''
        Extracts and updates data from the 3D original model configuration in the scene.

        This method updates the following attributes of the elements in the 
        'face_auxiliaries_list':
            - 'target_vertex_ind': The index of the target vertex on the face mesh.
            - 'world_target_vertex_group_original_location': The world coordinates of the associated target vertex.
            - 'original_offset_with_face': The offset between the original location of the face auxiliary and the target vertex.
        '''
        face_mesh_obj = self.get_face_mesh()

        if not face_mesh_obj:
            print('Face mesh object not found.')
            return

        for face_auxiliary in self.face_auxiliaries_list:

            face_auxiliary.target_vertex_ind = get_vertex_index(face_mesh_obj, face_auxiliary.target_vertex_group_on_face)


            face_auxiliary.get_associated_obj_in_the_scene()
            
            if face_auxiliary.obj:
                world_face_auxiliary_original_location = face_auxiliary.obj.location.copy()
                face_auxiliary.world_target_vertex_group_original_location = get_vertex_world_coords(face_mesh_obj, face_auxiliary.target_vertex_ind).copy()

                face_auxiliary.original_offset_with_face = world_face_auxiliary_original_location - face_auxiliary.world_target_vertex_group_original_location
            else:
                print(f'Face auxiliary object {face_auxiliary.name} not found.')

    def get_mapped_landmarks(self, landmarks_filepath):
        '''
        Maps body chain landmarks from a file with the armature of the scene.
        Note: Landmarks corresping to armature are adjusted to the original 3D model.

        Parameters:
        -----------
        landmarks_filepath: str
            Path to the JSON file containing landmarks extracted with Mediapipe.
        
        Returns:
        --------
        lm_dict: dict
            Dictionary mapping chain names to their respective landmarks.
        '''
        chain_dict = self.chain_dict

        armature = self.get_armature()
        
        lm_dict = {}

        converted_landmarks = convert_landmarks_to_blender(landmarks_filepath)

        adjusted_landmarks = adjust_hand_to_arm_z_position(converted_landmarks)

        # ---- Build a dictionary with sub-dictionaries for each chain
        #      Rescale and translate landmarks to match the original model's size and position
        pose_landmarks = adjusted_landmarks.get('pose')

        # Set the reference bone name for scaling and translation
        reference_bone_name = 'arm_L_0'

        scale_factor, translation_vector = get_armature_scale_factor_and_translation_vector(armature, pose_landmarks, reference_bone_name)
        
        # Loop over each body subdivision specified in the chain dictionnary
        for body_subdivision, chain_sub_dict in chain_dict.items():
            # Loop over each chain of the current body subdivision
            for chain_name, connections_list in chain_sub_dict.items():

                lm_dict[chain_name] = {}

                # ---- Determine the set of keypoints represented in the current chain
                keypoints_set = set()
                for connection in connections_list:
                    for keypoint in connection:
                        keypoints_set.add(str(keypoint))

                # Build a subdictionnary for the current chain and current frame with adjusted landmarked
                for frame, lm_frame_dict in adjusted_landmarks.get(body_subdivision).items():

                    if lm_frame_dict == {}:
                        continue

                    lm_dict[chain_name][frame] = {}

                    # Loop over each keypoint
                    for keypoint, landmarks in lm_frame_dict.items():

                        # Add keypoint and rescaled landmarks if they correspond to the current chain
                        if keypoint in keypoints_set:
                            lm_dict[chain_name][frame][keypoint] = landmarks * scale_factor + translation_vector

                    # Remove the frame if it contains no landmarks
                    if lm_dict[chain_name][frame] == {}:
                        del lm_dict[chain_name][frame]

        # ---- Add adjusted landmarks associated with the face
        #      We will use shape key to animated the face, therefore we don't need to rescale face landmarks
        face_landmarks = adjusted_landmarks.get('face')
        lm_dict['face'] = {}
        
        for frame, lm_frame_dict in face_landmarks.items():

            if lm_frame_dict == {}:
                continue

            lm_dict['face'][frame] = lm_frame_dict

        return lm_dict

    def get_face_scale_factor(self, face_landmarks):
        '''
        Calculate the scaling factor to apply to face landmarks to match distance between to eye in the landmarks with original 3D model
        
        Parameters:
        ----------
        face_landmarks: dict
            Dictionary of face landmarks with each keypoint as a vector.

        Return:
        -------
        scale_factor: float
            The scaling factor to be applied to the face landmarks. 
            Returns 1.0 if no landmarks are detected or if there is an error in detection.
        '''
        scale_factor = 1

        if not face_landmarks:
            print('No face landmarks detected. Face not resized')
            return scale_factor

        original_eye_location_dict = {}
        eye_location_dict = {}

        first_frame_ind = min(map(int, face_landmarks.keys()))

        for face_auxiliary in self.face_auxiliaries_list:
            if face_auxiliary.name in ['eye_R', 'eye_L']:
                original_eye_location_dict[face_auxiliary.name] = face_auxiliary.world_target_vertex_group_original_location.copy()
                eye_location_dict[face_auxiliary.name] = face_landmarks[str(first_frame_ind)].get(str(face_auxiliary.target_vertex_ind)).copy()

                if eye_location_dict[face_auxiliary.name] is None:
                    print('Vertex group coordinate empty. Face not rezised')
                    return scale_factor

        if len(original_eye_location_dict)!=2:
            print('Error in original model eye detection. Not exactly to eye are detected. Face not resied')
            return scale_factor
        
        original_distance_between_eye = (original_eye_location_dict.get('eye_R') - original_eye_location_dict.get('eye_L')).length
        first_frame_distance_between_eye = (eye_location_dict.get('eye_R') - eye_location_dict.get('eye_L')).length

        scale_factor = original_distance_between_eye / first_frame_distance_between_eye

        return scale_factor
    
    def adapt_face_landmarks_to_original_3D_model(self, face_landmarks):
        '''
        Adjust face landmarks to align the 'Chin_Target' vertex in both the landmarks and the face mesh object in the scene.

        Parameters:
        -----------
        face_landmarks: dict
            Dictionary of face landmarks with each keypoint as a vector.

        Returns:
        --------
        None
        '''
        face_mesh_obj = self.get_face_mesh()      
        scale_factor = self.get_face_scale_factor(face_landmarks)

        # Get the vertex index for the chin target from the face mesh
        chin_vertex_id = get_vertex_index(face_mesh_obj, 'Chin_Target')

        # The position of the chin is used to specify the position of the face
        world_face_original_location = get_vertex_world_coords(face_mesh_obj, chin_vertex_id)
        local_face_original_location = face_mesh_obj.matrix_world.inverted() @ world_face_original_location

        if not face_landmarks:
            print('No face landmarks detected. Face not adapated to original')
            return
        first_face_lm_frame_ind = min(map(int, face_landmarks.keys()))
        first_face_lm_frame_dict = face_landmarks.get(str(first_face_lm_frame_ind))

        # Get the face location for the first frame, based on the chin vertex ID
        first_frame_face_location = first_face_lm_frame_dict.get(str(chin_vertex_id)).copy()
        rescaled_first_frame_face_location = first_frame_face_location * scale_factor

        # Calculate the offset needed to align the face with the local reference position for shape keys
        offset = rescaled_first_frame_face_location - local_face_original_location

        # Apply the calculated offset and scaling factor to all keypoints in the face landmarks for every frame
        for face_lm_frame_dict in  face_landmarks.values():
            for keypoint in face_lm_frame_dict.keys():
                face_lm_frame_dict[keypoint] = face_lm_frame_dict.get(keypoint) * scale_factor - offset

    def apply_stretch_to_contraints_to_bones(self):
        '''
        Apply a "Stretch To" constraint to bones in a specified chain, forcing the tail of each bone to follow the head of the next bone in the chain.
        If the bone name is `arm_L_1` or `arm_R_1`, the constraint will force the bone to follow the corresponding hand (left or right).
        '''
        chain_dict = self.chain_dict
        
        armature = self.get_armature()

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

    def hidden_each_bone_of_the_chain(self, chain_name):
        '''
        Selected the armature in the scene.
        Selected the chain associated to the given chain_name.
        Hidden each armature bone of the chain from frame 1.

        Parameters:
        -----------
        chain_name: str
            The name of the bone chain to process.

        Returns:
        --------
        None
        '''
        chain_dict = self.chain_dict

        list_of_pairs_of_keypoints_of_the_chain = get_list_of_pairs_of_keypoints(chain_dict, chain_name)

        armature = self.get_armature()

        # Loop through the bones in the specified chain
        for i in range(len(list_of_pairs_of_keypoints_of_the_chain)) :

            bone_name = f'{chain_name}_{i}'
            pose_bone = armature.pose.bones.get(bone_name)
            
            # ---- Set bone to invisible from frame 1
            pose_bone.bone.hide = True
            pose_bone.bone.keyframe_insert(data_path="hide", frame=1)

        # Set to invisible extra bone of the given chain from frame 1
        # Note: Chains 'arm_L' and 'arm_R' do not contain extra bones
        if (chain_name != 'arm_L') and (chain_name != 'arm_R'):
            
            extra_bone_name = f'{chain_name}_extra'
            pose_extra_bone = armature.pose.bones.get(extra_bone_name)

            pose_extra_bone.bone.hide = True
            pose_extra_bone.bone.keyframe_insert(data_path="hide", frame=1)

    def generate_armature_keyframes(self, lm_dict):
        '''
        Generate keyframes for the armature based on landmark data.

        Parameters:
        -----------
        lm_dict : dict
            Dictionary mapping each chain to its landmark across frames.
        '''
        chain_dict = self.chain_dict

        armature = self.get_armature()
    
        # Loop over each subdictionnary of each body subdivision
        for chain_sub_dict in chain_dict.values():

            # Loop over each chain
            for chain_name in chain_sub_dict.keys():

                if lm_dict[chain_name]=={}:
                    print(f'{chain_name} is not detected on the video')
                    continue

                self.hidden_each_bone_of_the_chain(chain_name)

                # Generate keyframe for each frame of the current chain
                for frame_ind in lm_dict.get(chain_name).keys():

                    generate_a_specific_armature_keyframe(
                        armature,
                        frame_ind,
                        lm_dict=lm_dict,
                        chain_sub_dict=chain_sub_dict,
                        chain_name=chain_name
                        )

    def generate_face_shape_keys(self, face_landmarks):
        '''
        Generate and animate shape keys for a face mesh based on face landmarks data.
        The function creates a basis shape key from the first frame's landmarks, then generates and animates specific shape keys for each frame of the provided landmarks.

        Parameters:
        -----------
        face_landmarks: dict
            Dictionary where each key is a frame index, and each value is a sub-dictionary mapping keypoint IDs to their vector coordinates.
        Returns:
        --------
        None
        '''
        face_mesh_obj = self.get_face_mesh()

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

    def generate_face_accessories_keyframes(self, face_landmarks):
        '''
        Animate face accessories in the scene by applying their offset relative to the target vertex for each frame.

        Parameters:
        -----------
        face_landmarks: dict
            Dictionary where each key is a frame index, and each value is a sub-dictionary mapping keypoint IDs to their vector coordinates.

        Returns:
        --------
        None
        '''
        face_mesh_obj = self.get_face_mesh()

        # Each face accessories follow face move by keeping the original offset of the original model for each frame
        for frame_ind in face_landmarks.keys():

            bpy.context.scene.frame_set(int(frame_ind))
        
            # Loop over face auxiliary object
            for original_model_auxiliary in self.face_auxiliaries_list:

                # Select the current face auxiliary object in the scene
                face_auxiliary = None
                for obj in bpy.context.view_layer.objects:
                    if obj and obj.type == 'MESH': 
                        if original_model_auxiliary.name in obj.name:
                            face_auxiliary = obj

                if face_auxiliary:

                    # ---- Get the target vertex position of the current face auxiliary object for the current frame
                    world_target_vertex_coord = get_vertex_world_coords(face_mesh_obj, original_model_auxiliary.target_vertex_ind)

                    # ---- Specify the position to the face auxiliary object for the current frame
                    face_auxiliary.location = world_target_vertex_coord + original_model_auxiliary.original_offset_with_face
                    face_auxiliary.keyframe_insert(data_path="location", frame=int(frame_ind))

                else:
                    print(f'{ face_auxiliary} not found.')

    def create_world_for_rendering(self, bg_color=(0, 0, 0, 0), intensity=1):
        '''
        Create and configure a world for final rendering.
    
        Parameters:
        -----------
        bg_color: tuple
            A 4-tuple representing the RGBA color for the background.
        intensity: float
            The intensity of the background lighting.
        '''
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")

        # ---- Configurate the lighting of the environment
        bpy.context.scene.world.use_nodes = True
        bg_node = bpy.context.scene.world.node_tree.nodes.get("Background")

        # Set the colorbackground to black
        bg_node.inputs[0].default_value = bg_color

        # Set the intensity
        bg_node.inputs[1].default_value = intensity
    
    def add_lights(self):
        '''
        Add 4 points light
        '''
        for i in range(4):
            bpy.ops.object.light_add(type='POINT', location=(i * 3 - 4.5, -4, 2))
            light = bpy.context.object
            light.data.energy = 200
            light.data.use_shadow = True
    
    def add_camera(self):
        '''
        If not already exist, add a camera at a predefined location pointing at the 3D model and set it as the active camera.
        '''
        if bpy.context.scene.camera:
            print('A camera already exists in the scene. Setting the new camera as active.')

   
        bpy.ops.object.camera_add(location=(0, -5, 1.3), rotation=(1.57, 0, 0))

        camera = bpy.context.object  # Get the newly created camera
        bpy.context.scene.camera = camera  # Set the new camera as the active camera

    def generate_video(self, output_filepath, frame_end):
        '''
        Configure the scene for video rendering and generate a video file.

        Parameters:
        -----------
        output_filepath : str
            The file path where the generated video will be saved.
        frame_end : int
            The last frame of the animation to be rendered.
        '''
        scene = self.scene

        # ---- Configurate the scene for MP4 video generation
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = 'H264'
        scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
        scene.render.ffmpeg.ffmpeg_preset = 'GOOD'


        # ---- Set the resolution
        scene.render.resolution_x = 640
        scene.render.resolution_y = 360
        scene.render.resolution_percentage = 100

        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        scene.render.filepath = output_filepath

        # ---- Specifiy the frame range for the video generation
        scene.frame_start = 1
        scene.frame_end = frame_end
        
         # ---- Set the render quality by specifying the number of samples
        bpy.context.scene.eevee.taa_render_samples = 8
        bpy.context.scene.eevee.taa_samples = 8

        # ---- Set callbacks to be sent for each change of frame during rendering
        bpy.app.handlers.render_pre.append(render_callback)

        bpy.ops.render.render(animation=True, use_viewport=True, scene=scene.name)
        
        print(f'Rendering completed. Video file saved at: {output_filepath}')

def main(filepath):

    # Create the scene
    scene = SceneManager("New_scene", CHAIN_DICT)

    scene.load_original_3D_model(MODEL_FILEPATH)

    # Eyes and hair are considered as face auxiliaries object
    scene.initialize_face_auxiliaries_list(FACE_AUXILIARIES_OBJ_DICT)
    
    # Get the initial offset of face auxilaries object with the face in the original 3D model.
    # Later, we will ensure that the eyes and hair maintain this offset relative to the face throughout the animation
    scene.update_face_auxiliaries_list()

    # Resize an reorganize armature landmarks to match with original model armature size and position for each specified chain in CHAIN DICT
    lm_dict = scene.get_mapped_landmarks(filepath)

    face_landmarks = lm_dict['face']

    scene.adapt_face_landmarks_to_original_3D_model(face_landmarks)

    scene.apply_stretch_to_contraints_to_bones()

    scene.generate_armature_keyframes(lm_dict)

    scene.generate_face_shape_keys(face_landmarks)

    scene.generate_face_accessories_keyframes(face_landmarks)

    scene.create_world_for_rendering()

    scene.add_camera()

    scene.add_lights()
    
    # Get the upper index of the frame
    max_frame_ind = max(
        max(map(int, lm_dict[chain_name].keys()))
        for chain_name in lm_dict
        if lm_dict[chain_name]
        )

    scene.generate_video(VIDEO_OUTPUT_PATH, max_frame_ind)


if __name__ == "__main__":

    sio = socketio.Client()
    sio.connect('http://localhost:5000')

    landmarks_filepath = sys.argv[-1] # The landmarks_filepath is the last element of the list
    print(f'File {landmarks_filepath} has been added')

    main(landmarks_filepath)
    
    sio.disconnect()