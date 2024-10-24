import cv2
import json
import mediapipe as mp
import os
from scripts.generate_video import CHAIN_DICT

class VideoProcess:
    def __init__(self, video_path, chain_dict, start_frame, end_frame):
        '''
        Parameters:
        -----------
        video_path: str
            Path to the video.
        chain_dict: dict
            Dictionary containing sub-dictionaries. Each sub-dictionary maps body chain names to lists of keypoint ID pairs.
        start_frame: int, optional (default=1)
            Frame index number to start reading the video from (1-based index).
        end_frame: int, optional (default=-1)
            Frame index number to stop reading the video. Default is -1, which means until the end of the video.
        '''
        self.video_path = video_path
        self.chain_dict = chain_dict
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose_keypoints_set = None
        self.hand_keypoints_set = None

        # Open video capture to extract properties
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure starting frame is at least 1
        if start_frame <= 1:
            self.start_frame = 1

        # Adjust end frame if not specified
        if end_frame < 0:
            self.end_frame = self.total_frames
        
    def release_cap(self):
        '''Release the video capture object'''
        self.cap.release()
    
    def get_pose_keypoints_set(self):
        '''
        Build the set of keypoints selected for pose landmarks.

        Returns:
        --------
        pose_keypoints_set: set of int
            Set of keypoint IDs from the pose chains.
        '''
        chain_dict = self.chain_dict
        pose_keypoints_set = set()

        pose_keypoints_chains = chain_dict.get('pose').values()
        for chain in pose_keypoints_chains:
            for connection in chain:
                for keypoint in connection:
                    pose_keypoints_set.add(keypoint)

        self.pose_keypoints_set = pose_keypoints_set

        return pose_keypoints_set

    def get_hand_keypoints_set(self):
        '''
        Extracts the set of keypoints used for hand landmarks (left hand assumed identical to right).

        Returns:
        --------
        hand_keypoints_set: set of int
            Set of keypoint IDs from the left hand chains.
        '''
        chain_dict = self.chain_dict
        hand_keypoints_set = set()

        hand_keypoints_chains = chain_dict.get('hand_L').values()
        for chain in hand_keypoints_chains:
            for connection in chain:
                for keypoint in connection:
                    hand_keypoints_set.add(keypoint)

        self.hand_keypoints_set = hand_keypoints_set

        return hand_keypoints_set
    
def get_frame_face_landmarks(frame, video_process):
    '''
    Generate facial landmarks for the given frame using the MediaPipe Face Mesh model.

    Parameters:
    -----------
    frame: numpy array
        The image on which to detect landmarks.
    video_process: VideoProcess
        The video process object containing video details.

    Returns
    -------
    lm_dict: dictionnary
        Keys are the Ids of the keypoints and values is a tuple of 3 floats corresponding to the landmarks.

    Notes
    -----
    If multiple faces are detected in the frame, only the landmarks for the first face are returned.
    '''
    width = video_process.width
    height = video_process.height
    mp_face_mesh = video_process.mp_face_mesh

    with mp_face_mesh.FaceMesh(static_image_mode=False) as face:
        results = face.process(frame)

    lm_dict = {}

    # Check if landmarks are detected in the frame
    if results.multi_face_landmarks:
        # Iterate over each keypoint in the face and calculate its coordinates
        for face_ind , face_landmarks in enumerate(results.multi_face_landmarks):
  
            face_lm_dict = {}

            # Iterate over each keypoint and associated landmarks coordinates
            for keypoint, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * width)
                y = int(lm.y * height)
                z = lm.z
            
                face_lm_dict[keypoint] = (x, y, z)
            
            lm_dict[face_ind] = face_lm_dict

    # Return the landmarks for the first detected face, if any
    # If no face was detected, an empty dictionary is returned
    return lm_dict.get(0, {})

def get_frame_hands_landmarks(frame, video_process, world_landmarks=False):
    '''
    Generate pose landmarks of the frame associated to selected keypoints.

    Paramaters
    ---------
    frame: numpy array
        The image on which to detect landmarks.
    video_process: VideoProcess
        The video process object containing video details.
    world_landmarks: bool
        If true the funcion return wordl_landmarks else landmarks. Default, True.
    Returns
    -------
    hands_lm_dict: dictionnary
        Dictionnary with 2 subdictionnaries associated to left hand and right hand.
        Keys are the Ids of the keypoints and values is a tuple of 3 floats corresponding to the landmarks.

    '''
    width = video_process.width
    height = video_process.height
    keypoints_set = video_process.get_hand_keypoints_set()
    mp_hands = video_process.mp_hands

    with mp_hands.Hands(model_complexity=1)  as hands:
        results = hands.process(frame)
    
    if world_landmarks == True:
        lm_results = results.multi_hand_world_landmarks
    else:
        lm_results = results.multi_hand_landmarks

    lm_dict = {'hand_L': {}, 'hand_R': {}}


    if lm_results is None:
    
        return lm_dict

    # Iterate over the 2 elements of lm_results (one is for left hand, the other is for right hand)
    for i, landmarks in enumerate(lm_results):

        handedness_label = 'hand_L'

        if results.multi_handedness[i].classification[0].index == 1:
            handedness_label = 'hand_R'

        # Iterate over each keypoint
        for keypoint, lm in enumerate(landmarks.landmark):
                        
            # Populate the left_hand_lm dictionnary if keypoints are in keypoints_set
            if keypoint in keypoints_set:
                            
                # Adapt format of the landmark weither it is a world landmark or not
                if world_landmarks == True:
                    lm_dict.get(handedness_label)[keypoint] = (lm.x, lm.y, lm.z)
                else:
                    lm_dict.get(handedness_label)[keypoint] = (int(width * lm.x), int(height *lm.y), lm.z)

    return lm_dict

def get_frame_pose_landmarks(frame, video_process, world_landmarks=False):
    '''
    Generate pose landmarks of the frame associated to selected keypoints.

    Paramaters
    ----------
    frame: numpy array
        The image on which to detect landmarks.
    video_process: VideoProcess
        The video process object containing video details.
    world_landmarks: bool
        If true the funcion return wordl_landmarks else landmarks. Default, True.
    Returns
    -------
    lm_dict: dictionnary
        Keys are the Ids of the keypoints and values is a tuple of 3 floats corresponding to the landmarks.
    '''
    width = video_process.width
    height = video_process.height
    keypoints_set = video_process.get_pose_keypoints_set()
    mp_pose = video_process.mp_pose

    with mp_pose.Pose(model_complexity=2)  as pose:
        results = pose.process(frame)
    lm_dict = {}

    if results.pose_landmarks:
        if world_landmarks == True:
            landmarks = results.pose_world_landmarks.landmark
        else: 
            landmarks = results.pose_landmarks.landmark

        for keypoint, lm in enumerate(landmarks):

            # Check if keypoint is in the set of keypoints to keep
            if keypoint in keypoints_set:

                if world_landmarks == True:
                    lm_dict[keypoint] = (lm.x , lm.y, lm.z)
                else:
                    lm_dict[keypoint] = (int(lm.x*width) , int(lm.y*height), lm.z)

    return lm_dict

def get_video_landmarks(video_process, world_landmarks=False):
    '''
    Generate the pose, hand, and face landmarks of the selected keypoints for each frame of a video.

    Parameters
    ----------
    video_process:  VideoProcess
        VideoProcess object containing video properties.
    world_landmarks: bool, optional (default=True)
        Whether to return 3D world landmarks or 2D pixel landmarks.

    Returns
    -------
    video_landmarks: dict
        A dictionary containing the pose, hand, and face landmarks for each frame in the video.
        Keys: 'pose', 'hand_R', 'hand_L', 'face', 'width', 'height'
        Each landmark entry is a dictionary where the frame index is the key, and the corresponding landmarks are the values.
    '''
    start_frame = video_process.start_frame
    end_frame = video_process.end_frame
    cap = video_process.cap
    width = video_process.width
    height = video_process.height
        
    frame_index = 1
    pose_lm_dict = {}
    hand_L_lm_dict = {}
    hand_R_lm_dict = {}
    face_lm_dict = {}

    # Iterate over the frames and collect landmarks
    while cap.isOpened(): #and frame_index <= end_frame: #problem with total_frames_count when video is .WEBM
        # Read the current frame
        ret, frame = cap.read()

        # If frame could not be read, exit the loop
        if not ret:
            break
        
        # Process frames only after reaching the start frame
        if frame_index >= start_frame:

            # Prepare the frame for landmark extraction
            # Make frame non-writable and convert to RGB
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_lm_dict[frame_index] = get_frame_pose_landmarks(
                frame,
                video_process,
                world_landmarks=world_landmarks
                )
            
            temp_dict= get_frame_hands_landmarks(
                frame, 
                video_process,
                world_landmarks=world_landmarks
                )
                                            
            hand_L_lm_dict[frame_index] = temp_dict.get('hand_L')
            hand_R_lm_dict[frame_index] = temp_dict.get('hand_R')

            face_lm_dict[frame_index] = get_frame_face_landmarks(
                frame,
                video_process
                )
        frame_index += 1

    # Release the video capture object
    video_process.release_cap()


    video_landmarks = {
        'pose': pose_lm_dict,
        'hand_R': hand_R_lm_dict,
        'hand_L': hand_L_lm_dict,
        'face': face_lm_dict
        }
    
    video_landmarks['width'] = width
    video_landmarks['height'] = height

    return video_landmarks

def generate_landmarks_file(
        input_video_path,
        landmarks_filepath,
        queue,
        chain_dict=CHAIN_DICT,
        start_frame=1,
        end_frame=-1,
        world_landmarks=False
        ):
    '''
    Extract face, pose and hand landmarks from the given video and store them in JSON file at the specified filepath
    
    Parameters:
    -----------
    video_path: str
        Path to the video.
    landmarks_filepath: str
        Path to the file to store the resuts.
    queue: multiprocessing.Queue
        Queue used to communicate between processes. Once the landmarks extraction is complete, the string "landmarks_done" 
        is placed into the queue.
    chain_dict: dict
            Dictionary containing sub-dictionaries. Each sub-dictionary maps body chain names to lists of keypoint ID pairs.
    start_frame: int, optional (default=1)
        Frame index number to start reading the video from (1-based index).
    end_frame: int, optional (default=-1)
        Frame index number to stop reading the video. Default is -1, which means until the end of the video.
    world_landmarks: bool, optional (default=True)
        Whether to return 3D world landmarks or 2D pixel landmarks.
    
    Returns:
    --------
    None
    '''
    video_process = VideoProcess(input_video_path, chain_dict, start_frame, end_frame)
    video_landmarks = get_video_landmarks(video_process, world_landmarks=world_landmarks)
    
    os.makedirs(os.path.dirname(landmarks_filepath), exist_ok=True)

    with open(landmarks_filepath, 'w') as f:
        json.dump(video_landmarks, f, indent=4)
    
    queue.put('Landmarks generation terminated successfully.')





    
    