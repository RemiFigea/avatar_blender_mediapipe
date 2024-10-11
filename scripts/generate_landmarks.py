import cv2
import json
import mediapipe as mp

# ---- CONSTANTS ---- #
# ------------------- #

# ---- Define the Mediapipe keypoints IDs an dcoonectionns for pose landmarks
#
POSE_KEYPOINTS_CHAINS = [
    [[11, 12]],
    [[12, 14], [14, 16]],
    [[11, 13], [13, 15]],
    [[16 , 18]],
    [[16, 20]],
    [[16 , 22]],
    [[15, 17]],
    [[15 , 19]],
    [[15, 21]],
    ]

# ---- Build the set of keypoints selected for pose landmarks
#
pose_keypoints_set = set()

for chain in POSE_KEYPOINTS_CHAINS:
    for connection in chain:
        for keypoint in connection:
            pose_keypoints_set.add(keypoint)

# ---- Define the Mediapipe keypoints IDs an dcoonectionns for hand landmarks
#
HAND_KEYPOINTS_CONNECTIONS = [
    [[0, 1], [1, 2], [2, 3], [3, 4]],
    [[0,5], [5, 6], [6, 7], [7, 8]],
    [[9, 10], [10, 11], [11, 12]],
    [[13, 14], [14, 15], [15, 16]],
    [[17, 18], [18, 19], [19, 20]],
]

# ---- Build the set of keypoints selected for hand landmarks
#
hand_keypoints_set = set()

for chain in HAND_KEYPOINTS_CONNECTIONS:
    for connection in chain:
        for keypoint in connection:
            hand_keypoints_set.add(keypoint)

# ---- Specify the path to the video to process
#
VIDEO_PATH = '/home/remifigea/code/avatar/raw_videos/00632.mp4'

# ---- FUNCIONS ---- #
# ------------------ #

# ---- Initialize Mediapipe object
#
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh


def get_face_landmarks(frame, width, height):
    '''
    Generate facial landmarks for the given frame using the MediaPipe Face Mesh model.

    Paramaters
    ---------
    frame: numpy array
        The image image to make detection on.
    width: float
        The width of the given frame.
    height: float
        The height of the given frame.

    Returns
    -------
    lm_dict: dictionnary
        Keys are the Ids of the keypoints and values is a tuple of 3 floats corresponding to the landmarks.

     Notes
    -----
    If multiple faces are detected in the frame, only the landmarks for the first face are returned.
    '''
    # ---- Calculate landmarks for the frame
    #
    with mp_face_mesh.FaceMesh(static_image_mode=False) as face:
        results = face.process(frame)

    # ---- Initialize an empty dictionnary to collect landmarks
    #
    lm_dict = {}

    # ---- Check if landmarks are detected in the frame
    #
    if results.multi_face_landmarks:
        # ---- Iterate over each keypoint in the face and calculate its coordinates
        #
        for face_ind , face_landmarks in enumerate(results.multi_face_landmarks):
            # ---- Initialize an empty dictionnary to collect landmarks of the current detected face
            #
            face_lm_dict = {}

            # ---- Iterate over each keypoint and associated landmarks coordinates
            #
            for keypoint, lm in enumerate(face_landmarks.landmark):
                x = int(lm.x * width)
                y = int(lm.y * height)
                z = lm.z

                # ---- Store the (x, y, z) tuple in the current face's landmark dictionary
                #
                face_lm_dict[keypoint] = (x, y, z)
            

            # ---- Add the current face's landmark dictionary to the overall dictionary
            #
            lm_dict[face_ind] = face_lm_dict

    # ---- Return the landmarks for the first detected face, if any
    # If no face was detected, an empty dictionary is returned
    #
    return lm_dict.get(0, {})


def get_hands_landmarks(frame, width, height, keypoints_set=hand_keypoints_set, world_landmarks=False):
    '''
    Generate pose landmarks of the frame associated to selected keypoints.
    Paramaters
    ---------
    frame: numpy array
        The image image to make detection on.
    width: float
        The width of the given frame.
    height: float
        The height of the given frame.
    keypoint_set: set of int
        The set of keypoints to collect the associated landmarks to.
    world_landmarks: bool
        If true the funcion return wordl_landmarks else landmarks. Default, True.
    Returns
    -------
    hands_lm_dict: dictionnary
        Dictionnary with 2 subdictionnaries associated to left hand and right hand.
        Keys are the Ids of the keypoints and values is a tuple of 3 floats corresponding to the landmarks.

    '''
    # ---- Calculate landmarks for the frame
    #
    with mp_hands.Hands(model_complexity=1)  as hands:
        results = hands.process(frame)
    
    # ---- Select the appropriate landmark required
    #
    if world_landmarks == True:
        lm_results = results.multi_hand_world_landmarks
    else:
        lm_results = results.multi_hand_landmarks

    # ---- Initialize an empty dictionnary to collect th landmarks
    #
    lm_dict = {'hand_L': {}, 'hand_R': {}}


    # ---- Skip if landmarks are detected in the frame
    #
    if lm_results is None:
    
        return lm_dict

    # ---- Iterate over the 2 elements of lm_results (one is for left hand, the other is for right hand)
    #
    for i, landmarks in enumerate(lm_results):

        # ---- Get handedness label
        #
        handedness_label = 'hand_L'

        if results.multi_handedness[i].classification[0].index == 1:
            handedness_label = 'hand_R'

        # ---- Iterate over each keypoint
        #
        for keypoint, lm in enumerate(landmarks.landmark):
                        
            # ---- Populate the left_hand_lm dictionnary if keypoints are in keypoints_set
            #
            if keypoint in keypoints_set:
                            
                # ---- Adapt format of the landmark weither it is a world landmark or not
                #
                if world_landmarks == True:
                    lm_dict.get(handedness_label)[keypoint] = (lm.x, lm.y, lm.z)
                else:
                    lm_dict.get(handedness_label)[keypoint] = (int(width * lm.x), int(height *lm.y), lm.z)


    return lm_dict


def get_pose_landmarks(frame, width, height, keypoints_set=pose_keypoints_set, world_landmarks=False):
    '''
    Generate pose landmarks of the frame associated to selected keypoints.
    Paramaters
    ---------
    frame: numpy array
        The image image to make detection on.
    width: float
        The width of the given frame.
    height: float
        The height of the given frame.
    keypoint_set: set of int
        The set of keypoints to collect the associated landmarks to.
    world_landmarks: bool
        If true the funcion return wordl_landmarks else landmarks. Default, True.
    Returns
    -------
    lm_dict: dictionnary
        Keys are the Ids of the keypoints and values is a tuple of 3 floats corresponding to the landmarks.
    '''
    # ---- Calculate landmarks for the frame
    #
    with mp_pose.Pose(model_complexity=2)  as pose:
        results = pose.process(frame)
    
    # ---- Initialize an empty dictionnary to collect landmarks
    #
    lm_dict = {}

    # ---- Check if landmarks are detected in the frame
    #
    if results.pose_landmarks:

        if world_landmarks == True:
            # ---- Get world landmarks
            #
            landmarks = results.pose_world_landmarks.landmark
        else:
            # ---- Get landmarks
            #    
            landmarks = results.pose_landmarks.landmark

        for keypoint, lm in enumerate(landmarks):

            # ---- Check if keypoint is in the set of keypoints to keep
            #
            if keypoint in keypoints_set:

                if world_landmarks == True:
                    lm_dict[keypoint] = (lm.x , lm.y, lm.z)
                else:
                    lm_dict[keypoint] = (int(lm.x*width) , int(lm.y*height), lm.z)

    return lm_dict


def get_video_pose_landmarks(
        video_path,
        pose_keypoints_set=pose_keypoints_set,
        hand_keypoints_set=hand_keypoints_set,
        start_frame=1,
        end_frame=-1,
        world_landmarks=False
        ):
    '''
    Generate the pose, hand, and face landmarks of the selected keypoints for each frame of a video.

    Parameters
    ----------
    video_path: str
        Path to the video file.
    pose_keypoints_set: set of int
        IDs of the keypoints to extract for the pose landmarks.
    hand_keypoints_set: set of int
        IDs of the keypoints to extract for the hand landmarks.
    start_frame: int, optional (default=1)
        Frame index number to start reading the video from (1-based index).
    end_frame: int, optional (default=-1)
        Frame index number to stop reading the video. Default is -1, which means until the end of the video.
    world_landmarks: bool, optional (default=True)
        Whether to return 3D world landmarks or 2D pixel landmarks.

    Returns
    -------
    video_landmarks: dict
        A dictionary containing the pose, hand, and face landmarks for each frame in the video.
        Keys: 'pose', 'hand_R', 'hand_L', 'face', 'width', 'height'
        Each landmark entry is a dictionary where the frame index is the key, and the corresponding landmarks are the values.
    '''
    
    # ---- Open video file
    #
    cap = cv2.VideoCapture(video_path)

    # ---- Get original size of the frame
    #
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ---- Handle start and end frame indices
    # ---- Ensure starting frame is at least 1
    #
    if start_frame <= 1:
        start_frame = 1

    # ---- If no end frame is specified, set it to the total frame count
    #
    if end_frame < 0: 
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- Initialize the current frame index
    #
    frame_index = 1
    
    # ---- Initialize dictionaries to store landmarks for each frame
    #
    pose_lm_dict = {}
    hand_L_lm_dict = {}
    hand_R_lm_dict = {}
    face_lm_dict = {}

    # ---- Iterate over the frames and collect landmarks
    #
    while cap.isOpened() and frame_index <= end_frame:
        # ---- Read the current frame
        #
        ret, frame = cap.read()

        # ---- If frame could not be read, exit the loop
        #
        if not ret:
            break
        
        # ---- Process frames only after reaching the start frame
        #
        if frame_index >= start_frame:

            # ---- Prepare the frame for landmark extraction
            # ---- Make frame non-writable and convert to RGB
            #
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---- Extract pose landmarks for the current frame
            #
            pose_lm_dict[frame_index] = get_pose_landmarks(
                frame,
                width,
                height,
                keypoints_set=pose_keypoints_set,
                world_landmarks=world_landmarks
                )
            
            # ---- Extract hand landmarks (left and right) for the current frame
            #
            temp_dict= get_hands_landmarks(
                frame, 
                width,
                height,
                keypoints_set=hand_keypoints_set,
                world_landmarks=world_landmarks
                )
                                             
            hand_L_lm_dict[frame_index] = temp_dict.get('hand_L')
            hand_R_lm_dict[frame_index] = temp_dict.get('hand_R')

            # ---- Extract face landmarks for the current frame
            #
            face_lm_dict[frame_index] = get_face_landmarks(
                frame,
                width,
                height
                )
            
        # ---- Move to the next frame
        #
        frame_index += 1
    
    # ---- Release the video capture object
    #
    cap.release()

    # ---- Combine all landmark dictionaries into one result dictionary
    #
    video_landmarks = {
        'pose': pose_lm_dict,
        'hand_R': hand_R_lm_dict,
        'hand_L': hand_L_lm_dict,
        'face': face_lm_dict
        }
    
    # ---- Add video dimensions (width and height) to the result dictionary
    #
    video_landmarks['width'] = width
    video_landmarks['height'] = height

    return video_landmarks

def generate_landmarks_file(
        video_path,
        filepath,
        pose_keypoints_set=pose_keypoints_set,
        hand_keypoints_set=hand_keypoints_set,
        start_frame=1,
        end_frame=-1,
        world_landmarks=False
        ):
    '''
    Extract face, pose and hand landmarks from the given video and store them in JSON file at the specified filepath
    
    Parameters:
    -----------
    VIDEO_PATH: str
        Path to the video.
    FILEPATH: str
        Path to the file to store the resuts.
    pose_keypoints_set: set of int
        IDs of the keypoints to extract for the pose landmarks.
    hand_keypoints_set: set of int
        IDs of the keypoints to extract for the hand landmarks.
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
    video_landmarks = get_video_pose_landmarks(
        video_path=video_path,
        pose_keypoints_set=pose_keypoints_set,
        hand_keypoints_set=hand_keypoints_set,
        start_frame=start_frame,
        end_frame=end_frame,
        world_landmarks=world_landmarks
        )
    
    with open(filepath, 'w') as f:
        json.dump(video_landmarks, f, indent=4)


    