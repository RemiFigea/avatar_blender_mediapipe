import cv2
import json


# ---- CONSTANTS ---- #
# ------------------- #

# ---- Define the Mediapipe keypoints IDs an connectionns for pose landmarks
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

# ---- Define the Mediapipe keypoints IDs an connectionns for hand landmarks
#
HAND_KEYPOINTS_CHAINS = [
    [[0, 1], [1, 2], [2, 3], [3, 4]],
    [[0,5], [5, 6], [6, 7], [7, 8]],
    [[9, 10], [10, 11], [11, 12]],
    [[13, 14], [14, 15], [15, 16]],
    [[17, 18], [18, 19], [19, 20]],
]


# ---- FUNCTIONS ---- #
# ------------------- #

def draw_face(frame, face_landmarks):
    '''
    Draw a blue circle on the given frame for each given landmarks coordinates.
    
    Paramaters:
    -----------
    frame: numpy array
        The frame on which we want to draw.
    face_landmarks: dictionnary
        The dictionnary with the face landmarks.
    
    Returns:
    -------'
    frame: numpy array
        The frame with the drawing added.
    '''
    
    for landmark in face_landmarks.values():
        
        # ---- Draw only if landmark are presents
        #
        if landmark:
            cv2.circle(frame, landmark[:-1], 1, (255, 0, 0), -1)

    return frame

def draw_pose_connections(frame, pose_landmarks, pose_keypoints_chains):
    '''
    Draw white lines between relevant points of pose landmarks.
    
    Parameters
    ----------
    frame: numpy array
        The frame on which we want to draw.
    pose_landmarks: dictionnary
        The dictionnary of pose landmarks.
    pose_keypoints_chains: list of list of integer
        The list of each pair of id of keypoint the link with a line.
    
    Return
    ------
    frame: numpy array
        The frame with the drawing added.
    '''
    keypoints_connections = []
    
    # ---- Determmine the connection to draw in each chain
    #
    for chain in pose_keypoints_chains:
        for connection in chain:
            keypoints_connections.append(connection)

    # ---- Iterate over each connection to draw
    #
    for connection in keypoints_connections:
        # ---- Get the coordinate of the keypoints to connect
        #
        landmark_start = pose_landmarks.get(str(connection[0]))
        landmark_end = pose_landmarks.get(str(connection[1]))

        # ---- Draw only if landmark are presents
        #
        if landmark_start and landmark_end:
            cv2.line(frame, landmark_start[:-1], landmark_end[:-1], (255, 255, 255), 2)

    return frame

def draw_hands_connections(frame, hand_landmarks, hand_keypoints_chains):
    '''
    Draw white lines between relevant points of hands landmarks
    
    Parameters
    ----------
    frame: numpy array
        Array of pixel on which we want to draw.
    hand_landmarks: dict
        The dictionnary of hands landmarks.
    hand_keypoints_chains: list of list of int
        The list of each pair of id of keypoint the link with a line.

    Return
    ------
    frame: numpy array, with the newly drawing of the hands
    '''
    keypoints_connections = []

    # ---- Determmine the connection to draw in each chain
    #
    for chain in hand_keypoints_chains:
        for connection in chain:
            keypoints_connections.append(connection)


    # ---- Iterate over each connection to draw
    #
    for connection in keypoints_connections:
        # ---- Get the coordinate of the keypoints to connect
        #
        landmark_start = hand_landmarks.get(str(connection[0]))
        landmark_end = hand_landmarks.get(str(connection[1]))

        # ---- Draw only if landmark are presents
        #
        if landmark_start and landmark_end:
            cv2.line(frame, landmark_start[:-1], landmark_end[:-1], (255, 255, 255), 2)
    
    return frame

def display_landmarks(video_path, filepath, scaling_rate=1, delay=1):
    '''
    Display a video with landmarks overlaid on each frame.

    Parameters:
    -----------
    video_path : str
        Path to the video file.
    filepath : str
        Path to the JSON file containing landmarks data.
    scaling_rate : int, optional, default=1
        Factor by which to scale the video for display.
    delay : int, optional, default=1
        Higher values slow down the change between two frames.

    Returns:
    --------
    None
    '''
    # ---- Copy original landmarks dictionnary
    #
    with open(filepath) as f:
        landmarks = json.load(f)
    
    pose_landmarks_frames = landmarks.get('pose')
    hand_L_landmarks_frames = landmarks.get('hand_L')
    hand_R_landmarks_frames = landmarks.get('hand_R')
    face_landmarks = landmarks.get('face')
    
    # ---- Create the window to display
    #
    window_name = 'Check landmarks result'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ---- define location for window and resize
    #
    cv2.moveWindow(window_name, 100, 100)  # location (100, 100) on the screen

    # ---- Open the video file
    #
    cap = cv2.VideoCapture(video_path)

    # ---- Check if video was opened successfully
    #
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # ---- Get the video properties
    #
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_max_ind = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # ---- Draw landmarks for each frame
    #
    for frame_ind in range(1, frame_max_ind + 1):

        # ---- Read the current frame from the video
        #
        ret, frame = cap.read()

        if not ret:
            break  # End of video
    
        # ---- Draw connections between keypoints
        #
        draw_pose_connections(frame, pose_landmarks_frames.get(str(frame_ind)), pose_keypoints_chains=POSE_KEYPOINTS_CHAINS)
        draw_hands_connections(frame, hand_L_landmarks_frames.get(str(frame_ind)), hand_keypoints_chains=HAND_KEYPOINTS_CHAINS)
        draw_hands_connections(frame, hand_R_landmarks_frames.get(str(frame_ind)), hand_keypoints_chains=HAND_KEYPOINTS_CHAINS)
        draw_face(frame, face_landmarks.get(str(frame_ind)))

        # ---- Resize frame and window
        #
        cv2.resizeWindow(window_name, scaling_rate * original_width, scaling_rate * original_height)
        frame_resized = cv2.resize(frame, (scaling_rate * original_width, scaling_rate * original_height))

        # ---- Display the frame
        cv2.imshow(window_name, frame_resized)

        # ---- Add delay before skipping to next frame
        if cv2.waitKey(delay*int(1000 / frame_rate)) & 0xFF == ord('q'):
            break

    # ---- Close the window after exiting the loop
    cap.release()
    cv2.destroyAllWindows()

