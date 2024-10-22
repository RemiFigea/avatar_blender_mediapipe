from flask_socketio import SocketIO
from scripts.generate_landmarks import generate_landmarks_file
from multiprocessing import Process
import subprocess


main_process = None


def run_pipeline_process(
        uploaded_video_path,
        paths_dict,
        socketio: SocketIO,
    ):
    '''
    Starts the `pipeline` function, waits for its completion, and emits messages through SocketIO.

    Parameters:
    -----------
    uploaded_video_path: str
        Path to the video file.
    paths_dict: dict
        A dictionary mapping keys to file paths in the application's directory.
    socketio: SocketIO
        The socketio instance to emit messages.
    '''

    main_process = Process(
        target=pipeline,
        args=(uploaded_video_path, paths_dict, socketio)
        )

    main_process.start()
    main_process.join()

    if main_process and not main_process.is_alive():
        socketio.emit('redirect', '/')

def pipeline(
        uploaded_video_path,
        paths_dict,
        socketio: SocketIO,
    ):
    '''
    Generates landmarks and runs the Blender process.

    Parameters:
    -----------
    uploaded_video_path: str
        Path to the video file.
    paths_dict: dict
        A dictionary mapping keys to file paths in the application's directory.
    socketio: SocketIO
        The socketio instance to emit messages.
    '''
    socketio.emit('message', 'Generating landmarks...')

    try:
        generate_landmarks_file(uploaded_video_path, paths_dict.get('landmarks'))
    except Exception as e:
        socketio.emit('message', f"Generating landmarks failed: {str(e)}")
        return
    
    socketio.emit('message', 'Preparing Blender environment...')

    command = [
        paths_dict.get('blender_executable'),
        '-b',
        '--python', paths_dict.get('blender_script'),
        '--',
        paths_dict.get('landmarks'),
        paths_dict.get('original_model'),
        paths_dict.get('generated_video')
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
        socketio.emit('message', f"Blender finished successfully: {result.stdout}")
    except subprocess.CalledProcessError as e:
        socketio.emit('message', f"Generating video failed: {e.stderr}")
