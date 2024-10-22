from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO
import os
import time
from scripts.pipeline_process import run_pipeline_process
from scripts.utils import is_video_file, is_allowed_file, save_uploaded_file
import threading

PATHS_DICT = {
    'blender_executable': './opt/blender/blender',
    'blender_script': './scripts/blender.py',
    'generated_video': './static/generated_video.mp4',
    'input_dir': './input',
    'landmarks': './static/landmarks.json',
    'original_model': './data/original_model.blend',
}

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app = Flask(__name__)
#app.config['SECRET_KEY'] = os.urandom(24)
socketio =  SocketIO(app, 
                    cors_allowed_origins='*', 
                    message_queue=os.environ.get('REDIS_URL'), 
                    channel='my_channel')

app.config['INPUT_DIRPATH'] = PATHS_DICT.get('input_dir')
app.config['SERVER_NAME'] = 'localhost:5000'
app.config['APPLICATION_ROOT'] = '/'
app.config['PREFERRED_URL_SCHEME'] = 'http'

generated_video_name = os.path.basename(PATHS_DICT.get('generated_video'))

main_process = None
def background_thread():
    '''
    Function to emit messages in the background.
    '''
    while True:
        time.sleep(10)  # Wait for 10 seconds
        socketio.emit('message', f'It is: {time.time()}')  # Emit current time to clients

@app.route('/')
def index():
    '''
    Render the video template with the video generated by Blender.
    '''

    return render_template('animation.html', video_name=generated_video_name)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    '''
    Handles video file upload, validates the file, starts the processing pipeline and redirects to index page.
    '''
    global main_process


    if not is_video_file(request):
        socketio.emit('message', 'No video file in the request')
        return redirect(url_for('index'))
    
    file = request.files['video_file']

    if not is_allowed_file(file, ALLOWED_EXTENSIONS):
        socketio.emit(f'message', 'Extension must be in {ALLOWED_EXTENSIONS}')
        return redirect(url_for('index'))

    uploaded_video_path = save_uploaded_file(file, PATHS_DICT.get('input_dir'))

    if main_process and main_process.is_alive():
        main_process.terminate()
        main_process.join()
        socketio.emit('message', 'Previous process terminated')

    socketio.emit('message', 'Starting a new process')

    thread = threading.Thread(
        target=run_pipeline_process,
        args=(uploaded_video_path, PATHS_DICT, socketio))
    thread.start()

    return redirect(url_for('index'))


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)