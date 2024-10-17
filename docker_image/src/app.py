from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO
import os
import subprocess

app = Flask(__name__)
#app.config['SECRET_KEY'] = os.urandom(24)
#socketio = SocketIO(app)
socketio = SocketIO(app)

OUTPUT_DIRPATH = './static'
BLENDER_EXECUTABLE_PATH = '../opt/blender/blender'
BLENDER_SCRIPT_PATH = './blender_script.py'
VIDEO_NAME = 'generated_video.mp4'

@socketio.on('message')
def handle_message(message):
    print('Recieved message', message)
    socketio.send(message)

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@app.route('/')
def index():
    '''
    Render the video template with the specified video filename.
    
    '''
    return render_template('animation.html', video_name=VIDEO_NAME)


@app.route('/upload', methods=['POST'])
def upload():
    '''
    Retrieves the landmarks filename from the form. If provided, 
    runs the Blender script to generate the animation. Redirects 
    to the index page afterward.
    '''

    landmarks_filename = request.form['landmarks_filename']
    
    socketio.emit('status_update', 'Preparing Blender environment')

    if landmarks_filename:

        command = [
            BLENDER_EXECUTABLE_PATH,            # Path to Blender executable
            '-b',                               # Background mode
            '--python', BLENDER_SCRIPT_PATH,    # Path to Blender script
            '--',                               # Specify arguments for the script
            landmarks_filename                  # User-provided landmarks filename
            ]

        # ---- Execute the Blender command
        subprocess.run(command)

    socketio.emit('status_update', 'Video loaded')

    if os.path.exists(os.path.join(OUTPUT_DIRPATH, VIDEO_NAME)):
        print(f"{VIDEO_NAME} created successfully.")
    else:
        print(f"Error : {VIDEO_NAME} not found.")

    return redirect(url_for('index'))

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
