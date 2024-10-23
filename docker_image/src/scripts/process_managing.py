def stop_landmarks_generation_if_running(landmark_process, socketio):
    if landmark_process and landmark_process.is_alive():
        landmark_process.terminate()
        landmark_process.join()
        socketio.emit('message', 'Current landmarks generation process stopped.')

def stop_video_generation_if_running(video_process, socketio):
    if video_process and not video_process.poll():
        video_process.terminate()
        video_process.wait()
        socketio.emit('message', 'Current video generation process stopped.')