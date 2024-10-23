import os
from werkzeug.utils import secure_filename

def get_video_filename(request):
    '''
    Checks if a video file is uploaded in the request.

    Parameters:
    -----------
    request: The Flask request object.

    Returns:
    --------
    bool
        True if a video file is present, otherwise False.
    '''
    video_filename = None

    if 'video_file' in request.files:
        file = request.files['video_file']
        video_filename  = file.filename
        
    return video_filename 


def is_allowed_extension(filename, allowed_extensions):
    '''
    Check the extension of the given file.

    Parameters:
    -----------
    file: file-like object
        The file to check.
    allowed_extensions: set of str
        The set of allowed file extensions.

    Returns:
    --------
    bool
        True if the file extension is allowed, otherwise False.
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def is_valid_video_file(request, allowed_extensions, socketio):
    is_valid = True
    video_filename = get_video_filename(request)
    
    if video_filename=='' or  video_filename==None:
        socketio.emit('message', 'No video file in the request')
        is_valid = False

    elif not is_allowed_extension(video_filename, allowed_extensions):
        socketio.emit('message', 'Extension must be in {allowed_extensions}')
        is_valid = False
    
    return is_valid


def save_uploaded_file(file, input_dirpath):
    '''
    Save the uploaded file to the specified directory.

    Parameters:
    -----------
    file: file-like object
        The file to save.
    input_dirpath: str
        The directory path where the file will be saved.

    Returns:
    --------
    str
        The full path to the saved file.
    '''
    filename = secure_filename(file.filename)
    os.makedirs(input_dirpath, exist_ok=True)
    filepath = os.path.join(input_dirpath, filename)
    file.save(filepath)

    return filepath