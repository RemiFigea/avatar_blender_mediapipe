import os
from werkzeug.utils import secure_filename

def is_video_file(request):
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

    if 'video_file' in request.files:
        file = request.files['video_file']

        if file.filename != '':
            return True
    
    return False


def is_allowed_file(file, allowed_extensions):
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
    filename = file.filename

    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

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