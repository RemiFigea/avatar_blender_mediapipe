# avatar_blender_mediapipe

Welcome to the **avatar_blender_mediapipe** repository!

The aim of this repository is to generate a Blender animation of a human character reproducing the movement of a real person in a video by using landmarks extracted with Mediapipe.

This repository is not finished yet, and there is still a lot to do and improve!

Currently, you can upload a video featuring only one person. The video should be short, as the processing may take a considerable amount of time. The application will then generate a video with a character mimicking your gestures.


## Main challenges

Challenges are various. Here are some relevant ones:

- One challenge is to find a solution to adapt landmark coordinates generated by Mediapipe to the various coordinate systems in a scene for a character modeled in Blender.
- Another challenge is to make use of Blender's object properties to generate a 3D model character.

## Repository Structure

The repository is structured as follows:
```
/avatar_blender_mediapipe/
   docker_image/
      src/
         data/
            - original_model.blend
            /textures/
               - face_skin.png
               - mona2_Packed0_Diffuse.png
               - mona2_Packed0_Gloss.png
               - mona2_Packed0_Specular.png
         scripts/
            - generate_landmarks.py
            - generate_video.py
            - process_managing.py
            - utils.py
         static/
         templates/
            - animation.html
         - app.py
      - Dockerfile
      - requirements.txt
   /miscellaneous_scripts/
      - create_original_face_mesh.py
      - display_landmarks.py
      - requirements_scripts.txt
   - README.md
```

- **`/docker_image`**: Contains the Docker image for the Flask application.
- **`/data/`**: Holds the data used in application scripts.
- **`/original_model.blend`**: Blender file with the prepared 3D character model for animation.
- **`/textures/`**: Contains PNG images for the 3D character's textures.
- **`/scripts/`**:  Directory with scripts executed by the application.
- **`generate_landmarks.py`**:  Extracts and saves landmarks from a video into a JSON file.
- **`generate_video.py`**:  Runs Blender to generate the animation.
- **`process_managing.py`**: Functions for managing processes within the application.
- **`utils.py`**:  Utility functions for handling files, filenames, and file paths.
- **`/static/`**:  Stores the generated MP4 video file.
- **`/templates/`**: Contains HTML templates for the application.
- **`app.py`**: Main script of the Flask application
- **`/miscelleneanous_scripts/`**: Utility scripts for data preparation and checks; not used in the application.
- **`create_original_face_mesh.py`**: Script to create the face of the original 3D model character that will be use for the animation.
- Note: The character model used is based on the predefined "Mona" model from BlenderKit. I suppress the orginal armature to create an armature adapted to Mediapipe keypoints. I also removed the original face mesh and create one adapted to Mediapipe keypoints.
- **`display_landmarks.py`**:   Displays the original video with drawn landmarks for JSON verification.
- **`requirements_miscelleneous_scripts.txt`** Dependencies required for the miscelleneous scripts.


## Getting Started

To run the Flask application, you only need the contents of the docker_image folder and Docker installed on your computer. Follow the steps below to get started:


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RemiFigea/avatar_blender_mediapipe.git
   
2. **Navigate to the docker_image folder:**
   ```bash
   cd /avatar_blender_mediapipe/docker_image

3. **Run the application using Docker:**
   ```bash
   docker build -t avatar_blender_mediapipe .
   docker run -p 5000:5000 avatar_blender_mediapipe

4. **Access the Application:**
   ```
   - Open your web browser and navigate to http://localhost:5000.

   - On the HTML page, load your short video file.

   - After a few minutes, the generated animation video will be displayed.
   ```
   
## Contributing

Feel free to contribute to the projects by opening issues or submitting pull requests.

If you have suggestions or improvements, I welcome your feedback!

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

