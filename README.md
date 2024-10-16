# avatar_blender_mediapipe

Welcome to the **avatar_blender_mediapipe** repository!

The aim of this repository is to generate a Blender animation of a human character reproducing the movement of a real person in a video by using landmarks extracted with Mediapipe.

This repository is not finished yet, and there is still a lot to do and improve!

For now, you can take a glance at the results by running the Flask application with specific landmark files after building the docker image on your computer.

This project generates results using data files prepared in the /data folder. The scripts used to prepare these files are in the /scripts folder. They are not necessary to run the application.


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
            - test_landmarks_files.json
            /textures/
               - face_skin.png
               - mona2_Packed0_Diffuse.png
               - mona2_Packed0_Gloss.png
               - mona2_Packed0_Specular.png
         static/
         templates/
            - animation.html
         - app.py
         - blender_script.py
      - Dockerfile
      - requirements.txt
   /scripts/
      - create_original_face_mesh.py
      - display_landmarks.py
      - generate_landmarks.py
      - requirements_scripts.txt
    - README.md
```

- **`/docker_image`**: Directory containing the Docker image of the Flask application.
- **`/data/`**: Directory containing the data loaded in the scripts run by the application.
- **`/test_landmarks_files.json`**: JSON files of Mediapipe landmarks I have extracted from a video to test the script.
- **`/original_model.blend`**: Blender files containing the prepared 3D model character to animate.
- **`/textures/`**: Directory containing PNG images used to create texture of the 3D model character.
- **`/static/`**: Folder used to store the MP4 video file generated by the application.
- **`/templates/`**: Directory containing the html templates.
- **`app.py`**: Main script of the Flask application
- **`blender_script.py`**: Script executed by the application to run Blender and generate the animation.
- **`/scripts/`**: Directory containing utility scripts used to prepare the data files.
- **`create_original_face_mesh.py`**: Script to create the face of the original 3D model character that will be use for the animation.
- Note: The character model used is based on the predefined "Mona" model from BlenderKit. I suppress the orginal armature to create an armature adapted to Mediapipe keypoints. I also removed the original face mesh and create one adapted to Mediapipe keypoints.
- **`display_landmarks.py`**:  Script to display the original video with the landmarks drawn on it, to check the landmarks JSON file.
- **`generate_landmarks.py`**:  Script to extract and save the landmarks from a video into a JSON file.
- **`requirements_scripts.txt`** Dependencies required for the utility scripts.


## Getting Started

To run the Flask application, you only need the contents of the docker_image folder and Docker installed on your computer. Follow the steps below to get started:


1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username//avatar_blender_mediapipe.git
   
2. **Navigate to the docker_image folder:**
   ```bash
   cd /avatar_blender_mediapipe/docker_image

3. **Run the application using Docker:**
   ```bash
   docker build -t avatar_blender_mediapipe .
   docker run -p 5000:5000 avatar_blender_mediapipe

4. **Access the Application:**
You need to type exactly 'localhost:5000' and not 170.0001.5000
   ```
   - Open your web browser and navigate to http://localhost:5000.
   Note: Make sure to type the URL exactly as shown: localhost:5000. Do not use variations like 170.0001.5000, as they will not work.

   - On the HTML page, type the filename 'test_landmarks_files.json' and click on submit.

   - After a few minutes, the generated animation video will be displayed.
   ```
   *Note: The animation will be created using the default landmarks from the test_landmarks_files.json file. The functionality to generate landmarks directly within the application will be added soon.*
   
## Contributing

Feel free to contribute to the projects by opening issues or submitting pull requests.

If you have suggestions or improvements, I welcome your feedback!

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

