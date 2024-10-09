# avatar_blender_mediapipe

Welcome to the **avatar_blender_mediapipe** repository!

The aim of this repository is to generate a Blender animation of a human character reproducing the movement of a real person in a video by using landmarks extracted with Mediapipe.

This repository is not finished yet, and there is still a lot to do and improve!

For now, you can take a glance at the results with specific landmark files if you have Blender installed on your computer.

## Main challenges

Challenges are various. Here are some relevant ones:

- One challenge is to find a solution to adapt landmark coordinates generated by Mediapipe to the various coordinate systems in a scene for a character modeled in Blender.
- Another challenge is to make use of Blender's object properties to generate a 3D model character.

## Repository Structure

The repository is structured as follows:
```
/avatar_blender_mediapipe
    /blender_script.py
    /data/
        - test_landmarks_files.json
        - original_model.blend
        /textures/
            - face_skin.png
            - mona2_Packed0_Diffuse.png
            - mona2_Packed0_Gloss.png
            - mona2_Packed0_Specular.png
    - README.md
```

- **`/blender_script.py`**: The script to run Blender to generate the animation.
- **`/data/`**: Directory containing the data loaded in the script `blender_script.py`.
- **`/test_landmarks_files.json`**: JSON files of Mediapipe landmarks I have extracted from a video to test the script.
- **`/original_model.blend`**: Blender files containing the prepared 3D model character to animate.
- **`/textures/`**: Directory containing PNG images used to create texture of the 3D model character.

## Getting Started


You need to have Blender installed on your computer.

I worked on a windows computer with Blender 4.2. 

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username//avatar_blender_mediapipe.git
   
2. **Navigate to the Project Folder:**
   ```bash
   cd /avatar_blender_mediapipe

3. **Set Up the Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
  
4. **Generate the animation:**
   ```
   - Open Blender on your computer.
   - Load the script 'blender_script.py' in the Text Editor.
   - Run the script.
   ```
   *The animation will be generated with the default landmarks of the files 'test_landmarks_files.json'.*

   ```
   - Open the animation window.
   ```

   *You can now run the animation using the Timeline.*
   
## Contributing

Feel free to contribute to the projects by opening issues or submitting pull requests.

If you have suggestions or improvements, I welcome your feedback!

## License

This repository is licensed under the MIT License. See the LICENSE file for more details.

