# Digital Crime Scene Investigation Using 3D Gaussian Splatting

## Project Overview
This project focuses on utilizing **3D Gaussian Splatting** for digital crime scene investigation, enabling the creation of **photorealistic and interactive 3D models** of crime scenes that can be explored in real-time using **Unity**. This approach enhances forensic analysis by preserving scene integrity and allowing detailed examination in virtual environments.

## Features
- **3D Scene Reconstruction** using **3D Gaussian Splatting**.
- **High-fidelity crime scene visualization** for forensic analysis.
- **Optimized rendering** for real-time interaction.
- **Integration with Unity (.fbx, .obj)** for immersive exploration.
- **Preprocessing pipeline** for dataset preparation.

## System Architecture
1. **Dataset Preparation**: Image acquisition and preprocessing.
2. **3D Gaussian Splatting Implementation**: Scene reconstruction with optimization techniques.
3. **Integration with Unity**: Visualization and real-time exploration.
4. **Performance Evaluation**: Metrics analysis for quality and efficiency.

## Technologies Used
- **Python** (Neural rendering & Gaussian Splatting optimization)
- **COLMAP** (Feature matching and 3D point cloud generation)
- **Unity** (Scene rendering and interactive visualization)
- **PyTorch** (Deep learning framework for Gaussian Splatting)
- **OpenCV** (Image preprocessing)

## Installation & Setup
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- OpenCV
- COLMAP
- Unity 2022.3+

### Clone the Repository
```sh
git clone https://github.com/yourusername/3D-CrimeScene-Investigation.git
cd 3D-CrimeScene-Investigation
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Step-by-Step Execution

### 1. Dataset Preparation
1. Capture high-resolution images ensuring **60%+ overlap** between frames.
2. Preprocess images using the `convert.py` script:
   ```sh
   python preprocess.py --input-folder ./dataset --output-folder ./processed_dataset
   ```
3. Ensure the images are consistent in lighting and resolution to improve COLMAP processing accuracy.

### 2. 3D Reconstruction using COLMAP
1. Run COLMAP to generate the **sparse point cloud**:
   ```sh
   colmap feature_extractor --database_path ./colmap.db --image_path ./processed_dataset
   colmap exhaustive_matcher --database_path ./colmap.db
   colmap mapper --database_path ./colmap.db --image_path ./processed_dataset --output_path ./colmap_output
   ```
2. Convert the **sparse** reconstruction to a **dense** point cloud:
   ```sh
   colmap image_undistorter --image_path ./processed_dataset --input_path ./colmap_output --output_path ./colmap_dense
   ```
3. Generate the final **meshed model**:
   ```sh
   colmap poisson_mesher --input_path ./colmap_dense --output_path ./colmap_mesh.ply
   ```

### 3. Training 3D Gaussian Splatting Model
1. Train the model using the preprocessed dataset:
   ```sh
   python train.py --dataset ./colmap_dense --output ./model_output
   ```
2. Apply **regularization techniques** to minimize noise in the point cloud:
   ```sh
   python refine.py --input ./model_output --output ./refined_model
   ```

### 4. Unity Integration
#### 4.1 Importing the Point Cloud
1. **Install Unity and Set Up a New Project**:
   - Install Unity Hub and Unity 2022.3+.
   - Create a new **3D project** (e.g., "CrimeSceneReconstruction").

2. **Install Gaussian Splatting Plugin**:
   - Clone the [UnityGaussianSplatting](https://github.com/aras-p/UnityGaussianSplatting) repository.
   - Open `projects/GaussianExample` as a Unity project.
   - Ensure the project uses DX12 or Vulkan (DX11 is not supported).

3. **Create GaussianSplat Assets**:
   - Navigate to `Tools -> Gaussian Splats -> Create GaussianSplatAsset`.
   - Specify the path to your `.ply` file from `point_cloud/iteration_*/`.
   - Choose compression options.
   - Click **Create Asset**.

4. **Add the Gaussian Splat Renderer**:
   - Create an empty GameObject.
   - Attach the `GaussianSplatRenderer` script.
   - Assign the created GaussianSplat asset.
   - Adjust transformation settings (rotation ~-160° X-axis).

5. **Adjust Rendering Settings**:
   - Modify density, opacity, and color settings for photorealism.

#### 4.2 Scene Editing and Enhancement
1. **Clean Up the Point Cloud**:
   - Remove unwanted points using Unity tools or filtering scripts.

2. **Add Lighting**:
   - Configure **Directional Light** for sunlight and **Point Lights** for realism.
   - Adjust intensity and shadows.

3. **Add a Fly Camera for Exploration**:
   - Attach a **Fly Camera script** to the main camera.
   - Use **WASD** keys for movement, mouse for rotation.
   - Customize speed and sensitivity for smooth navigation.

4. **Configure Input for Windows**:
   - Map keyboard and mouse controls to the Fly Camera.
   - Adjust input sensitivity using the Unity Input System.

#### 4.3 Exporting the Scene to Windows
1. **Set Up Windows Development**:
   - Open **Build Settings** (`File -> Build Settings`).
   - Select **Windows**, then **Switch Platform**.
   - Choose **x86 or x86_64** based on system specs.

2. **Configure Build Settings**:
   - Adjust **resolution**, **graphics API (DX12/Vulkan)**, and **quality settings**.
   - Select **Fullscreen/Windowed mode**.

3. **Optimize for Desktop Performance**:
   - Adjust **Quality Settings** for performance.
   - Enable **GPU instancing** to optimize rendering.

4. **Build and Deploy**:
   - Click **Build and Run**.
   - Unity compiles an executable for Windows.
   - Explore the reconstructed scene interactively.

5. **Testing and Debugging**:
   - Test responsiveness and rendering quality.
   - Use **Unity Profiler** (`Window -> Analysis -> Profiler`) to optimize performance.

6. **Final Deployment**:
   - Distribute the executable via **USB, network, or digital download**.

## Evaluation Metrics
- **Visual Quality**: Image similarity & realism.
- **Performance**: FPS during real-time interaction.
- **Computational Efficiency**: Training & rendering speed.

## Future Enhancements
- **Multi-user collaboration** in forensic VR environments.
- **Automated object detection** for evidence marking.
- **AI-enhanced forensic analysis** using deep learning models.

## Contributors
- **Saksham Ashwini Rai** ([LinkedIn](https://linkedin.com/in/yourprofile))

## License
MIT License - See [LICENSE](LICENSE) for details.
