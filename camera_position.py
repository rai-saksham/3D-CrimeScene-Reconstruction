import json

# Load and inspect Camera Parameters from cameras.json
def load_camera_params(camera_json_path):
    with open(camera_json_path, 'r') as f:
        camera_data = json.load(f)
    
    # Print out the JSON structure for inspection
    print("Camera JSON Structure:", camera_data)
    
    # Depending on the structure, adapt the extraction process here
    # For example, if camera_data is a list of cameras:
    camera_params = camera_data[0]  # Adjust the indexing based on your file's structure

    return camera_params

# Example usage
camera_json_path = "C:/Users/raisa/gaussian-splatting_best/output/Gundum3/cameras.json"
load_camera_params(camera_json_path)
