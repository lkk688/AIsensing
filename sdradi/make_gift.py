import os
import re
from PIL import Image

def create_gif(image_folder, output_path, start_idx=0, end_idx=39, duration=500):
    images = []
    
    # Regex to extract number from filename (e.g., capture_005.png)
    pattern = re.compile(r'radar_v2_roi_(\d+)\.png')
    
    files = []
    # List all files in the directory
    try:
        all_files = os.listdir(image_folder)
    except FileNotFoundError:
        print(f"Error: Directory '{image_folder}' not found.")
        return

    for filename in all_files:
        match = pattern.match(filename)
        if match:
            idx = int(match.group(1))
            # Filter by the requested range
            if start_idx <= idx <= end_idx:
                files.append((idx, filename))
    
    # Sort by index to ensure correct order
    files.sort(key=lambda x: x[0])
    
    if not files:
        print(f"No files found in the range {start_idx} to {end_idx} in '{image_folder}'.")
        return

    print(f"Found {len(files)} images in range [{start_idx}, {end_idx}].Processing...")

    # Load images
    for idx, filename in files:
        file_path = os.path.join(image_folder, filename)
        try:
            img = Image.open(file_path)
            # Ensure consistent mode (e.g. RGB)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not open {file_path}: {e}")
        
    if images:
        # Save as GIF
        # duration is in milliseconds per frame
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Successfully created GIF: {output_path}")
    else:
        print("No valid images collected.")

if __name__ == "__main__":
    # Define paths
    # Assuming script is run from /Developer/AIsensing
    folder_path = 'sdradi/output/radar_v2i'
    output_gif = 'sdradi/output/radar_v2i/animation3.gif'
    
    create_gif(folder_path, output_gif, start_idx=0, end_idx=39, duration=500)
