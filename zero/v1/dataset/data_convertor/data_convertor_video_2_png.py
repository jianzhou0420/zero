from tqdm import tqdm
import ffmpeg
import os


def save_frames_as_png(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        # Use FFmpeg to extract frames as PNGs
        ffmpeg.input(input_file).output(
            os.path.join(output_folder, 'frame_%04d.png'),  # Save frames as frame_0001.png, frame_0002.png, ...
            start_number=0,  # Frame numbering starts at 0
            # vf="fps=1",      # Extract frames at 1 frame per second (adjust if needed)
            format='image2',  # Format for individual images
        ).run(overwrite_output=True)
        print(f"Frames saved successfully in {output_folder}")
    except ffmpeg.Error as e:
        print("An error occurred while extracting frames:", e.stderr.decode())


# Example Usage

input_path = '/media/jian/data/pusht/videos/chunk-000/observation.image/'
mp4_list = [d for d in os.listdir(input_path) if d.endswith('.mp4')]

for mp4 in tqdm(mp4_list):
    input_file = os.path.join(input_path, mp4)
    output_folder = os.path.join(input_path, mp4.replace('.mp4', ''))
    save_frames_as_png(input_file, output_folder)
