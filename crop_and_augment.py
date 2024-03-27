import cv2
import os

brightness_factor = 1.1  # Increase or decrease for brightness adjustment
contrast_factor = 1.1  # Increase or decrease for contrast adjustment
rotation_angle = 5  # Angle in degrees for rotation

def augment_videos(input_dirs, output_dir, desired_frames=30, skip_frames=10):
    for input_dir in input_dirs:
        input_dir_name = os.path.basename(os.path.normpath(input_dir))  # Extract the last part of the input directory path
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".mov") or file.endswith(".mp4"):  # Adjust file extensions as needed
                    input_video_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_dir)
                    output_video_dir = os.path.join(output_dir, input_dir_name, rel_path)  # Include the last part of the input directory path in the output directory
                    os.makedirs(output_video_dir, exist_ok=True)
                    output_video_base = os.path.splitext(file)[0]
                    
                    # Original video
                    original_output_path = os.path.join(output_video_dir, output_video_base + '_original' + os.path.splitext(file)[1])
                    crop_and_adjust(input_video_path, original_output_path, desired_frames, skip_frames, brightness_factor=1.0, contrast_factor=1.0, rotation_angle=0)
                    
                    # Brightness increased
                    bright_output_path = os.path.join(output_video_dir, output_video_base + '_bright' + os.path.splitext(file)[1])
                    crop_and_adjust(input_video_path, bright_output_path, desired_frames, skip_frames, brightness_factor=brightness_factor, contrast_factor=1.0, rotation_angle=0)
                    
                    # Brightness decreased
                    dark_output_path = os.path.join(output_video_dir, output_video_base + '_dark' + os.path.splitext(file)[1])
                    crop_and_adjust(input_video_path, dark_output_path, desired_frames, skip_frames, brightness_factor=1/brightness_factor, contrast_factor=1.0, rotation_angle=0)
                    
                    # Contrast increased
                    high_contrast_output_path = os.path.join(output_video_dir, output_video_base + '_high_contrast' + os.path.splitext(file)[1])
                    crop_and_adjust(input_video_path, high_contrast_output_path, desired_frames, skip_frames, brightness_factor=1.0, contrast_factor=contrast_factor, rotation_angle=0)
                    
                    # Contrast decreased
                    low_contrast_output_path = os.path.join(output_video_dir, output_video_base + '_low_contrast' + os.path.splitext(file)[1])
                    crop_and_adjust(input_video_path, low_contrast_output_path, desired_frames, skip_frames, brightness_factor=1.0, contrast_factor=1/contrast_factor, rotation_angle=0)
                    
                    # Rotation
                    rotated_output_path = os.path.join(output_video_dir, output_video_base + '_rotated' + os.path.splitext(file)[1])
                    crop_and_adjust(input_video_path, rotated_output_path, desired_frames, skip_frames, brightness_factor=1.0, contrast_factor=1.0, rotation_angle=rotation_angle)

def crop_and_adjust(input_path, output_path, desired_frames, skip_frames=10, brightness_factor=1.0, contrast_factor=1.0, rotation_angle=0):
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    writer = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= skip_frames and frame_count < (skip_frames + desired_frames):
            if writer is None:
                # Initialize the video writer with the same properties as the input video
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # Apply brightness adjustment
            frame_brightened = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
            
            # Apply contrast adjustment
            frame_contrast = cv2.convertScaleAbs(frame_brightened, alpha=contrast_factor, beta=0)
            
            # Apply rotation
            matrix = cv2.getRotationMatrix2D((frame_width / 2, frame_height / 2), rotation_angle, 1)
            frame_rotated = cv2.warpAffine(frame_contrast, matrix, (frame_width, frame_height))
            
            # Write the frame to the output video
            writer.write(frame_rotated)
            
        elif frame_count >= (skip_frames + desired_frames):
            break

        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()

# Example usage:
input_dirs = ['datasets/my_videos/angry', 'datasets/my_videos/bye', 'datasets/my_videos/chair',
              'datasets/my_videos/computer', 'datasets/my_videos/confused', 'datasets/my_videos/drink',
              'datasets/my_videos/eat', 'datasets/my_videos/evening', 'datasets/my_videos/excited']
output_dir = 'datasets/extracted_rar/Videos'
augment_videos(input_dirs, output_dir)
