import cv2
import os

brightness_factor=1.5

def crop_and_save_videos(input_dirs, output_dir, desired_frames=30, skip_frames=10):
    for input_dir in input_dirs:
        input_dir_name = os.path.basename(os.path.normpath(input_dir))  # Extract the last part of the input directory path
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".mov") or file.endswith(".mp4"):  # Adjust file extensions as needed
                    input_video_path = os.path.join(root, file)
                    rel_path = os.path.relpath(root, input_dir)
                    output_video_dir = os.path.join(output_dir, input_dir_name, rel_path)  # Include the last part of the input directory path in the output directory
                    os.makedirs(output_video_dir, exist_ok=True)
                    output_video_path = os.path.join(output_video_dir, os.path.splitext(file)[0] + '_cropped' + os.path.splitext(file)[1])
                    # crop_video(input_video_path, output_video_path, desired_frames, skip_frames)
                    crop_and_adjust_brightness(input_video_path, output_video_path, desired_frames, skip_frames, brightness_factor)


# def crop_and_save_videos(input_dirs, output_dir, desired_frames=30, skip_frames=10, brightness_factor=1.5):
#     for input_dir in input_dirs:
#         for root, _, files in os.walk(input_dir):
#             for file in files:
#                 if file.endswith('.mov') or file.endswith('.mp4'):
#                     input_video_path = os.path.join(root, file)
#                     output_video_subdir = os.path.relpath(root, input_dir)  # Relative path from input directory to current directory
#                     output_video_path = os.path.join(output_dir, output_video_subdir, os.path.splitext(file)[0])
#                     crop_and_adjust_brightness(input_video_path, output_video_path, desired_frames, skip_frames, brightness_factor)


def crop_and_adjust_brightness(input_path, output_path, desired_frames, skip_frames=10, brightness_factor=1.5):
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    writer_cropped = None
    writer_brightened = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count >= skip_frames and frame_count < (skip_frames + desired_frames):
            if writer_cropped is None:
                # Initialize the video writer for cropped frames
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                writer_cropped = cv2.VideoWriter(output_path + '_cropped.mov', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # Write the frame to the output video
            writer_cropped.write(frame)

            if writer_brightened is None:
                # Initialize the video writer for brightened frames
                output_path_brightened = os.path.splitext(output_path)[0] + '_cropped_b' + os.path.splitext(output_path)[1]
                writer_brightened = cv2.VideoWriter(output_path_brightened, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

            # Increase brightness
            frame_brightened = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)

            # Write the brightened frame to the output video
            writer_brightened.write(frame_brightened)
            
        elif frame_count >= (skip_frames + desired_frames):
            break

        frame_count += 1

    cap.release()
    if writer_cropped is not None:
        writer_cropped.release()
    if writer_brightened is not None:
        writer_brightened.release()

# def crop_video(input_path, output_path, desired_frames, skip_frames=10):
#     cap = cv2.VideoCapture(input_path)
#     frame_count = 0
#     writer = None

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_count >= skip_frames and frame_count < (skip_frames + desired_frames):
#             if writer is None:
#                 # Initialize the video writer with the same properties as the input video
#                 frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#                 frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#                 fps = cap.get(cv2.CAP_PROP_FPS)
#                 writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

#             # Write the frame to the output video
#             writer.write(frame)
#         elif frame_count >= (skip_frames + desired_frames):
#             break

#         frame_count += 1

#     cap.release()
#     if writer is not None:
#         writer.release()


input_dirs = ['datasets/my_videos/afternoon']  # Add more directories as needed
output_dir = 'datasets/extracted_rar/Videos'
crop_and_save_videos(input_dirs, output_dir)