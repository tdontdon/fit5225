# tagging_processor.py
from ultralytics import YOLO
import supervision as sv
import cv2 as cv # OpenCV
import numpy as np
import os

# Default confidence threshold if not provided
DEFAULT_CONFIDENCE_THRESHOLD = 0.5 # You can adjust this

def extract_tags_from_image(image_path, model_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Loads an image, uses YOLO model for prediction, and returns a list of detected bird names (tags)
    and a dictionary of their counts.
    """
    detected_tags_list = []
    detected_counts_dict = {}

    try:
        # Load YOLO model
        model = YOLO(model_path)
        class_dict = model.names # Dictionary of class IDs to class names

        # Load image from local path
        img = cv.imread(image_path)

        if img is None:
            print(f"Error: Couldn't load image from {image_path}")
            return [], {} # Return empty if image can't be loaded

        # Run the model on the image
        result = model(img)[0] # result is a list, take the first element for single image

        # Convert YOLO result to Detections format
        detections = sv.Detections.from_ultralytics(result)

        # Filter detections based on confidence threshold
        if detections.class_id is not None and len(detections.class_id) > 0:
            confident_detections = detections[detections.confidence > confidence_threshold]

            if confident_detections.class_id is not None and len(confident_detections.class_id) > 0:
                for cls_id in confident_detections.class_id:
                    tag_name = class_dict.get(int(cls_id), "Unknown") # Ensure cls_id is int
                    detected_tags_list.append(tag_name)
                    detected_counts_dict[tag_name] = detected_counts_dict.get(tag_name, 0) + 1
        
        unique_tags = sorted(list(set(detected_tags_list)))
        print(f"Image - Detected unique tags: {unique_tags}, Counts: {detected_counts_dict}")
        return unique_tags, detected_counts_dict

    except Exception as e:
        print(f"Error during image prediction in tagging_processor: {str(e)}")
        return [], {} # Return empty on error

def extract_tags_from_video(video_path, model_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, frames_to_sample=10, frame_skip_rate=30):
    """
    Processes video frames to extract aggregated bird names and counts.
    Samples frames to avoid processing the entire video which can be resource-intensive.
    """
    aggregated_tags_list = []
    aggregated_counts_dict = {}
    cap = None

    try:
        model = YOLO(model_path)
        class_dict = model.names

        cap = cv.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Couldn't open video from {video_path}")
            return [], {}

        video_info = sv.VideoInfo.from_video_path(video_path=video_path)
        # tracker = sv.ByteTrack(frame_rate=video_info.fps) # Tracker might be overkill if just aggregating counts

        total_frames_in_video = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # Adjust frame_skip_rate if video is too short to get enough samples
        if total_frames_in_video < frame_skip_rate * frames_to_sample and total_frames_in_video > 0 :
             # if video has 100 frames, and we want 10 samples, skip rate should be 10
             # if video has 20 frames, and we want 10 samples, skip rate should be 2
            calculated_skip_rate = total_frames_in_video // frames_to_sample
            frame_skip_rate = max(1, calculated_skip_rate) # ensure it's at least 1
            print(f"Adjusted frame_skip_rate to {frame_skip_rate} for video {video_path} with {total_frames_in_video} frames.")


        current_frame_num = 0
        processed_sample_count = 0

        while cap.isOpened() and processed_sample_count < frames_to_sample:
            ret, frame = cap.read()
            if not ret:
                break # End of video

            current_frame_num += 1

            if current_frame_num % frame_skip_rate == 0 or frame_skip_rate == 1: # Process this frame
                processed_sample_count += 1
                # print(f"Processing frame {current_frame_num} (Sample {processed_sample_count}) for video {video_path}")
                
                result = model(frame)[0] # Perform detection
                detections = sv.Detections.from_ultralytics(result)
                # detections = tracker.update_with_detections(detections=detections) # Optional tracking

                if detections.class_id is not None and len(detections.class_id) > 0:
                    confident_detections = detections[detections.confidence > confidence_threshold]
                    
                    if confident_detections.class_id is not None and len(confident_detections.class_id) > 0:
                        for cls_id in confident_detections.class_id:
                            tag_name = class_dict.get(int(cls_id), "Unknown")
                            # For videos, we collect all detections across sampled frames
                            # and then count them at the end.
                            aggregated_tags_list.append(tag_name) 
            
            # If frame_skip_rate is high, ensure we don't read too many unskipped frames either
            # This logic might need refinement depending on desired video processing depth
            if current_frame_num > frame_skip_rate * (frames_to_sample + 5) and frame_skip_rate > 1: # Add a buffer
                 print(f"Reached max frames to scan for video {video_path}")
                 break


        # After processing sampled frames, count the aggregated tags
        for tag in aggregated_tags_list:
            aggregated_counts_dict[tag] = aggregated_counts_dict.get(tag, 0) + 1
        
        unique_aggregated_tags = sorted(list(set(aggregated_tags_list)))

        print(f"Video processing: Frames read up to {current_frame_num}, Samples processed: {processed_sample_count}")
        print(f"Video - Detected unique tags: {unique_aggregated_tags}, Counts: {aggregated_counts_dict}")
        return unique_aggregated_tags, aggregated_counts_dict

    except Exception as e:
        print(f"Error during video prediction in tagging_processor: {str(e)}")
        return [], {}
    finally:
        if cap:
            cap.release()

def extract_tags_from_audio(audio_path, model_path, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Placeholder for audio file tagging. YOLO is a visual model.
    A different model/approach is needed for audio.
    """
    print(f"Warning: Audio file tagging with a visual YOLO model ({model_path}) is not directly supported.")
    print(f"Audio file received: {audio_path}. Returning empty tags for now.")
    # Implement actual audio processing logic here if applicable,
    # possibly calling a different model or service.
    return [], {}