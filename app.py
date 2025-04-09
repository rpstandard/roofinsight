from flask import Flask, Response, render_template, url_for, request, jsonify
from pypylon import pylon
import numpy as np
import cv2
import io
from PIL import Image
import base64
import threading
import time
import sys
import argparse
import os
from datetime import datetime
import json

from jetson_inference import backgroundNet
from jetson_utils import (videoSource, videoOutput, loadImage, Log, cudaFromNumpy, cudaToNumpy,
                          cudaAllocMapped, cudaMemcpy, cudaResize, cudaOverlay)
from segnet_utils import *
from landingai.predict import Predictor
from scipy.spatial.distance import pdist

# Enter your API Key
endpoint_id = "9346ab6c-1938-4b87-ac42-f698d7ee0eda" # "a7f224da-618e-44d3-8bf7-90132df1d6c2"
api_key = "land_sk_2rHJ96g1xDsTML3fkwwtHGY7ffnR4aPI1Xrw9RvKtC23UjklA5"

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Directory to save images
SAVE_DIR = '/home/roofinsights/workspace/roofinsight/static/images'
RAW_SAVE_DIR = '/home/roofinsights/workspace/roofinsight/static/raw_images'
CONFIG_FILE = '/home/roofinsights/workspace/roofinsight/config.json'
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RAW_SAVE_DIR, exist_ok=True)

# Global variables
latest_image = None
processed_image_1 = None  # New variable for processed image 1
processed_image_2 = None  # New variable for processed image 2
similarity_score = 100  # Initialize with 100%
distribution_score = 0.0
previous_sheen_percentage = None
count_score = 0
pixel_count_shingle = 0
background_removal_enabled = True  # New variable for background removal state
landing_lens_enabled = True  # New variable for LandingLens model state
sheen_category = 'no_sheen'  # Default sheen category

image_lock = threading.Lock()
camera_source = None # Add a variable to store the camera source
sheen_percentage = 0.0
predictor_c = None
new_capture = False
switch_model = False
current_model = False
total_sheen_area = 0
shingle_analysis_dst = None
last_10_files = []
chartLabels = []
chartValueDS = []
chartValueSP = []
chartValueCS = []

# Load configuration from file
def load_config():
    global background_removal_enabled, landing_lens_enabled, sheen_category
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                background_removal_enabled = config.get('background_removal_enabled', True)
                landing_lens_enabled = config.get('landing_lens_enabled', True)
                sheen_category = config.get('sheen_category', 'no_sheen')
    except Exception as e:
        print(f"Error loading config: {e}")

# Save configuration to file
def save_config():
    try:
        config = {
            'background_removal_enabled': background_removal_enabled,
            'landing_lens_enabled': landing_lens_enabled,
            'sheen_category': sheen_category
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except Exception as e:
        print(f"Error saving config: {e}")

# Load configuration on startup
load_config()

def load_chart_data():
    global chartLabels, chartValueDS, chartValueSP, chartValueCS
    try:
        with open('chart_data.json', 'r') as f:
            data = json.load(f)
            chartLabels = data.get('chartLabels', [])
            chartValueDS = data.get('chartValueDS', [])
            chartValueSP = data.get('chartValueSP', [])
            chartValueCS = data.get('chartValueCS', [])
    except FileNotFoundError:
        # File does not exist, initialize with empty lists
        chartLabels = []
        chartValueDS = []
        chartValueSP = []
        chartValueCS = []

# filepath: /home/roofinsights/workspace/roofinsight/app.py
def save_chart_data():
    data = {
        'chartLabels': chartLabels,
        'chartValueDS': chartValueDS,
        'chartValueSP': chartValueSP,
        'chartValueCS': chartValueCS
    }
    with open('chart_data.json', 'w') as f:
        json.dump(data, f)

# Update history with new values
def update_history(distribution_score, sheen_percentage, count_score):
    timestamp = datetime.now().strftime('%m/%d %H:%M')

    # Ensure the lists do not exceed 7 elements
    if len(chartLabels) > 10:
        chartLabels.pop(0)
    if len(chartValueDS) > 10:
        chartValueDS.pop(0)
    if len(chartValueSP) > 10:
        chartValueSP.pop(0)
    if len(chartValueCS) > 10:
        chartValueCS.pop(0)

    chartLabels.append(timestamp)
    chartValueDS.append(round(distribution_score*5, 2))
    chartValueSP.append(round(sheen_percentage, 2))
    chartValueCS.append(count_score)
    save_chart_data()
    
def run_inference(image):
    global predictor_c, api_key, switch_model, current_model
    """
    Runs inference on the provided image using the Predictor.

    Args:
        image_path (str): Path to the image file.

    Returns:
        predictions (dict or list): The inference results.
    """
    try:
        if current_model != switch_model:
            if switch_model:
                endpoint_id = "84f11d29-7e8c-417f-97af-9c11342e0c49"
            else:
                endpoint_id = "9346ab6c-1938-4b87-ac42-f698d7ee0eda"
            predictor_c = Predictor(endpoint_id, api_key=api_key)
            current_model = switch_model
        
        # Run inference
        predictions = predictor_c.predict(image)
        return predictions
    
    except Exception as e:
        print(f"Inference Error: {e}")
        return None
    
def capture_image():
    global latest_image, camera_source
    if camera_source == 'pylon':
        camera.Open()
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        try:
            while camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    with image_lock:
                        latest_image = grab_result.Array.copy()
                grab_result.Release()
                time.sleep(1)  # Adjust as needed
        except Exception as e:
            print(f"Error: {e}")
        finally:
            camera.StopGrabbing()
            camera.Close() 
    elif camera_source == 'usb':
        # Use the USB camera at /dev/video0
        cap = cv2.VideoCapture('/dev/video0')
        if not cap.isOpened():
            print("Cannot open USB camera")
            return
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                with image_lock:
                    latest_image = frame.copy()
                time.sleep(1) # Adjust as needed
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
    else:
        print("Invalid camera source specified")

def process_image_in_thread():
    global latest_image, processed_image_1, processed_image_2, sheen_percentage, new_capture, pixel_count_shingle
    global total_sheen_area, distribution_score, shingle_analysis_dst, last_10_files, count_score
    global background_removal_enabled, landing_lens_enabled

    while True:
        if latest_image is not None:
            with image_lock:
                # Copy the latest image to avoid threading issues
                image_to_process = latest_image.copy()

            # Convert the NumPy image to a CUDA image
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2BGRA)
            cuda_image = cudaFromNumpy(image_to_process)

            # Perform background removal only if enabled
            if background_removal_enabled:
                net.Process(cuda_image, filter="linear")
                image_wo_background = cudaToNumpy(cuda_image)
                image_wo_background_noalpha = cv2.cvtColor(image_wo_background, cv2.COLOR_BGRA2RGB)
            else:
                image_wo_background = cudaToNumpy(cuda_image)
                image_wo_background_noalpha = cv2.cvtColor(image_to_process, cv2.COLOR_BGRA2RGB)

            if new_capture:
                predictions = None
                if landing_lens_enabled:
                    predictions = run_inference(image_wo_background_noalpha)
                if predictions:
                    
                    # Initialize variables
                    total_sheen_area = 0
                    bbox_centers = []
                    count_score = len(predictions)
                    for obj in predictions:
                        label = obj.label_name
                        confidence = obj.score
                        x1, y1, x2, y2 = map(int, obj.bboxes)

                        # Calculate the area of the bounding box
                        area = (x2 - x1) * (y2 - y1)
                        total_sheen_area += area

                        # Calculate center of the bounding box
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        bbox_centers.append((center_x, center_y))

                        # Draw red bounding box
                        cv2.rectangle(image_wo_background_noalpha, (x1, y1), (x2, y2), (0, 0, 255), 4)

                        # Optionally, add label and confidence
                        text = f"*{confidence:.2f}"
                        cv2.putText(image_wo_background_noalpha, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)

                    # Calculate distribution percentage based on spatial spread
                    if len(bbox_centers) > 1:

                        # Convert centers to NumPy array
                        centers_array = np.array(bbox_centers)

                        # Compute pairwise distances between bounding box centers
                        distances = pdist(centers_array)

                        # Calculate average pairwise distance
                        avg_distance = np.mean(distances)

                        # Normalize average distance by image diagonal
                        height, width = image_wo_background_noalpha.shape[:2]
                        image_diagonal = np.sqrt(height**2 + width**2)
                        normalized_distance = avg_distance / image_diagonal

                        # Calculate distribution percentage (scaled to 0-100)
                        distribution_score = normalized_distance * 100
                    else:
                        # If only one bounding box, set distribution to minimal value
                        distribution_score = 0

                else:
                    # If no predictions, set distribution percentage to zero
                    distribution_score = 0
                    total_sheen_area = 0
                    count_score = 0

                processed_image_2 = image_wo_background_noalpha.copy()
                tmp = cv2.cvtColor(processed_image_2, cv2.COLOR_BGR2GRAY)
                _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                pixel_count_shingle = cv2.countNonZero(alpha)
                b, g, r = cv2.split(processed_image_2)
                rgba = [b,g,r, alpha]
                shingle_analysis_dst = cv2.merge(rgba,4)
                success, png = cv2.imencode('.png', shingle_analysis_dst.copy())
                if success:
                    frame = png.tobytes()
                    # Save the image with UTC time and date
                    utc_now = datetime.now()
                    date_str = utc_now.strftime('%Y%m%d')
                    time_str = utc_now.strftime('%H%M%S')
                    date_dir = os.path.join(SAVE_DIR, date_str)
                    os.makedirs(date_dir, exist_ok=True)
                    filename = os.path.join(date_dir, f'shingle_{time_str}.png')
                    with open(filename, 'wb') as f:
                        f.write(frame)

                # Get list of files in the directory
                files = []
                for date_dir in os.listdir(SAVE_DIR):
                    date_path = os.path.join(SAVE_DIR, date_dir)
                    if os.path.isdir(date_path):
                        files.extend([os.path.join(date_path, f) for f in os.listdir(date_path) if f.endswith('.png')])
                # Sort files by modification time
                files.sort(key=os.path.getmtime, reverse=True)
                # Get the last 10 files
                last_10_files = files[:4]
                new_capture = False

                # Calculate sheen percentage from processed_image_1
                sheen_percentage = calculate_sheen_percentage(shingle_analysis_dst)
                update_history(distribution_score, sheen_percentage, count_score)

        time.sleep(1)  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html',
                           sheen_percentage=round(sheen_percentage, 2),
                           processed_image_2=processed_image_2,
                           switch_model=switch_model,
                           similarity_score=round(similarity_score, 2),
                           distribution_score=round(distribution_score*5, 2),
                           last_images=last_10_files,
                           num_patches=count_score,
                           chartLabels=chartLabels,
                           chartValueDS=chartValueDS,
                           chartValueSP=chartValueSP,
                           chartValueCS=chartValueCS,
                           background_removal_enabled=background_removal_enabled,
                           landing_lens_enabled=landing_lens_enabled,
                           sheen_category=sheen_category)

@app.route('/analytics')
def analytics_dashboard():
    # Directory to save images
    SHINGLE_IMAGES_DIR = 'static/images'
    
    # Get list of files in the directory
    files = []
    for date_dir in os.listdir(SHINGLE_IMAGES_DIR):
        date_path = os.path.join(SHINGLE_IMAGES_DIR, date_dir)
        if os.path.isdir(date_path):
            files.extend([os.path.join(date_path, f) for f in os.listdir(date_path) if f.endswith('.png')])
    
    # Sort files by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    # Get the last 50 files
    last_30_files = files[:50]

    # Make paths relative to the static directory
    last_30_files = [f.replace('static/', '') for f in last_30_files]

    # Pass these file paths to the template
    return render_template('analytics.html', 
                           images=last_30_files,
                           sheen_percentage=round(sheen_percentage, 2),
                           processed_image_2=processed_image_2,
                           switch_model=switch_model,
                           similarity_score=round(similarity_score, 2),
                           distribution_score=round(distribution_score*5, 2),
                           last_images=last_10_files,
                           num_patches=count_score,
                           chartLabels=chartLabels,
                           chartValueDS=chartValueDS,
                           chartValueSP=chartValueSP,
                           chartValueCS=chartValueCS)

def generate():
    global latest_image

    while True:
        if latest_image is not None:
            # Convert RGB to BGR
            latest_image = cv2.cvtColor(latest_image, cv2.COLOR_RGB2BGR)

            # Convert the image to JPEG format
            ret, jpeg = cv2.imencode('.jpg', latest_image)
            if ret:
                # Return the image as a byte stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


        # Add a delay of 1 second
        time.sleep(1)

@app.route('/video_stream')
def video_stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global latest_image, sheen_percentage
    global new_capture, switch_model

    if request.method == 'POST':
        # Get the selected model from the form
        model_selection = request.form.get('model_selection', 'model_a')
        switch_model = True if model_selection == 'model_b' else False
        new_capture = True

        # Save raw image when capture button is clicked
        if latest_image is not None:
            utc_now = datetime.now()
            date_str = utc_now.strftime('%Y%m%d')
            time_str = utc_now.strftime('%H%M%S')
            
            # Create category subfolder
            category_dir = os.path.join(RAW_SAVE_DIR, sheen_category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Create date subfolder within category
            date_dir = os.path.join(category_dir, date_str)
            os.makedirs(date_dir, exist_ok=True)
            
            raw_filename = os.path.join(date_dir, f'raw_{time_str}.png')
            cv2.imwrite(raw_filename, latest_image)

    return render_template('index.html',
                           sheen_percentage=round(sheen_percentage, 2),
                           processed_image_2=processed_image_2,
                           switch_model=switch_model,  # Pass to template
                           similarity_score=round(similarity_score, 2),
                           distribution_score=round(distribution_score*5, 2),
                           last_images=last_10_files,
                           num_patches=count_score,
                           chartLabels=chartLabels,
                           chartValueDS=chartValueDS,
                           chartValueSP=chartValueSP,
                           chartValueCS=chartValueCS,
                           background_removal_enabled=background_removal_enabled,
                           landing_lens_enabled=landing_lens_enabled,
                           sheen_category=sheen_category)


def calculate_sheen_percentage(image):
    global previous_sheen_percentage, similarity_score, total_sheen_area, pixel_count_shingle
    
    total_pixels = pixel_count_shingle
    percentage = ((total_sheen_area*4 / total_pixels) * 100) if total_pixels != 0 else 0

    # Calculate similarity score
    if previous_sheen_percentage is not None:
        similarity_score = 100 - abs(sheen_percentage - previous_sheen_percentage)
    else:
        similarity_score = 100  # First image comparison

    # Update previous sheen percentage
    previous_sheen_percentage = sheen_percentage
    return percentage


@app.route('/processed_image_2_stream')
def processed_image_2_stream():
    def generate():
        while True:
            with image_lock:
                if processed_image_2 is not None:
                    # Add a cache-busting parameter to the response headers
                    success, png = cv2.imencode('.png', shingle_analysis_dst.copy())
                    if success:
                        frame = png.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/png\r\n'
                               b'Cache-Control: no-cache, no-store, must-revalidate\r\n'
                               b'Pragma: no-cache\r\n'
                               b'Expires: 0\r\n\r\n' + frame + b'\r\n\r\n')
                    
            # Control the refresh rate (e.g., 1 second):
            time.sleep(1)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_bg_removal', methods=['POST'])
def toggle_bg_removal():
    global background_removal_enabled
    data = request.get_json()
    background_removal_enabled = data.get('enabled', True)
    save_config()  # Save configuration after toggle
    return jsonify({'status': 'success', 'enabled': background_removal_enabled})

@app.route('/toggle_landing_lens', methods=['POST'])
def toggle_landing_lens():
    global landing_lens_enabled
    data = request.get_json()
    landing_lens_enabled = data.get('enabled', True)
    save_config()  # Save configuration after toggle
    return jsonify({'status': 'success', 'enabled': landing_lens_enabled})

@app.route('/set_sheen_category', methods=['POST'])
def set_sheen_category():
    global sheen_category
    data = request.get_json()
    sheen_category = data.get('category', 'no_sheen')
    save_config()  # Save configuration after category change
    return jsonify({'status': 'success', 'category': sheen_category})

if __name__ == '__main__':
    # parse the command line
    parser = argparse.ArgumentParser(description="Roofinsights 1.0 \nA Computer Vision Application for Roof Inspection", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=backgroundNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
    parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
    parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
    parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")
    parser.add_argument("--camera", type=str, default="pylon", choices=["pylon", "usb"], help="Camera source to use ('pylon' or 'usb')")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    camera_source = args.camera  # Set the camera source based on the argument

    if camera_source == 'pylon':
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        
        # Load camera configuration from .pfs file if it exists
        config_file = '/home/roofinsights/workspace/roofinsight/camera_config.pfs'
        if os.path.exists(config_file):
            try:
                # Load the configuration file
                pylon.FeaturePersistence.Load(config_file, camera.GetNodeMap(), True)
                print(f"Successfully loaded camera configuration from {config_file}")
            except Exception as e:
                print(f"Error loading camera configuration: {e}")
                # Fall back to default RGB8 setting if config loading fails
                camera.PixelFormat.SetValue('RGB8')
        else:
            # Use default RGB8 setting if no config file exists
            camera.PixelFormat.SetValue('RGB8')
            print(f"No camera configuration file found at {config_file}, using default RGB8 setting")
        
        camera.Close()
    

    # load the background removal network
    net = backgroundNet(args.network, sys.argv)
    input = videoSource(args.input, argv=sys.argv)
    load_chart_data()

    # Start the image capture thread
    capture_thread = threading.Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    # Start the image processing thread
    process_thread = threading.Thread(target=process_image_in_thread)
    process_thread.daemon = True
    process_thread.start()

    if switch_model:
        endpoint_id = "84f11d29-7e8c-417f-97af-9c11342e0c49"
    else:
        endpoint_id = "9346ab6c-1938-4b87-ac42-f698d7ee0eda"
        
    predictor_c = Predictor(endpoint_id, api_key=api_key)

    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True)