from flask import Flask, Response, render_template, url_for, request
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

from jetson_inference import backgroundNet
from jetson_utils import (videoSource, videoOutput, loadImage, Log, cudaFromNumpy, cudaToNumpy,
                          cudaAllocMapped, cudaMemcpy, cudaResize, cudaOverlay)
from segnet_utils import *
from landingai.predict import Predictor
from scipy.spatial.distance import pdist

# Enter your API Key
endpoint_id = "a7f224da-618e-44d3-8bf7-90132df1d6c2"
api_key = "land_sk_2rHJ96g1xDsTML3fkwwtHGY7ffnR4aPI1Xrw9RvKtC23UjklA5"

app = Flask(__name__)

# Global variables
latest_image = None
captured_image = None
glossy_percentage = None
slab_mask_image = None  # To hold the slab mask image
gloss_mask_image = None  # To hold the gloss mask image
processed_image_1 = None  # New variable for processed image 1
processed_image_2 = None  # New variable for processed image 2
similarity_score = 100  # Initialize with 100%
distribution_percentage = 0.0
previous_sheen_percentage = None

image_lock = threading.Lock()
camera_source = None # Add a variable to store the camera source
img_replacement_scaled = None
img_output = None
sheen_percentage = 0.0
lower_gloss_thresh = 150
upper_gloss_thresh = 230
predictor_c = None
new_capture = False
switch_model = False
current_model = False
total_sheen_area = 0

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
                endpoint_id = "a7f224da-618e-44d3-8bf7-90132df1d6c2"
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
    global latest_image, processed_image_1, processed_image_2, sheen_percentage, new_capture
    global total_sheen_area, distribution_percentage

    while True:
        if latest_image is not None:
            with image_lock:
                # Copy the latest image to avoid threading issues
                image_to_process = latest_image.copy()

            # Convert the NumPy image to a CUDA image
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2BGRA)
            cuda_image = cudaFromNumpy(image_to_process)

            # Perform background removal
            net.Process(cuda_image, filter="linear")
            image_wo_background = cudaToNumpy(cuda_image)
            image_wo_background_noalpha = cv2.cvtColor(image_wo_background, cv2.COLOR_BGRA2RGB)

            if new_capture:
                predictions = run_inference(image_wo_background_noalpha)
                if predictions:
                    
                    # Initialize variables
                    total_sheen_area = 0
                    bbox_centers = []

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
                        text = f"{label}: {confidence:.2f}"
                        cv2.putText(image_wo_background_noalpha, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

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
                        distribution_percentage = normalized_distance * 100
                    else:
                        # If only one bounding box, set distribution to minimal value
                        distribution_percentage = 0

                else:
                    # If no predictions, set distribution percentage to zero
                    distribution_percentage = 0
                    total_sheen_area = 0

                processed_image_2 = image_wo_background_noalpha.copy()
                new_capture = False

            # Calculate sheen percentage from processed_image_1
            sheen_percentage = calculate_sheen_percentage(latest_image)

        time.sleep(1)  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html',
                           sheen_percentage=round(sheen_percentage, 2),
                           processed_image_2=processed_image_2,
                           switch_model=switch_model,
                           similarity_score=round(similarity_score, 2),
                           distribution_percentage=round(distribution_percentage, 2))

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

    return render_template('index.html',
                           sheen_percentage=round(sheen_percentage, 2),
                           processed_image_2=processed_image_2,
                           switch_model=switch_model,  # Pass to template
                           similarity_score=round(similarity_score, 2),
                           distribution_percentage=round(distribution_percentage, 2))


def calculate_sheen_percentage(image):
    global previous_sheen_percentage, similarity_score
    global lower_gloss_thresh, upper_gloss_thresh, total_sheen_area
    
    total_pixels = image.shape[0] * image.shape[1]
    percentage = (total_sheen_area / total_pixels) * 100

    # Calculate similarity score
    if previous_sheen_percentage is not None:
        similarity_score = 100 - abs(sheen_percentage - previous_sheen_percentage)
    else:
        similarity_score = 100  # First image comparison

    # Update previous sheen percentage
    previous_sheen_percentage = sheen_percentage
    return percentage


# @app.route('/processed_image_2_stream')
# def processed_image_2_stream():
#     def generate():
#         if processed_image_2 is not None:
#             with image_lock:
#                 ret, jpeg = cv2.imencode('.jpg', processed_image_2)
#             if ret:
#                 frame = jpeg.tobytes()
#                 yield (b'--frame\r\n'
#                         b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#     return Response(generate(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image_2_stream')
def processed_image_2_stream():
    def generate():
        while True:
            with image_lock:
                if processed_image_2 is not None:
                    success, jpeg = cv2.imencode('.jpg', processed_image_2.copy())
                    if success:
                        frame = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # Control the refresh rate (e.g., 1 second):
            time.sleep(2)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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
        camera.PixelFormat.SetValue('RGB8')
        camera.Close()
    

    # load the background removal network
    net = backgroundNet(args.network, sys.argv)
    input = videoSource(args.input, argv=sys.argv)

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
        endpoint_id = "a7f224da-618e-44d3-8bf7-90132df1d6c2"
        
    predictor_c = Predictor(endpoint_id, api_key=api_key)

    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True)
