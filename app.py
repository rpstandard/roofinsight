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
processed_image_3 = None  # New variable for processed image 3
processed_image_4 = None  # New variable for processed image 4

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
        # Load the image using PIL
        # image = Image.open(image_path)
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
        # Use the Pylon camera
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        try:
            while camera.IsGrabbing():
                grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grab_result.GrabSucceeded():
                    with image_lock:
                        latest_image = grab_result.Array.copy()
                grab_result.Release()
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
                time.sleep(1)  # Approximate 30 FPS
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
    else:
        print("Invalid camera source specified")

    time.sleep(1)  # Adjust as needed

def replaceBackground(img_input):
    # Define solid orange color (RGB)
    orange_color = (255, 165, 0, 255)  # Red, Green, Blue, Alpha
    # Create a solid orange image using NumPy
    orange_np = np.full((img_input.height, img_input.width, 4), orange_color, dtype=np.uint8)

    # Convert NumPy array to CUDA image
    orange_background_scaled = cudaFromNumpy(orange_np)

    # Overlay the original image onto the orange background
    img_output = cudaAllocMapped(like=img_input)
    cudaOverlay(img_input, orange_background_scaled, 0, 0)

    return orange_background_scaled

def process_image_in_thread():
    global latest_image, processed_image_1, processed_image_2, sheen_percentage, new_capture
    global total_sheen_area

    while True:
        if latest_image is not None:
            with image_lock:
                # Copy the latest image to avoid threading issues
                image_to_process = latest_image.copy()

            # Convert the NumPy image to a CUDA image
            image_to_process = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGBA)
            cuda_image = cudaFromNumpy(image_to_process)
            # Perform background removal
            net.Process(cuda_image, filter="linear")

            # Convert CUDA image back to NumPy for display or further processing
            processed_1 = cudaToNumpy(cuda_image)
            processed_1_final = cv2.cvtColor(processed_1, cv2.COLOR_RGBA2BGR)
            processed_image_1 = processed_1_final.copy()  # Store processed image 2

            if new_capture:
                processed_image_2_i = processed_image_1.copy()  # Store processed image 3
                predictions = run_inference(processed_image_2_i)
                if predictions:
                    # predictions is a list of ObjectDetectionPrediction instances
                    total_sheen_area = 0  # Initialize total area

                    for obj in predictions:
                        label = obj.label_name
                        confidence = obj.score
                        x1, y1, x2, y2 = map(int, obj.bboxes)

                        # Calculate the area of the bounding box
                        area = (x2 - x1) * (y2 - y1)
                        total_sheen_area += area  # Increment the total area

                        # Draw red bounding box
                        cv2.rectangle(processed_image_2_i, (x1, y1), (x2, y2), (0, 0, 255), 2)

                        # Optionally, add label and confidence
                        text = f"{label}: {confidence:.2f}"
                        cv2.putText(processed_image_2_i, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                processed_image_2 = processed_image_2_i.copy()
                new_capture = False
                processed_image_2_stream()

            # # Perform background replacement
            # img_output = replaceBackground(cuda_image)
            
            # # Convert CUDA image back to NumPy for display or further processing
            # processed_2 = cudaToNumpy(img_output)
            # processed_2_final = cv2.cvtColor(processed_2, cv2.COLOR_RGBA2BGR)

            # Calculate sheen percentage from processed_image_1
            sheen_percentage = calculate_sheen_percentage(processed_image_1)

        time.sleep(1)  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html',
                           captured_image=captured_image,
                           glossy_percentage=glossy_percentage,
                           sheen_percentage=sheen_percentage,
                           slab_mask_image=slab_mask_image,
                           gloss_mask_image=gloss_mask_image,
                           lower_gloss_thresh=lower_gloss_thresh,
                           upper_gloss_thresh=upper_gloss_thresh)

def generate():
    global latest_image
    # global mask_generator

    while True:
        if latest_image is not None:
            # Convert RGB to BGR
            latest_image = cv2.cvtColor(latest_image, cv2.COLOR_RGB2BGR)

            # Convert the image to JPEG format
            ret, jpeg = cv2.imencode('.jpg', latest_image)

            # mask = mask_generator.generate(latest_image)
            # show_anns(mask)
            # plt.figure(figsize=(20,20))

            if ret:
                # Return the image as a byte stream
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


        # Add a delay of 1 second
        time.sleep(1)

@app.route('/video')
def video_feed():
    return render_template('video_feed.html', captured_image=captured_image)

@app.route('/video_stream')
def video_stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET', 'POST'])
def capture():
    global latest_image, captured_image, glossy_percentage, slab_mask_image, gloss_mask_image, sheen_percentage
    global lower_gloss_thresh, upper_gloss_thresh, new_capture, switch_model

    if request.method == 'POST':
        # Get threshold values from the form
        lower_gloss_thresh = int(request.form.get('lower_gloss_thresh', 200))
        upper_gloss_thresh = int(request.form.get('upper_gloss_thresh', 255))

        # Get the selected model from the form
        model_selection = request.form.get('model_selection', 'model_a')
        switch_model = True if model_selection == 'model_b' else False

    if latest_image is not None:
        # Process the image with the provided thresholds
        processed_image, glossy_percentage, slab_mask_data, gloss_mask_data = process_image(
            latest_image, lower_gloss_thresh, upper_gloss_thresh)

        # Convert the original image from BGR to RGB
        original_image_rgb = cv2.cvtColor(latest_image, cv2.COLOR_BGR2RGB)
        # Encode the original image to display in the browser
        _, jpeg = cv2.imencode('.jpg', original_image_rgb)
        img_bytes = jpeg.tobytes()
        original_image = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        # Convert the processed image from BGR to RGB
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        # Encode the processed image to display in the browser
        _, jpeg = cv2.imencode('.jpg', processed_image_rgb)
        img_bytes = jpeg.tobytes()
        captured_image = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        # Set the intermediate images
        slab_mask_image = slab_mask_data
        gloss_mask_image = gloss_mask_data

        new_capture = True

    return render_template('index.html',
                           original_image=original_image,
                           captured_image=captured_image,
                           glossy_percentage=glossy_percentage,
                           sheen_percentage=sheen_percentage,
                           slab_mask_image=slab_mask_image,
                           processed_image_1=processed_image_1,
                           processed_image_2=processed_image_2,
                           switch_model=switch_model,  # Pass to template
                           gloss_mask_image=gloss_mask_image,
                           lower_gloss_thresh=lower_gloss_thresh,
                           upper_gloss_thresh=upper_gloss_thresh)

def process_image(image, lower_gloss_thresh, upper_gloss_thresh):
    """
    Process the image to segment wooden slabs, detect glossy patches,
    and calculate the glossy area percentage.
    Returns the processed image, glossy percentage, and intermediate images.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment wooden slabs
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours of the slabs
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Mask to hold the segmented slabs
    slab_mask = np.zeros_like(gray)

    # Draw the contours on the mask
    cv2.drawContours(slab_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Extract the slab regions
    slab_regions = cv2.bitwise_and(image, image, mask=slab_mask)

    # Convert the slab regions to HSV to detect glossy patches
    hsv = cv2.cvtColor(slab_regions, cv2.COLOR_BGR2HSV)

    # Define thresholds for detecting glossy patches using user-provided values
    lower_gloss = np.array([0, 0, lower_gloss_thresh])
    upper_gloss = np.array([180, 60, upper_gloss_thresh])

    # Create a mask for glossy areas
    gloss_mask = cv2.inRange(hsv, lower_gloss, upper_gloss)

    # Calculate the percentage of glossy area in the slab
    slab_area = cv2.countNonZero(slab_mask)
    gloss_area = cv2.countNonZero(gloss_mask)

    if slab_area > 0:
        glossy_percentage = (gloss_area / slab_area) * 100
    else:
        glossy_percentage = 0.0

    # Highlight the glossy patches on the image
    result_image = image.copy()
    # Overlay glossy areas in red
    overlay = result_image.copy()
    overlay[gloss_mask > 0] = [0, 0, 255]  # Red color in BGR
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)

    # Convert masks to RGB before encoding
    slab_mask_rgb = cv2.cvtColor(slab_mask, cv2.COLOR_GRAY2RGB)
    _, slab_mask_encoded = cv2.imencode('.jpg', slab_mask_rgb)
    slab_mask_bytes = slab_mask_encoded.tobytes()
    slab_mask_base64 = base64.b64encode(slab_mask_bytes).decode('utf-8')
    slab_mask_data = f"data:image/jpeg;base64,{slab_mask_base64}"

    gloss_mask_rgb = cv2.cvtColor(gloss_mask, cv2.COLOR_GRAY2RGB)
    _, gloss_mask_encoded = cv2.imencode('.jpg', gloss_mask_rgb)
    gloss_mask_bytes = gloss_mask_encoded.tobytes()
    gloss_mask_base64 = base64.b64encode(gloss_mask_bytes).decode('utf-8')
    gloss_mask_data = f"data:image/jpeg;base64,{gloss_mask_base64}"

    return result_image, glossy_percentage, slab_mask_data, gloss_mask_data

def calculate_sheen_percentage(image):
    global processed_image_3, processed_image_4, lower_gloss_thresh, upper_gloss_thresh, total_sheen_area
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed_image_3 = blurred.copy()  # Store processed image 3

    # Threshold to identify shiny areas
    _, thresh = cv2.threshold(blurred, lower_gloss_thresh, upper_gloss_thresh, cv2.THRESH_BINARY)
    processed_image_4 = thresh.copy()  # Store processed image 4

    # Calculate percentage of sheen
    sheen_pixels = cv2.countNonZero(thresh)
    total_pixels = image.shape[0] * image.shape[1]
    percentage = (total_sheen_area / total_pixels) * 100
    return percentage

@app.route('/processed_image_1_stream')
def processed_image_1_stream():
    def generate():
        if processed_image_1 is not None:
            with image_lock:
                ret, jpeg = cv2.imencode('.jpg', processed_image_1)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image_2_stream')
def processed_image_2_stream():
    def generate():
        if processed_image_2 is not None:
            with image_lock:
                ret, jpeg = cv2.imencode('.jpg', processed_image_2)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image_3_stream')
def processed_image_3_stream():
    def generate():
        if processed_image_3 is not None:
            with image_lock:
                ret, jpeg = cv2.imencode('.jpg', processed_image_3)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image_4_stream')
def processed_image_4_stream():
    def generate():
        if processed_image_4 is not None:
            with image_lock:
                ret, jpeg = cv2.imencode('.jpg', processed_image_4)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start a background thread to capture images continuously
    # parse the command line
    parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=backgroundNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="fcn-resnet18-voc", help="pre-trained model to load, see below for options")
    parser.add_argument("--filter-mode", type=str, default="linear", choices=["point", "linear"], help="filtering mode used during visualization, options are:\n  'point' or 'linear' (default: 'linear')")
    parser.add_argument("--visualize", type=str, default="overlay,mask", help="Visualization options (can be 'overlay' 'mask' 'overlay,mask'")
    parser.add_argument("--ignore-class", type=str, default="void", help="optional name of class to ignore in the visualization results (default: 'void')")
    parser.add_argument("--alpha", type=float, default=150.0, help="alpha blending value to use during overlay, between 0.0 and 255.0 (default: 150.0)")
    parser.add_argument("--stats", action="store_true", help="compute statistics about segmentation mask class output")
    parser.add_argument("--camera", type=str, default="pylon", choices=["pylon", "usb"], help="Camera source to use ('pylon' or 'usb')")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    camera_source = args.camera  # Set the camera source based on the argument

    # load the background removal network
    net = backgroundNet(args.network, sys.argv)
    input = videoSource(args.input, argv=sys.argv)
    output = videoOutput(args.output, argv=sys.argv)

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
