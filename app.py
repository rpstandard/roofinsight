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
#  import torch
import matplotlib.pyplot as plt
# import torchvision

from jetson_inference import segNet
from jetson_utils import videoSource, videoOutput, cudaOverlay, cudaDeviceSynchronize, Log, cudaFromNumpy
from segnet_utils import *
# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

app = Flask(__name__)

# Global variables
latest_image = None
captured_image = None
glossy_percentage = None
slab_mask_image = None  # To hold the slab mask image
gloss_mask_image = None  # To hold the gloss mask image
processed_image_1 = None  # New variable for processed image 1
processed_image_2 = None  # New variable for processed image 2
image_lock = threading.Lock()

# Add a variable to store the camera source
camera_source = None  # Will be set based on user's choice

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda"

# def load_sam_model(checkpoint, model_type):
#     # Load the state dictionary with weights_only=True
#     state_dict = torch.load(checkpoint, weights_only=True)
#     model = sam_model_registry[model_type]()
#     model.load_state_dict(state_dict)
#     return model

# sam = load_sam_model(sam_checkpoint, model_type)
# sam.to(device=device)

# mask_generator = SamAutomaticMaskGenerator(sam)

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
                time.sleep(0.033)  # Approximate 30 FPS
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
    else:
        print("Invalid camera source specified")

def process_image_in_thread():
    global latest_image, processed_image_1, processed_image_2

    while True:
        if latest_image is not None:
            with image_lock:
                # Copy the latest image to avoid threading issues
                image_to_process = latest_image.copy()

            # Process the image using your segmentation network
            # Convert to the format expected by the network
            img_input = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2RGBA)
            img_input_cuda = cudaFromNumpy(img_input)

            # Allocate buffers
            buffers.Alloc(img_input_cuda.shape, img_input_cuda.format)

            # Process the segmentation network
            net.Process(img_input_cuda, ignore_class=args.ignore_class)

            # Generate the overlay
            if buffers.overlay:
                net.Overlay(buffers.overlay, filter_mode=args.filter_mode)

            # Generate the mask
            if buffers.mask:
                net.Mask(buffers.mask, filter_mode=args.filter_mode)

            # Convert CUDA images to NumPy arrays
            if buffers.overlay:
                overlay_image = cudaToNumpy(buffers.overlay)
                overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGBA2BGR)
                with image_lock:
                    processed_image_1 = overlay_image.copy()  # Store processed image 1

            if buffers.mask:
                mask_image = cudaToNumpy(buffers.mask)
                mask_image = cv2.cvtColor(mask_image, cv2.COLOR_RGBA2BGR)
                with image_lock:
                    processed_image_2 = mask_image.copy()  # Store processed image 2

        time.sleep(1)  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html',
                           captured_image=captured_image,
                           glossy_percentage=glossy_percentage,
                           slab_mask_image=slab_mask_image,
                           gloss_mask_image=gloss_mask_image,
                           lower_gloss_thresh=200,
                           upper_gloss_thresh=255)
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

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
    global latest_image, captured_image, glossy_percentage, slab_mask_image, gloss_mask_image

    if request.method == 'POST':
        # Get threshold values from the form
        lower_gloss_thresh = int(request.form.get('lower_gloss_thresh', 200))
        upper_gloss_thresh = int(request.form.get('upper_gloss_thresh', 255))
    else:
        # Default threshold values
        lower_gloss_thresh = 200
        upper_gloss_thresh = 255

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

    return render_template('index.html',
                           original_image=original_image,
                           captured_image=captured_image,
                           glossy_percentage=glossy_percentage,
                           slab_mask_image=slab_mask_image,
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

@app.route('/processed_image_1_stream')
def processed_image_1_stream():
    def generate():
        while True:
            if processed_image_1 is not None:
                with image_lock:
                    ret, jpeg = cv2.imencode('.jpg', processed_image_1)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.1)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/processed_image_2_stream')
def processed_image_2_stream():
    def generate():
        while True:
            if processed_image_2 is not None:
                with image_lock:
                    ret, jpeg = cv2.imencode('.jpg', processed_image_2)
                if ret:
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.1)
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start a background thread to capture images continuously
    # parse the command line
    parser = argparse.ArgumentParser(description="Segment a live camera stream using an semantic segmentation DNN.", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=segNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

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
    
    # load the segmentation network
    net = segNet(args.network, sys.argv)

    # note: to hard-code the paths to load a model, the following API can be used:
    #
    # net = segNet(model="model/fcn_resnet18.onnx", labels="model/labels.txt", colors="model/colors.txt",
    #              input_blob="input_0", output_blob="output_0")

    # set the alpha blending value
    net.SetOverlayAlpha(args.alpha)

    # create video output
    output = videoOutput(args.output, argv=sys.argv)

    # create buffer manager
    buffers = segmentationBuffers(net, args)

    # create video source
    input = videoSource(args.input, argv=sys.argv)

    # Start the image capture thread
    capture_thread = threading.Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    # Start the image processing thread
    process_thread = threading.Thread(target=process_image_in_thread)
    process_thread.daemon = True
    process_thread.start()

    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True)
