from flask import Flask, Response, render_template, url_for, request
from pypylon import pylon
import numpy as np
import cv2
import io
from PIL import Image
import base64
import threading

app = Flask(__name__)

# Global variables
latest_image = None
captured_image = None
glossy_percentage = None
slab_mask_image = None  # To hold the slab mask image
gloss_mask_image = None  # To hold the gloss mask image

def capture_image():
    global latest_image
    # Create an instance of the camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Open the camera
    camera.Open()

    # Set the camera to continuous capture mode
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    try:
        while camera.IsGrabbing():
            # Retrieve the grabbed image
            grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                # Convert the image to a numpy array
                latest_image = grab_result.Array

            # Release the result to process the next frame
            grab_result.Release()

    except Exception as e:
        print(f"Error: {e}")
    
    # Stop grabbing and release the camera
    camera.StopGrabbing()
    camera.Close()

@app.route('/')
def index():
    return render_template('index.html',
                           captured_image=captured_image,
                           glossy_percentage=glossy_percentage,
                           slab_mask_image=slab_mask_image,
                           gloss_mask_image=gloss_mask_image,
                           lower_gloss_thresh=200,
                           upper_gloss_thresh=255)

def generate():
    global latest_image

    while True:
        if latest_image is not None:
            # Check if the image is grayscale (single channel)
            if len(latest_image.shape) == 2:  # Grayscale
                # Convert grayscale to RGB by duplicating the single channel
                latest_image = cv2.cvtColor(latest_image, cv2.COLOR_GRAY2BGR)

            # Ensure the image is in the proper format (3 channels)
            if latest_image.shape[2] == 3:  # RGB
                # Convert the image to JPEG format
                ret, jpeg = cv2.imencode('.jpg', latest_image)

                if ret:
                    # Return the image as a byte stream
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
            else:
                print("Warning: Unsupported image format, skipping frame.")

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

        # Convert the processed image to a format that can be displayed in the browser
        _, jpeg = cv2.imencode('.jpg', processed_image)
        img_bytes = jpeg.tobytes()
        captured_image = f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        # Set the intermediate images
        slab_mask_image = slab_mask_data
        gloss_mask_image = gloss_mask_data

    return render_template('index.html',
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
    hsv = cv2.cvtColor(slab_regions, cv2.COLOR_RGB2HSV)

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
    overlay[gloss_mask > 0] = [0, 0, 255]  # Red color
    alpha = 0.5  # Transparency factor
    cv2.addWeighted(overlay, alpha, result_image, 1 - alpha, 0, result_image)

    # Encode intermediate images to display
    _, slab_mask_encoded = cv2.imencode('.jpg', slab_mask)
    slab_mask_bytes = slab_mask_encoded.tobytes()
    slab_mask_base64 = base64.b64encode(slab_mask_bytes).decode('utf-8')
    slab_mask_data = f"data:image/jpeg;base64,{slab_mask_base64}"

    _, gloss_mask_encoded = cv2.imencode('.jpg', gloss_mask)
    gloss_mask_bytes = gloss_mask_encoded.tobytes()
    gloss_mask_base64 = base64.b64encode(gloss_mask_bytes).decode('utf-8')
    gloss_mask_data = f"data:image/jpeg;base64,{gloss_mask_base64}"

    return result_image, glossy_percentage, slab_mask_data, gloss_mask_data

if __name__ == '__main__':
    # Start a background thread to capture images continuously
    capture_thread = threading.Thread(target=capture_image)
    capture_thread.daemon = True
    capture_thread.start()

    # Start the Flask web server
    app.run(host='0.0.0.0', port=5000, threaded=True)
