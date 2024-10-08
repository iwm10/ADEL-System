import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
import numpy as np
import time
from inference_sdk import InferenceHTTPClient
import base64
import uuid
from moviepy.editor import VideoFileClip
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import traceback

import imageio
from paddleocr import PaddleOCR, draw_ocr
from matplotlib import pyplot as plt
import random
import sqlite3
import pandas as pd

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage



os.makedirs("detected_license_plates",exist_ok=True)
os.makedirs("Plates_Text",exist_ok=True)
os.makedirs("violating_vehicles",exist_ok=True)
os.makedirs('uploads',exist_ok=True)
os.makedirs('output_videos', exist_ok=True)

model = YOLO('MyEDVLmodel.pt')
tracker = DeepSort(max_age=30, nn_budget=70, nms_max_overlap=1.0)

def convert_mov_to_mp4(input_path, output_path):
    # Load the .mov file
    video_clip = VideoFileClip(input_path)
    
    # Write the video file to .mp4 format with audio, using a faster preset
    video_clip.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        ffmpeg_params=['-preset', 'fast', '-crf', '23', '-threads', '4']
    )




def draw_perspective_lanes_as_rois(frame):
    img_height, img_width = frame.shape[:2]



    # Define points for each lane as parallelograms (ROIs)
    pts_right_lane = np.array([[2600, 950], [6800, img_height], [3500, img_height], [2200, 950]], np.int32)

    pts_middle_lane = np.array([[2199, 950], [3500, img_height], [1200, img_height], [1902, 950]], np.int32)

    pts_left_lane = np.array([[1920, 950], [1150, img_height], [-1880, img_height], [1650, 950]], np.int32)

    # Draw lane boundaries (outlines) as ROIs with thinner lines
    color_right_lane = (0, 255, 0)    # Right lane (green)
    color_middle_lane = (0, 0, 255)   # Middle lane (red)
    color_left_lane = (255, 0, 0)     # Left lane (blue)

    thickness = 3

    # Draw lines to outline each lane
    cv2.polylines(frame, [pts_right_lane], isClosed=True, color=color_right_lane, thickness=thickness)#Ritght lane
    cv2.polylines(frame, [pts_middle_lane], isClosed=True, color=color_middle_lane, thickness=thickness)#Middel lane
    cv2.polylines(frame, [pts_left_lane], isClosed=True, color=color_left_lane, thickness=thickness)#left lane

    return frame



# Function to classify lane based on vehicle's center position and lane boundaries
def classify_lane(vehicle_x_center, lane_boundaries):
    """
    Classifies the vehicle based on the center of its bounding box.
    """
    # If vehicle is outside the leftmost boundary (e.g., exiting the road), it's 'left_outside'
    if vehicle_x_center < lane_boundaries['left_boundary']:
        return 'left_outside'  # Vehicle outside the left boundary
    elif vehicle_x_center < lane_boundaries['middle']:
        return 'left'
    elif lane_boundaries['middle'] <= vehicle_x_center < lane_boundaries['right']:
        return 'middle'
    else:
        return 'right'

    
    

    
def check_lane_violation(prev_lane, current_lane):
    """
    Check if a lane violation occurred.
    """
    # No violation if the vehicle is outside the left lane boundary
    if current_lane == 'left_outside':
        return False  # Vehicle is outside, no violation
    
    # If the vehicle changes lanes and is not entering 'left_outside', flag a violation
    if prev_lane != current_lane:
        return True  # Lane violation detected
    
    return False  # No violation, vehicle stays in the same lane




# Process the video and detect violations and license plates
def Process_video(model, tracker, input_path, output_video_path, vehicle_image_dir, license_plate_dir):
    try:
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


        # Draw perspective lanes
        #frame = draw_perspective_lanes_as_rois(frame)

        vehicle_directions = {}
        saved_violations = set()
        violating_cars_count = 0
        last_violation_time = {}

        lane_boundaries = {
            'left_boundary': 100,
            'middle': width // 3,
            'right': 2 * (width // 3)
        }

        # Create directories for saving violating vehicles and license plates if they don't exist
        if not os.path.exists(vehicle_image_dir):
            os.makedirs(vehicle_image_dir)
        if not os.path.exists(license_plate_dir):
            os.makedirs(license_plate_dir)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            img_height, img_width = frame.shape[:2]

            # Draw perspective lanes
            frame = draw_perspective_lanes_as_rois(frame)

            # Run the model inference on the current frame
            results = model(frame)  # Assuming you have an existing model for detection
            detections = []
            plate_boxes = {}

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])

                    if class_id == 0:  # License Plate
                        plate_boxes[(x1, y1, x2, y2)] = confidence
                    elif class_id == 1:  # Vehicle
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append((bbox, confidence, class_id))

            # Update tracks for vehicles
            tracks = tracker.update_tracks(detections, frame=frame)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                x1, y1, x2, y2 = map(int, track.to_tlbr())
                track_id = track.track_id

                # Calculate the center of the bounding box
                vehicle_x_center = (x1 + x2) // 2

                # Classify the lane based on the center of the bounding box
                current_lane = classify_lane(vehicle_x_center, lane_boundaries)

                if track_id in vehicle_directions:
                    prev_lane = vehicle_directions[track_id]
                    if check_lane_violation(prev_lane, current_lane):
                        # Extend the violation bounding box duration
                        last_violation_time[track_id] = time.time()

                        if track_id not in saved_violations:
                            # This is a new violation
                            violating_cars_count += 1
                            print(f"Violation detected for vehicle {track_id} from {prev_lane} to {current_lane}")

                            # Save the vehicle image
                            vehicle_img = frame[y1:y2, x1:x2]
                            vehicle_image_path = os.path.join(vehicle_image_dir, f"vehicle_{track_id}.jpg")
                            cv2.imwrite(vehicle_image_path, vehicle_img)
                            saved_violations.add(track_id)
                            print(f"Saved image of violating vehicle {track_id} at {vehicle_image_path}")

                            # Check for license plate detection within the violating vehicle
                            for plate_box in plate_boxes.keys():
                                px1, py1, px2, py2 = plate_box
                                if px1 > x1 and py1 > y1 and px2 < x2 and py2 < y2:
                                    # Save license plate image
                                    plate_img = frame[py1:py2, px1:px2]
                                    plate_image_path = os.path.join(license_plate_dir, f"plate_{track_id}.jpg")
                                    cv2.imwrite(plate_image_path, plate_img)
                                    print(f"Saved license plate image for vehicle {track_id} at {plate_image_path}")
                                    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 0), 5)  # Thicker bounding box for license plate

                        # Thicker red bounding box for violation
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)  # Red for violation
                    else:
                        # Thicker green bounding box for the vehicle with no violation
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)  # Green for no violation

                vehicle_directions[track_id] = current_lane

                # Make sure the bounding box stays with the vehicle during the process
                cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Keep violation bounding boxes visible for a fixed duration
            current_time = time.time()
            for track in tracks:
                if track.track_id in last_violation_time:
                    violation_time = last_violation_time[track.track_id]
                    if current_time - violation_time < 3:  # Keep bounding box for 3 seconds after violation
                        # Draw red bounding box for 3 seconds after violation detection
                        x1, y1, x2, y2 = map(int, track.to_tlbr())
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)

            # Write the processed frame to the output video
            out.write(frame)

        cap.release()
        out.release()

        print(f"Processing complete. Total number of violating cars: {violating_cars_count}")
        print(f"Video saved at: {output_video_path}")
        print(f"Violating vehicle images saved in: {vehicle_image_dir}")
        print(f"License plate images saved in: {license_plate_dir}")
        return violating_cars_count

    except Exception as e:
        print("An error occurred during video processing:", str(e))
        traceback.print_exc()


        return None, None




def enhance_license_plate(image):
    resized_image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    return resized_image


# Ensure output directories exist
def create_output_dir(base_dir, obj_id):
    dir_path = os.path.join(base_dir, f"Object_{obj_id}")
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def calculate_iou(boxA, boxB):
    # Calculate Intersection over Union (IOU) between two bounding boxes
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def assign_ids(predictions, tracked_objects, iou_threshold=0.3):
    new_tracked_objects = []
    for pred in predictions:
        # Create bounding box for current detection
        pred_box = [int(pred['x'] - pred['width'] / 2),
                    int(pred['y'] - pred['height'] / 2),
                    int(pred['x'] + pred['width'] / 2),
                    int(pred['y'] + pred['height'] / 2)]

        # Find if it matches with a previously tracked object
        assigned = False
        for obj in tracked_objects:
            obj_box = obj['bbox']
            iou = calculate_iou(pred_box, obj_box)

            if iou > iou_threshold:
                new_tracked_objects.append({
                    'id': obj['id'],
                    'bbox': pred_box,
                    'confidence': pred['confidence']
                })
                assigned = True
                break

        if not assigned:
            new_id = len(tracked_objects) + len(new_tracked_objects) + 1
            new_tracked_objects.append({
                'id': new_id,
                'bbox': pred_box,
                'confidence': pred['confidence']
            })

    return new_tracked_objects

def resize_image(image, width=640):
    return cv2.resize(image, (width, int(image.shape[0] * width / image.shape[1])))
    
def save_cropped_object(frame, bbox, obj_id, output_base_dir):
    # Crop the object based on the bounding box
    x1, y1, x2, y2 = bbox
    cropped_image = frame[y1:y2, x1:x2]

    # Create output directory for the object
    # output_dir = create_output_dir(output_base_dir, obj_id)

    # Save the cropped image in the corresponding directory
    cropped_image_path = os.path.join(output_base_dir, f"detected_licenseplate_" + str(uuid.uuid4())+ ".jpg")
    cv2.imwrite(cropped_image_path, cropped_image)

    print(f"Cropped image saved: {cropped_image_path}")
    
    

###################################################################################################################


def read_image(image_path):
    # Check the image format based on the file extension
    _, ext = os.path.splitext(image_path)
    ext = ext.lower()

    # If the image is in HEIC format
    if ext in ['.heic', '.heif']:
        image = imageio.imread(image_path)
    else:
        # For PNG, JPG, JPEG formats
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (common image format)

    return image




def extract_text_from_license_plate(image_path, ocr, output_dir):

    result = ocr.ocr(image_path, cls=True)
    image = read_image(image_path)
    print(image.shape)
    
    # Process each detected text block
    formatted_text = []
    
    if result[0] == None:
        return
    
    if len(result[0]) > 2:
        result[0] = result[0][1:]

    for line in result:
        for text_info in line:
            text, confidence = text_info[1][0], text_info[1][1]
            # formatted_text.append(f"Text: {text}, Confidence: {confidence:.2f}")
            formatted_text.append(f"{text}")
            

    # Save the formatted text to a file
    text_filename = str(os.path.join(output_dir, os.path.split(image_path)[1])).replace('jpg','txt')
    with open(text_filename, "w") as text_file:
        text_file.write("\n".join(formatted_text))





###################################################################################################################









def Connect_Database():
    # Database file path
    db_file = 'license_plate_fines.db'

    # Directories for images and extracted text
    image_dir = 'detected_license_plates'
    text_dir = 'Plates_Text'

    # Fine amount
    fine_ticket = 350

    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            extracted_text TEXT,
            fine_ticket INTEGER,
            paid BOOLEAN DEFAULT 0
            )
    ''')

    # Function to insert data into the database
    def insert_fine(image_path, extracted_text, fine_ticket):
        cursor.execute('''
            INSERT INTO fines (image_path, extracted_text, fine_ticket)
            VALUES (?, ?, ?)
        ''', (image_path, extracted_text, fine_ticket))
        conn.commit()

    # Function to get text from Plates_Text directory (assuming .txt files)
    def get_extracted_text(file_name):
        text_file_path = os.path.join(text_dir, file_name)
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r') as file:
                return file.read().strip()
        return None

    # Process all images and corresponding extracted text
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        
        # Extracted text filename (assuming .txt files corresponding to images)
        text_file = image_file.replace('.jpg', '.txt')
        extracted_text = get_extracted_text(text_file)
        
        # Insert data into the database
        if extracted_text:
            insert_fine(image_path, extracted_text, fine_ticket)
        else:
            print(f"No extracted text found for {image_file}")

    # Closing the connection to the database
    conn.close()
    
        
        # Connect to the SQLite database
    conn = sqlite3.connect('license_plate_fines.db')

    # Create a cursor object
    cursor = conn.cursor()

    # Create the 'email_records' table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS email_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            email TEXT NOT NULL
        )
    ''')
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print("Table 'email_records' created successfully.")
    
    
    def add_email_record(image_path, email):
        # Connect to the SQLite database
        conn = sqlite3.connect('license_plate_fines.db')
        cursor = conn.cursor()

        # Insert a new record into the email_records table
        cursor.execute('''
            INSERT INTO email_records (image_path, email) VALUES (?, ?)
        ''', (image_path, email))

        # Commit the changes and close the connection
        conn.commit()
        conn.close()
        print(f"Record added: Image Path: {image_path}, Email: {email}")
        
        
    ####################################################################
        
    emails=['tabt660@outlook.sa',
    'malrowais2001@gmail.com',
    'm7md200158@gmail.com']

    plates= os.listdir('detected_license_plates')


    # for email, plate in zip(emails, plates):
    #     add_email_record(plate, email)

    for plate in plates:
        add_email_record(plate, random.choice(emails))    




    
    
def Display_Database1():
    # Connect to the SQLite database
    conn = sqlite3.connect('license_plate_fines.db')

    # Read the data from the 'fines' table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM fines", conn)

    # Close the database connection
    conn.close()

    # Display the DataFrame in the Streamlit frontend
    st.dataframe(df)


def Display_Database2():
    # Connect to the SQLite database
    conn = sqlite3.connect('license_plate_fines.db')

    # Read the data from the 'fines' table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM email_records", conn)

    # Close the database connection
    conn.close()

    st.dataframe(df)




##########################################################################################################################







################################## Sending Emails Notifications ##########################################################


def Send_Emails():
    """Sending Emails"""

    # Email configuration
    Sender_email = "m7md200158@gmail.com"  # Replace with your email address
            
    def send_email(sender_email, recipient_email, subject, message):
        sender_password = "jxtu rndu zjud hxtr"  # Use the App Password (16 digits)

        # Setup the MIME
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject

        # Attach the message
        msg.attach(MIMEText(message, 'plain'))

        # Create an SSL session for sending the mail
        try:
            # Use Gmail's SMTP server with SSL
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
            server.login(sender_email, sender_password)  # Login with the App Password
            text = msg.as_string()  # Convert the message to string format
            server.sendmail(sender_email, recipient_email, text)  # Send the email
            print("Email sent successfully!")
        except Exception as e:
            print(f"Failed to send email. Error: {str(e)}")
        finally:
            server.quit()  # Terminate the SMTP session

    def get_fines_from_db():
        # Connect to the SQLite database
        conn = sqlite3.connect('license_plate_fines.db')
        cursor = conn.cursor()

        # Retrieve email records and associated fines
        cursor.execute('''
            SELECT e.email, l.image_path, l.extracted_text, l.fine_ticket
            FROM email_records e
            JOIN fines l ON e.image_path = l.image_path
        ''')

        records = cursor.fetchall()  # Fetch all records
        conn.close()
        return records

    # Get fine details and email addresses
    fines = get_fines_from_db()

    for email, image_path, extracted_text, fine_amount in fines:
        subject = "Fine Notification"
        body = f"عزيزي السائق,\n\n" \
               f"لقد تم قيد مخالفة تجاوز مسارات.\n\n" \
               f"الوصف:\n" \
               f"-  {image_path}:الصورة\n" \
               f"- اللوحة: {extracted_text}\n" \
               f"-ٍ {fine_amount} قيمة المخالفة:ريال\n\n" \
               f"يرجاء سداد المبلغ في اقرب وقت ممكن.\n\n" \
               f"وشكرا."

        try:
            send_email(sender_email=Sender_email, recipient_email=email, subject=subject, message=body)
            print(f"Email sent to: {email}")
        except Exception as e:
            print(f"Failed to send email to {email}: {e}")




# Custom CSS for styling
st.markdown("""
    <style>
    .css-1d391kg {
        background-color: #ffffff !important;
    }
    .stApp {
        background-color: #ffffff;
    }
    .header {
        font-size: 50px;
        color: #003366;
        text-align: center;
        font-family: 'Arial', sans-serif;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .upload-section {
        padding: 20px;
        background-color: #ffffff;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    .video-container {
        text-align: center;
        margin-top: 20px;
    }
    .logo-container {
        text-align: center;
        margin-top: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Uncomment to set max upload size (in MB)
# st.set_option('server.maxUploadSize', 1024)

st.image('labeled_output_image.jpg', width=650)

# Header Section
st.markdown("<div class='header'>Number of Violated Lane Lines Vehicles</div>", unsafe_allow_html=True)

# Main Upload Section for video files
st.markdown("<div class='upload-section'><h3 style='color: black;'>Upload Your Video File</h3></div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Handling video files
    st.markdown("<div class='video-container'><h4 style='color: black;'>Uploaded Video:</h4></div>", unsafe_allow_html=True)
    
    # Save the uploaded video to the 'uploads' directory
    video_save_path = os.path.join("uploads", uploaded_file.name)
    with open(video_save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Video saved at: {video_save_path}")

    # Add selector option for output display
    display_option = st.selectbox("Choose what to display:", ("Uploaded Video", "Simulation"))

    if display_option == "Uploaded Video":
        # Display the uploaded video
        st.video(video_save_path)  
    elif display_option == "Simulation":
        # Here, you would add your simulation display logic
        st.markdown("<h4 style='color: black;'>Simulation Placeholder</h4>", unsafe_allow_html=True)
        # You could include a placeholder for the simulation or any relevant information
        st.write("This section would display the simulation output based on the uploaded video.")

    # Process the video
    vehicle_image_dir = 'violating_vehicles'
    license_plate_dir = 'detected_license_plates'
    output_video_path = os.path.join("output_videos", "output_" + uploaded_file.name)
    violating_vehicles =  0
    
    # Uncomment to process the video with your model
    violating_vehicles = Process_video(model, tracker, video_save_path, output_video_path, vehicle_image_dir, license_plate_dir)

    st.success(f"Processed video saved at: {output_video_path}")
    
    convert_mov_to_mp4(output_video_path,'output_video.mp4')
    st.video('output_video.mp4')
    
    # Display results
    st.markdown(f"<h3 style='color: black;'>Violating vehicles: <strong>{violating_vehicles}</strong></h3>", unsafe_allow_html=True)

    # Display images of violating vehicles
    if os.path.exists(vehicle_image_dir):
        image_files = [f for f in os.listdir(vehicle_image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            st.markdown("<h4 style='color: black;'>Images of Violating Vehicles:</h4>", unsafe_allow_html=True)
            for image_file in image_files:
                image_path = os.path.join(vehicle_image_dir, image_file)
                st.image(image_path, caption=image_file, use_column_width=True)
        else:
            st.markdown("<h4 style='color: black;'>No images found!</h4>", unsafe_allow_html=True)



    ocr = PaddleOCR(use_angle_cls=False, lang='en')

    dir_path = 'detected_license_plates'
    output_dir = 'Plates_Text'

    for image in os.listdir(dir_path):
        image_path = os.path.join(dir_path, image)
        extract_text_from_license_plate(image_path=image_path, ocr=ocr, output_dir=output_dir)

    Connect_Database()
    
    #st.markdown("<h3 style='color: black;'>Displaying the Database Records</h3>", unsafe_allow_html=True)
    #Display_Database1()
    #Display_Database2()

    # Button to send emails
    if st.button("Send Emails"):
        Send_Emails()
        st.markdown("<h4 style='color: black;'>Emails Sent!</h4>", unsafe_allow_html=True)
        
    print('Emails Sent!')

else:
    st.info("Please upload a video file.")






######################################################################################################################################



