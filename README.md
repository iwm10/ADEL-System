# ADEL-System

<div align="center">
<img src="ADEL logo" alt="Alt Text" width="300"/>
</div>













## Smart Light Traffic System:
This topic aims To Count the density of each light traffic vehicle and manage timers of each light traffic according to how high and low the density of vehicles inside each light traffic with CCTV camera, and detect any vehicle that violated lane line rules and send them messages with their information and fee. 

### Key Feature:
- **Vehicles  Detection & tracking:** Identify each vehicle inside lanes with id by tracker and detect the type of the car.
- **Violated vehicles & license plate :** Detect violated vehicle with it's license plate.
- **PaddelOCR:** extract information from violated vehicle license plate.
- **Sahi:** innovative library designed to optimize object detection algorithms for large-scale and high-resolution imagery.



### Technology:
- **Computer vision:** opencv, yolov8, pytourch, sahi
- **Tracker:** DeepSort tracker
- **OCR:** PaddelOCR
- **Deployment:** streamlit,sumo simulator
- **Enviroment:** hugging face
- **Database:** sqlite3
- **Email server:** stmp

## Detection Violation Lane:
We trained our model by using yolov8s and divided it into two label license plates, vehicles where the license plate reaches 68% mAP, and vehicles reach 92% mAP.
<div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/Data_set/Data_set1" alt="Data set" width="500"/>
</div>



By addressing our logic on Python to specific lanes rule by each lane and using deepsort tracker to follow car movement through the video  to detect the violated vehicle and extracting the images of the license plate of the violated vehicle to extract the result by paddelOCR accuracy 85%.

<div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/Data_set/Result" alt="Result" width="500"/>
</div>

## Density Model:
We train the data to know if this vehicle has a small weight or not because some cars like FJ detect it as a truck so this is not will help the model to know the accelerations of the vehicle
<div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Density/Dataset/before Sahi.jpeg" alt="Result" width="500"/>
</div>

After that, we also used Sahi to help detect the long destance viechles and this is the result

<div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Density/Dataset/after sahi and train.jpeg" alt="Result" width="500"/>
</div>

### Installation:
1. **Clone the repository:**
    ```bash
   git clone https://github.com/iwm10/ADEL-System   
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
### Usage:
1.**Navegate to the project directory:**
    

    git clone https://github.com/iwm10/ADEL-System
2.**Run the following command to access the Streamlit app:**
   
    
       streamlit run app.py
**or**
1. **Access to the app:**
   ```bash
   https://huggingface.co/spaces/M12ths/ADEL_app
-----------------------------------
2. **Download video for deployment:**
   ```bash
   https://youtu.be/NUNaUCcmvRw
3. **Upload video through the app:**
   
    After you download the video within main page of app click "Browse files" and upload the video

   <div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/Interface%20ADEL_APP.jpg" alt="ADEL APP Interface" width="500"/>
</div>

4. **Result:**
The result will show as output video of processes and images of detected violating vehicles and then by pressing on send email it'll send message to emails attached with violated vehicles

<div align="center">
    <a href="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/DEMO_DETECTION_VIOLATION.mp4" style="display: inline-block; width: 500px;">
        <img src="./path-to-image-thumbnail.png" alt="Watch the video" width="500" />
    </a>
</div>

## Sumo Deployment:
this is simulator for Density model and how it's work

https://github.com/user-attachments/assets/c024e313-4bb6-4843-94ef-af3d80187d73


 
## 👥 Group Members Information

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Mohammed%20Alrowais-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mohammed-alrowais-565563277/)
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Mohammed%20Al%20Malki-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/mohammed-almalki-m2000/)
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Khaled%20AlGhamdi-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/khaled-alghamdi-b33718a5/)
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Waleed%20AlIkhwan-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/waleed-al-ikhwan-a9725a139/)

