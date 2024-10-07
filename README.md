# ADEL-System
















## Smart Light Traffic System:
This topic aims To Count the density of each light traffic vehicle and manage timers of each light traffic according to how high and low the density of vehicles inside each light traffic, and detect any vehicle that violated lane line rules and send them messages with their information and fee. 

### Key Feature:
- **Vehicles  Detection:** Identify each vehicle inside lanes with id and detect the type of the car.
- **Violated vehicles & license plate :** Detect violated vehicle with it's license plate.
- **PaddelOCR:** extract information from violated vehicle license plate.
- **Sahi:** innovative library designed to optimize object detection algorithms for large-scale and high-resolution imagery.



### Technology:
- **Computer vision:** opencv, yolov8, pytourch, sahi
- **OCR:** PaddelOCR
- **Deployment:** streamlit
- **Enviroment:** hugging face
- **Database:** sqlite3
- **Email server:** istmp

## Detection Violation Lane:
We trained our model by using yolov8s and divided it into two label license plates, vehicles where the license plate reaches 68% mAP, and vehicles reach 92% mAP.
<div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/Interface%20ADEL_APP.jpg" alt="ADEL APP Interface" width="500"/>
</div>



By addressing our logic on Python to specific lanes rule by each lane to detect the violated vehicle and extracting the images of license plate of the violated vehicle to extract the result by paddelOCR accuracy 85%.

<div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/Interface%20ADEL_APP.jpg" alt="ADEL APP Interface" width="500"/>
</div>

### Installation:
1. **Clone the repository:**
    ```bash
       git clone https://github.com/iwm10/ADEL-System   
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
### Usage:
1. **Navigate to the project directory:**
   ```bash
   https://huggingface.co/spaces/M12ths/ADEL_app
2. **Download video for deployment:**
   ```bash
   https://youtu.be/NUNaUCcmvRw
3. **Upload video through the app:**
   
    After you download the video within main page of app click "Browse files" and upload the video

   <div align="center">
    <img src="https://github.com/iwm10/ADEL-System/blob/main/Detection-Violation-System/Interface%20ADEL_APP.jpg" alt="ADEL APP Interface" width="500"/>
</div>

 4. **Result:**
    the result will show as output video of processes and images of detected violating vehicles and then by pressing on send email it'll send message to emails attached with violated vehicles

    
