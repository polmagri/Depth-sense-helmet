#!/usr/bin/env python3

"""
The code is the same as for Tiny Yolo V3 and V4, the only difference is the blob file
- Tiny YOLOv3: https://github.com/david8862/keras-YOLOv3-model-set
- Tiny YOLOv4: https://github.com/TNTWEN/OpenVINO-YOLOV4
"""
from pathlib import Path
import cv2
import depthai as dai
from networkx import is_empty
import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque
import pygame
import time
from pathlib import Path
import sys
import time

# Get argument first
nnPath = str((Path('C:/Users/paolo/Desktop/Depth sense helmet/depth-sense helmet/models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
if 1 < len(sys.argv):
    arg = sys.argv[1]
    if arg == "yolo3":
        nnPath = str((Path('C:/Users/paolo/Desktop/Depth sense helmet/depth-sense helmet/models/yolo-v3-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    elif arg == "yolo4":
        nnPath = str((Path('C:/Users/paolo/Desktop/Depth sense helmet/depth-sense helmet/models/yolo-v4-tiny-tf_openvino_2021.4_6shave.blob')).resolve().absolute())
    else:
        nnPath = arg
else:
    print("Using Tiny YoloV4 model. If you wish to use Tiny YOLOv3, call 'tiny_yolo.py yolo3'")

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# tiny yolo v4 label texts
labelMap = [
    "person",         "bicycle",    "car",           "motorbike",     "aeroplane",   "bus",           "train",
    "truck",          "boat",       "traffic light", "fire hydrant",  "stop sign",   "parking meter", "bench",
    "bird",           "cat",        "dog",           "horse",         "sheep",       "cow",           "elephant",
    "bear",           "zebra",      "giraffe",       "backpack",      "umbrella",    "handbag",       "tie",
    "suitcase",       "frisbee",    "skis",          "snowboard",     "sports ball", "kite",          "baseball bat",
    "baseball glove", "skateboard", "surfboard",     "tennis racket", "bottle",      "wine glass",    "cup",
    "fork",           "knife",      "spoon",         "bowl",          "banana",      "apple",         "sandwich",
    "orange",         "broccoli",   "carrot",        "hot dog",       "pizza",       "donut",         "cake",
    "chair",          "sofa",       "pottedplant",   "bed",           "diningtable", "toilet",        "tvmonitor",
    "laptop",         "mouse",      "remote",        "keyboard",      "cell phone",  "microwave",     "oven",
    "toaster",        "sink",       "refrigerator",  "book",          "clock",       "vase",          "scissors",
    "teddy bear",     "hair drier", "toothbrush"
]

class DataPlot:
    def __init__(self, max_entries=20):
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)

        self.max_entries = max_entries

        self.buf1 = deque(maxlen=5)
        self.buf2 = deque(maxlen=5)

    def add(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)


class RealtimePlot:
    def __init__(self, axes):
        self.axes = axes

        self.lineplot, = axes.plot([], [], "ro-")

        # Initialize pygame for audio playback
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.set_volume(1.0)  # Imposta il volume a 1.0 (massimo)

        # Define paths to your WAV files
        wav_file_1_path = r'C:\Users\paolo\Desktop\Depth sense helmet\depth-sense helmet\beep1.wav'
        wav_file_2_path = r'C:\Users\paolo\Desktop\Depth sense helmet\depth-sense helmet\beep_DX.wav'
        wav_file_3_path = r'C:\Users\paolo\Desktop\Depth sense helmet\depth-sense helmet\beep_SX.wav'

        # Load three different sound files
        self.sound_1 = pygame.mixer.Sound(wav_file_1_path)
        self.sound_2 = pygame.mixer.Sound(wav_file_2_path)
        self.sound_3 = pygame.mixer.Sound(wav_file_3_path)
        self.playing_sound = False

    def plot(self, dataPlot):
        self.lineplot.set_data(dataPlot.axis_x, dataPlot.axis_y)

        self.axes.set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = 0
        ymax = max(dataPlot.axis_y) + 10
        self.axes.set_ylim(ymin, ymax)
        self.axes.relim()

    def play_sound(self, sound):
        sound.play()
        
def xconvert(x):
    xf=0.25+x*0.5/416
    return xf

def yconvert(y):
    yf=y/416
    return yf

syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

outputDepth = True
outputRectified = False
lrcheck = False
subpixel = False

# StereoDepth
stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

topLeft = dai.Point2f(0.4, 0.4)
bottomRight = dai.Point2f(0.6, 0.6)

spatialLocationCalculator.setWaitForConfigInput(False)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)
# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(416, 416)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(80)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (0, 255, 0)

    fig, axes = plt.subplots()
    plt.title('Plotting Data')

    data = DataPlot()
    dataPlotting = RealtimePlot(axes)
    newConfig = False
    stepSize = 0.01 

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        # Cecking if there is a close object
        emercency=0        
        topLeft = dai.Point2f(0.4, 0.4)
        bottomRight = dai.Point2f(0.6, 0.6)
        config.roi = dai.Rect(topLeft, bottomRight)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        spatialCalcConfigInQueue.send(cfg)
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        inDepthAvg = spatialCalcQueue.get() # Blocking call, will wait until a new data has arrived
        depthFrame = inDepth.getFrame()
        spatialData = inDepthAvg.getSpatialLocations()

        for depthData in spatialData:
            emergency=depthData.spatialCoordinates.z

        # Acquiring with RGB camera
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
        detections=[]
        if (emergency>=700):

            detections = []

            if inDet is not None:
                detections = inDet.detections
                if len(detections) > 0:
                    inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived
                    inDepthAvg = spatialCalcQueue.get()  # Blocking call, will wait until a new data has arrived
                    depthFrame = inDepth.getFrame()
                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
                    counter += 1
                    spatialData = inDepthAvg.getSpatialLocations()
                    min_distance = 10000
                    min_roi = []
                    spatialData_m = []

                    for detection in detections:
                        x1, y1, x2, y2 = map(int, frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)))

                        x1=float(xconvert(x1))
                        x2=float(xconvert(x2))
                        y1=float(yconvert(y1))
                        y2=float(yconvert(y2))

                        topLeft = dai.Point2f(x1, y1)
                        bottomRight = dai.Point2f(x2, y2)
                        config.roi = dai.Rect(topLeft, bottomRight)
                        cfg = dai.SpatialLocationCalculatorConfig()
                        cfg.addROI(config)
                        spatialCalcConfigInQueue.send(cfg)
                        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
                        inDepthAvg = spatialCalcQueue.get() # Blocking call, will wait until a new data has arrived

                        depthFrame = inDepth.getFrame()

                        spatialData = inDepthAvg.getSpatialLocations()
                        print("spatial data len", len(spatialData))
                        for depthData in spatialData:
                            # Assuming that you want to find the minimum distance among all spatial coordinates
                            distance = depthData.spatialCoordinates.z

                            if distance < min_distance:
                                min_distance = distance
                                min_roi=depthData.config.roi
                                spatialData_m = inDepthAvg.getSpatialLocations()
                                print(min_roi)

                    # Print of bounding box for depth
                    for depthData in spatialData_m:
                        roi = depthData.config.roi
                        xc=(roi.topLeft().x+roi.bottomRight().x)/2
                        roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
                        xmin = int(roi.topLeft().x)
                        ymin = int(roi.topLeft().y)
                        xmax = int(roi.bottomRight().x)
                        ymax = int(roi.bottomRight().y)
                        print(xc)
                        fontType = cv2.FONT_HERSHEY_TRIPLEX
                        cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymax + 20), fontType, 0.5, color)
                        cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymax + 35), fontType, 0.5, color)
                        cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymax + 50), fontType, 0.5, color)
                        data.add(counter, int(depthData.spatialCoordinates.z))
                    
                        dataPlotting.plot(data)

                                # Check if the distance (Z) is less than 2000 mm
                        if 2000 < int(depthData.spatialCoordinates.z) <= 3000:
                            print("oggetto_trovato_posizione_lontana")
                            # Play the sound
                            if (xc<0.35):
                                dataPlotting.play_sound(dataPlotting.sound_3)
                            elif (xc>0.65):
                                dataPlotting.play_sound(dataPlotting.sound_2)
                            else:
                                dataPlotting.play_sound(dataPlotting.sound_1)
                            time.sleep(1)

                        # Check if the distance (Z) is less than 1000 mm
                        elif 1000 < int(depthData.spatialCoordinates.z) <= 2000:
                            # Play the sound
                            if (xc<0.35):
                                dataPlotting.play_sound(dataPlotting.sound_3)
                            elif (xc>0.65):
                                dataPlotting.play_sound(dataPlotting.sound_2)
                            else:
                                dataPlotting.play_sound(dataPlotting.sound_1)
                            time.sleep(0.5)

                        # Check if the distance (Z) is less than 500 mm
                        elif 0 < int(depthData.spatialCoordinates.z) <= 1000:
                            print("oggetto_trovato_posizione_vicina")
                            # Play the sound
                            if (xc<0.35):
                                dataPlotting.play_sound(dataPlotting.sound_3)
                            elif (xc>0.65):
                                dataPlotting.play_sound(dataPlotting.sound_2)
                            else:
                                dataPlotting.play_sound(dataPlotting.sound_1)

                    plt.pause(0.5)              

                else:
                    inDepth = depthQueue.get()  # Blocking call, will wait until a new data has arrived
                    inDepthAvg = spatialCalcQueue.get()
                    depthFrame = inDepth.getFrame()
                    spatialData = inDepthAvg.getSpatialLocations()
                    depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
                    depthFrameColor = cv2.equalizeHist(depthFrameColor)
                    depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            if frame is not None:
                displayFrame("rgb", frame)

            cv2.imshow("depth", depthFrameColor)        

            if cv2.waitKey(1) == ord('q'):
                break
        else :
            # Updating distance plot
            counter+=1
            data.add(counter, int(emergency))
            dataPlotting.plot(data)


            # Showing cameras
            depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
            depthFrameColor = cv2.equalizeHist(depthFrameColor)
            depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

            dataPlotting.play_sound(dataPlotting.sound_1)
            plt.pause(0.5) 
            if frame is not None:
                displayFrame("rgb", frame)

            cv2.imshow("depth", depthFrameColor)


