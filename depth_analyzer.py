 #!/usr/bin/env python3

import cv2
import depthai as dai
import matplotlib.pyplot as plt
from collections import deque
import pygame
import time


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
        wav_file_1_path = r'C:\Users\paolo\Desktop\PROJECT\beep1.wav'
        #wav_file_2_path = r'C:\Users\paolo\Desktop\PROJECT\beep2.wav'
        #wav_file_3_path = r'C:\Users\paolo\Desktop\PROJECT\beep3.wav'

        # Load three different sound files
        self.sound_1 = pygame.mixer.Sound(wav_file_1_path)
        #self.sound_2 = pygame.mixer.Sound(wav_file_2_path)
        #self.sound_3 = pygame.mixer.Sound(wav_file_3_path)
        self.playing_sound = False

    def plot(self, dataPlot):
        self.lineplot.set_data(dataPlot.axis_x, dataPlot.axis_y)

        self.axes.set_xlim(min(dataPlot.axis_x), max(dataPlot.axis_x))
        ymin = 0
        ymax = max(dataPlot.axis_y) + 10
        self.axes.set_ylim(ymin, ymax)
        self.axes.relim();

    def play_sound(self, sound):
        sound.play()
        


# Start defining a pipeline
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

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")

    color = (0, 255, 0)

    print("Use WASD keys to move ROI!")

    fig, axes = plt.subplots()
    plt.title('Plotting Data')

    data = DataPlot()
    dataPlotting = RealtimePlot(axes)

    count = 0
    newConfig = False
    stepSize = 0.01 
    

    if newConfig:
        config.roi = dai.Rect(topLeft, bottomRight)
        cfg = dai.SpatialLocationCalculatorConfig()
        cfg.addROI(config)
        spatialCalcConfigInQueue.send(cfg)
        newConfig = False 
    
    while True:
        count += 1
        inDepth = depthQueue.get() # Blocking call, will wait until a new data has arrived
        inDepthAvg = spatialCalcQueue.get() # Blocking call, will wait until a new data has arrived

        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        
        spatialData = inDepthAvg.getSpatialLocations()
        print("spatial data len", len(spatialData))

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymax + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymax + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymax + 50), fontType, 0.5, color)

            # Check if the distance (Z) is less than 2000 mm
            if 1000 < int(depthData.spatialCoordinates.z) <= 2000:
                print("oggetto_trovato_posizione_lontana")
                
                # Play the sound
                dataPlotting.play_sound(dataPlotting.sound_1)
                time.sleep(0.7)

            # Check if the distance (Z) is less than 1000 mm
            elif 500 < int(depthData.spatialCoordinates.z) <= 1000:
                print("oggetto_trovato_posizione_intermedia")
                
                # Play the sound
                dataPlotting.play_sound(dataPlotting.sound_1)
                time.sleep(0.3)

            # Check if the distance (Z) is less than 500 mm
            elif 0 < int(depthData.spatialCoordinates.z) <= 500:
                print("oggetto_trovato_posizione_vicina")
                
                # Play the sound
                dataPlotting.play_sound(dataPlotting.sound_1)
                

            data.add(count, int(depthData.spatialCoordinates.z))
            dataPlotting.plot(data)

            plt.pause(0.5)


        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        elif key == ord('e'):
            topLeft.x += 0.01
            topLeft.y += 0.01
            bottomRight.x -= 0.01
            bottomRight.y -= 0.01
            newConfig = True
        elif key == ord('r'):
            topLeft.x -= 0.01
            topLeft.y -= 0.01
            bottomRight.x += 0.01
            bottomRight.y += 0.01
            newConfig = True
        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
