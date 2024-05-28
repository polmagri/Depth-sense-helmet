# Depth-sense-helmet

This Python program demonstrates real-time object detection using Tiny YOLO (You Only Look Once) models (versions 3 and 4) with the DepthAI library.
The primary difference between the two models lies in the blob file used.


Ensure you have the necessary dependencies by running:

-command for installing depthai-python here: https://github.com/luxonis/depthai-python

-python install_requirements.py

the Tiny YOLOv3 model and Tiny YOLOv4 model are contained inside examples>model.

Modify the program to use your own model by changing the nnPath variable to the path of your desired blob file.

In the first part of the code there is the initialisation of the stereocamera for the depth perception and of the RGB camera for the object detection.

In the second part there is the proper programm:

-creation of the pipeline

-while loop

Inside the while loop the object detection algorithm finds all the object inside the FOV of the camera. Then it is computed the distance to each of them, and it is saved the bounding box of the closed one.
In the end, based on the position and on the distance different sounds at different time intervals are given.
There is also an emergency functionality that checks the distance of the center of the camera and gives an emercy sound in case thare is an object at a distance<700mm that hasn't been detected by the object detection algorithm. 
