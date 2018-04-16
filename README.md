# DPAAS
Deep Learning based Automated Attendance System

### Face Detection
#### Algorithm used: Single Shot Multibox Detector

Images captured via the Secondary Capture Device will be stored inside the media folder inside tensorflow-face-detection folder. Please make a folder there with the name media.

The algorithm is trained on WIDER dataset.

### Face Recognition
#### Algorithm used: VGG 16

The algorithm is trained on a custom dataset made to ensure that face detection works in the Indian Context. The Face Detection code takes in the photos cropped by the Face Detector network (reads from the media folder inside the tensorflow-face-detection folder)
