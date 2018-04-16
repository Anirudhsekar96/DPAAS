import cv2
import os
dir_path = './i_video/'
outout_path = "/media/anirudh/Data/Code/Dpaas/KERAS_FACE_RECOGNITION/i_video_new/"
for video_path in sorted(os.listdir(dir_path)):
	#os.mkdir(outout_path+video_path)
	count = 0
	for vdeo in os.listdir(dir_path+video_path):
		vidcap = cv2.VideoCapture(dir_path+video_path+"/"+vdeo)
		success,image = vidcap.read()
		success = True
		while success:
			cv2.imwrite(outout_path+video_path+"/"+"frame%d.jpg" % (count), image)     # save frame as JPEG file
			success,image = vidcap.read()
			print('Read a new frame: ', success)
			count += 1