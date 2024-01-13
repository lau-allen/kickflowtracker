#import libraries
import cv2 as cv 
import os
import numpy as np
import glob
from tqdm import tqdm 
from time import sleep
import pickle

def camera_calibration(chessboardsize,framesize,img_path):
    #define termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    #init object points
    objp = np.zeros((chessboardsize[0] * chessboardsize[1],3),np.float32)
    objp[:,:2] = np.mgrid[0:chessboardsize[0],0:chessboardsize[1]].T.reshape(-1,2)

    #arrays to store object points and image points for images 
    objPoints = [] #3D 
    imgPoints = [] #2D image plane 

    #images 
    images = glob.glob(img_path)

    #camera calibration 
    for image in tqdm(images):
        sleep(3) 
        img = cv.imread(image)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        #find chess board corners 
        ret, corners = cv.findChessboardCorners(gray, chessboardsize, None)
        #if corners found, add object points and image points 
        if ret == True:
            objPoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1),criteria)
            imgPoints.append(corners)

            #dray and display the corners 
            cv.drawChessboardCorners(img,chessboardsize,corners2,ret)
            cv.waitKey(1000)
    cv.destroyAllWindows()

    #camera calibration 
    ret,cameraMatrix,dist,rvecs,tvecs = cv.calibrateCamera(objPoints,imgPoints,framesize,None,None)

    #return camera calibration
    return ret,cameraMatrix,dist,rvecs,tvecs

#function to write frames to image
def write_frames(video_input,out,fname,frame_skip):
    #write out dir if doesnt exist
    if not os.path.exists(out):
        os.makedirs(out)

    #define CV video capture obj 
    video = cv.VideoCapture(video_input)

    #check if video is unable to be obtained
    if video.isOpened() == False:
        raise Exception('Error opening video stream') 

    count = 0
    #loop through frames
    while True:
        #read current frame 
        ret,frame = video.read()

        if ret:
            if count%frame_skip==0:
                cv.imwrite(os.path.join(out,f'{fname+"_"+str(count)+".png"}'),frame)
        else:
            break
        count+=1
    
    return  0


def detect_circles(frame):
    grayFrame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blurFrame = cv.GaussianBlur(grayFrame,(9,9),0)
    circles = cv.HoughCircles(blurFrame,cv.HOUGH_GRADIENT,1,40,param1=65, param2=30, minRadius=35, maxRadius=85)
    ball_x, ball_y,ball_r = 0,0,0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        circles = circles[0]
        for (x,y,r) in circles:
            if y>500 and x>100:
                ball_x, ball_y,ball_r = x,y,r
                break
    return (ball_x-ball_r,ball_y-ball_r,2*ball_r,2*ball_r)
        




if __name__ == "__main__":
    # input = '/Users/allenlau/Documents/CCNY/Fall23/CSCI6516_ComputerVision/project/KickFlowTracker/data/external/vid5.mov'
    # write_frames(input,'data/external/optFlowTest','cal_IMG',1)

    #camera calibration 
    # chessboardsize = (6,4)
    # framesize = (1080,1920)
    # calib_img_path = './data/external/camera_calibration/*.png'
    # cam_calib_params = camera_calibration(chessboardsize,framesize,calib_img_path)
    # print(cam_calib_params[1])
    # with open('./camera_parameters/camera_params.pkl','wb') as f:
    #     pickle.dump(cam_calib_params,f)

    pass