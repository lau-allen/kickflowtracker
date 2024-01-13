#import libraries
import cv2 as cv 
import utils.motion_utils as mu 
import utils.camera_utils as cu
import numpy as np
import pickle as pkl



#entry into kickflow tracker
def main(video_input):

    tracker = cv.TrackerCSRT_create()
    BB = None
    
    with open('camera_parameters/camera_params.pkl','rb') as f:
        data = pkl.load(f)
        f.close()
    f = 24
    fx = data[1][0][0]
    fy = data[1][1][1]
    sx = f/(fx)
    sy = f/(fy)

    def track(frame):
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        return success, frame, box
    #define CV video capture obj 
    video = cv.VideoCapture(video_input)
    #get fps of video feed 
    fps = int(video.get(cv.CAP_PROP_FPS))

    #check if video is unable to be obtained
    if video.isOpened() == False:
        raise Exception('Error opening video stream') 
    
    #get first frame 
    ret,first_frame = video.read()
    #define shape 
    shape = first_frame.shape 
    #init prev_img 
    prev_img = cv.cvtColor(first_frame,cv.COLOR_BGR2GRAY)

    #ball tracking 
    success = False
    while True:
        #read current frame 
        ret,frame = video.read()
        #if successfully reading frame
        if ret: 
            #convert cur frame to grayscale 
            gray_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
            key = cv.waitKey(1) & 0xFF

            if BB is not None:
                    success, frame, box = track(frame) # Track object
            
            if key == ord("c"): # Select Region of Interest (ROI) to track
                # cv.imwrite('tracking_frame.png',frame)
                # BB = cv.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
                BB = cu.detect_circles(frame)
                tracker.init(frame, BB)

            # x,y,r = cu.detect_circles(frame)
            #optical flow 
            mag,angle = mu.perform_optical_flow(prev_img,gray_frame)
            #optical flow visualization
            opt_flow_viz = mu.visualize_optical_flow(mag,angle,shape)
            data = {}
            # print(success)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                avg_vel,avg_angle = mu.get_vector_velocity(mag,angle,sx,sy,fx,fy,x,y,w,60)
                print(avg_vel)
            #display 
            concat_frame = np.concatenate((frame, opt_flow_viz), axis=1)
            # cv.rectangle(concat_frame, (x-r,y-r), (x+r, y+r), (0,255,0), 2)
            cv.imshow('KickFlow Tracker',concat_frame)
            #quit loop if 'q' pressed
            
            if key == ord("q"):
                break
            
            #update prev_img
            prev_img = gray_frame
        
        #quit if frame not returned 
        else:
            raise Exception('Frame not returned')
    
    #stop and release video capture object 
    video.release()
    #close all windows
    cv.destroyAllWindows()
    
    return  

if __name__ == "__main__":
    video_input = 'data/external/media/IMG_4152.MOV'
    main(video_input)