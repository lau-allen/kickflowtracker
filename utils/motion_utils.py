import numpy as np 
import cv2 as cv 

#function to return the magnitude 
def perform_optical_flow(prev_img,img):
    #computes the dense optical flow using Farneback method 
    flow = cv.calcOpticalFlowFarneback(prev_img,img,None,0.5, 3, 10, 3, 5, 1.1, 0)

    #computes the magnitude and angle for the 2D vectors
    mag,angle = cv.cartToPolar(flow[...,0],flow[...,1])

    return mag, angle

#function to return colored image for visualizing optical flow
def visualize_optical_flow(mag,angle,shape):
    #define empty mask 
    mask = np.zeros(shape,dtype=np.uint8)
    #set saturation of mask to max
    mask[...,1] = 255
    #sets image hue based on optical flow direction 
    mask[...,0] = angle * 180 / np.pi / 2
    #sets image value based on optical flow magnitude 
    mask[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
    #convert hsv to rgb
    return cv.cvtColor(mask, cv.COLOR_HSV2BGR)

#function to calculate the real-world velocity in x and y of the ball from optical flow 
def get_vector_velocity(mag,angle,sx,sy,fx,fy,x,y,r,fps):
    #define bounding box 
    left = y
    right = y+(r) 
    top = x
    bottom = x+(r)
    #define shape
    shape = mag.shape 
    #calculate the magnitude of sx and sy 
    s = np.sqrt(sx**2 + sy**2)
    #checking if we are within bounds 
    m_per_s,avg_dir = 0,0
    # if left >= 0 and right <= shape[1] and top >= 0 and bottom <= shape[0]:
    #define window for mag and angle 
    window_mag = mag[left:right,top:bottom]
    window_angle = angle[left:right,top:bottom]
    #mean to get pixels/frame 
    mag_pixels_per_frame = np.mean(window_mag)
    m_per_frame = mag_pixels_per_frame * np.sqrt(0.001905**2+0.001746**2)
    # #meters/frame 
    # m_per_frame = mag_pixels_per_frame * s * (1/np.sqrt(fx**2+fy**2))
    #meters/sec
    m_per_s = m_per_frame * fps
    #average direction 
    avg_dir = np.mean(window_angle)
        
    return m_per_s,avg_dir
    

if __name__ == '__main__':
    #defining iamges 
    path_imgt1 = './data/external/optFlowTest/cal_IMG_28.png'
    path_imgt2 = './data/external/optFlowTest/cal_IMG_29.png'
    #reading images 
    imgt1 = cv.imread(path_imgt1)
    imgt2 = cv.imread(path_imgt2)
    #shape 
    shape = imgt1.shape
    #grayscale 
    imgt1 = cv.cvtColor(imgt1,cv.COLOR_BGR2GRAY)
    imgt2 = cv.cvtColor(imgt2,cv.COLOR_BGR2GRAY)
    #perform optical flow
    mag,angle = perform_optical_flow(imgt1,imgt2)
    print(angle)
    #visualizing optical flow 
    opt_flow_viz = visualize_optical_flow(mag,angle,shape)
    #display 
    cv.imshow('Optical Flow',opt_flow_viz)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
    pass