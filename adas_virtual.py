import cv2
import numpy as np
# A code with Short explanation....!!!!
def canny(img):  # This defines a function named canny that takes an input image img as its parameter. 
    #The function will perform Canny edge detection on this image.
    if img is None: #This conditional statement checks if the input image img is None, indicating that no image was passed to the function. 
        #If this condition is true, it executes the following block of code.
        
        #If the image is None, it assumes that the video capture (cap) needs to be released. This line releases the video capture resource.
        cap.release()
        # This closes all OpenCV windows that might be open.
        cv2.destroyAllWindows()
       # This exits the program.
        exit()

        #This converts the input image (img) from the BGR (Blue-Green-Red) color space to grayscale using the 
        #cv2.cvtColor() function. Grayscale images are easier to process and require less computational resources.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #This defines the size of the Gaussian kernel for blurring.
    kernel = 5
    #This applies Gaussian blur to the grayscale image (gray) using the specified kernel size (kernel).
    # Gaussian blur helps in reducing noise and unwanted details in the image, making edge detection more effective.
    blur = cv2.GaussianBlur(gray,(kernel, kernel),0)
    #This performs Canny edge detection on the blurred grayscale image (gray).
    # The parameters 50 and 150 are the lower and upper thresholds for the hysteresis procedure in the Canny edge detector. 
    #These values determine which edges are considered as strong, weak, or non-edges.
    canny = cv2.Canny(gray, 50, 150)
    #This returns the resulting Canny edge-detected image.
    return canny

def region_of_interest(canny): # function defination region_of_interst
    #This line calculates the height of the input image (canny) by accessing its shape and retrieving the number of rows (height).
    height = canny.shape[0]
    #This line calculates the width of the input image (canny) by accessing its shape and retrieving the number of columns (width).
    width = canny.shape[1]
    #This line creates a blank mask image with the same dimensions as the input image (canny). It initializes all pixels to zero.
    mask = np.zeros_like(canny)
    #This line defines a triangular region of interest in the image.
    # It creates an array (triangle) containing three vertices of the triangle: (200, height), (800, 350), and (1200, height).
    triangle = np.array([[
    (200, height),
    (800, 350),
    (1200, height),]], np.int32)
    #This line fills the triangular region defined by the triangle array with white color (255) on the blank mask image (mask).
    # It effectively masks everything outside this triangular region.
    cv2.fillPoly(mask, triangle, 255)
    #This line performs a bitwise AND operation between the Canny edge-detected image (canny) and the mask image (mask).
    #It retains only the edges within the triangular region of interest.
    masked_image = cv2.bitwise_and(canny, mask)
    #This line returns the masked image containing edges only within the defined region of interest.
    return masked_image

def houghLines(cropped_canny):
    #This line utilizes the Hough Line Transform (cv2.HoughLinesP()) to detect lines in the cropped Canny edge-detected image (cropped_canny).
    #It returns the detected lines.
    return cv2.HoughLinesP(cropped_canny, 2, np.pi/180, 100, 
        np.array([]), minLineLength=40, maxLineGap=5)

    ##This line overlays the detected lines (line_image) onto the original frame (frame) using weighted addition (cv2.addWeighted()).
    #It returns the resulting composite image.
def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)
 
def display_lines(img,lines):
    #This line creates a black image (line_image) with the same dimensions as the input image (img).
    line_image = np.zeros_like(img)
    #This conditional statement checks if any lines were detected (lines is not None).
    if lines is not None:
        #If lines were detected, this loop iterates over each detected line and draws it on the black image (line_image) using cv2.line(). 
        # The lines are drawn in red color (BGR: 0, 0, 255) with a thickness of 10 pixels.
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(0,0,255),10)
    #This line returns the image containing the drawn lines.            
    return line_image
 
 #This line defines a function named make_points that takes two arguments: image and line.
def make_points(image, line):
    #This line unpacks the line tuple into slope and intercept variables. The line tuple presumably contains the slope and y-intercept of a detected line.
    slope, intercept = line
    #This line calculates the y-coordinate of the bottom of the image (assuming the origin is at the top-left corner) and stores it in y1.
    y1 = int(image.shape[0])
    #This line calculates the y-coordinate at a higher position in the image (3/5 of the image height) and stores it in y2.
    # This likely represents a point higher up in the image, where the lane lines typically converge.
    y2 = int(y1*3.0/5)      
    #This line calculates the x-coordinate corresponding to y1 on the detected line using the slope-intercept formula (y = mx + b).
    x1 = int((y1 - intercept)/slope)
    #This line calculates the x-coordinate corresponding to y2 on the detected line using the same slope-intercept formula.
    x2 = int((y2 - intercept)/slope)
    #This line returns a list containing the coordinates of two points (x1, y1) and (x2, y2). These points define a line segment in the image.
    return [[x1, y1, x2, y2]]
 
 #This line defines a function named average_slope_intercept that takes two arguments: image and lines.
def average_slope_intercept(image, lines):
    #These lines initialize empty lists left_fit and right_fit to store slope-intercept pairs for left and right lane lines, respectively.
    left_fit    = []
    right_fit   = []
    #This conditional statement checks if no lines were detected (lines is None). If so, it returns None, indicating no lane lines were found.
    if lines is None:
        return None
    #These lines iterate over each detected line in lines and unpack the coordinates of the line's endpoints into variables x1, y1, x2, and y2.
    for line in lines:
        for x1, y1, x2, y2 in line:
            #These lines fit a first-degree (linear) polynomial to the endpoints of the line to calculate the slope and y-intercept of the line.
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            #These lines classify the detected line as belonging to the left lane line (if the slope is negative) or the right lane line (if the slope is positive or zero).
            # The slope-intercept pairs are then added to the corresponding lists.
            if slope < 0: 
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    # These lines calculate the average slope and y-intercept for the left and right lane lines, respectively, by averaging the values stored in left_fit and right_fit lists.            
    left_fit_average  = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    #These lines call the make_points function to generate line segments (represented by coordinate pairs) based on the average slope-intercept values for the left and right lane lines.
    left_line  = make_points(image, left_fit_average)
    right_line = make_points(image, right_fit_average)
    #This line creates a list containing the left and right lane line segments.
    averaged_lines = [left_line, right_line]
    #This line returns the list of averaged lane line segments.
    return averaged_lines

#This line initializes a video capture object (cap) by opening the video file named "lane_detect.mp4" using cv2.VideoCapture() function.
cap = cv2.VideoCapture("lane_detect.mp4")
#This line starts a while loop that iterates as long as the video capture object (cap) is open and valid.
while(cap.isOpened()):
    #This line reads a frame (frame) from the video capture object (cap). The _ is used to discard the return value, and frame contains the read frame.
    _, frame = cap.read()
    #This line calls the canny function to perform Canny edge detection on the current frame (frame), resulting in a Canny edge-detected image (canny_image).
    canny_image = canny(frame)
    #This line calls the region_of_interest function to crop the Canny edge-detected image (canny_image) to a region of interest, likely containing the lane area only.
    #The result is stored in cropped_canny.
    cropped_canny = region_of_interest(canny_image)

    #This line detects lines in the cropped Canny edge-detected image (cropped_canny) using the Hough Line Transform, which is implemented in the houghLines function.
    #The detected lines are stored in lines.
    lines = houghLines(cropped_canny)
    #This line calculates the average slope and intercept of the detected lines using the average_slope_intercept
    #function, based on the original frame (frame) and the detected lines (lines). The result is stored in averaged_lines.
    averaged_lines = average_slope_intercept(frame, lines)
    #This line draws the averaged lines (averaged_lines) onto a black image (line_image) using the display_lines function. The black image has the same dimensions as the original frame (frame).
    line_image = display_lines(frame, averaged_lines)
    #This line overlays the image with the detected lines (line_image) onto the original frame (frame) using the addWeighted function. The result is stored in combo_image.
    combo_image = addWeighted(frame, line_image)
    #This line displays the resulting image (combo_image) in a window titled "result" using cv2.imshow() function.
    cv2.imshow("result", combo_image)
    #This line waits for a key press for 1 millisecond using cv2.waitKey(1). If the pressed key is 'q' (ord('q')), the loop breaks, effectively terminating the program.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#This line releases the video capture object (cap), closing the video file.
cap.release()
#This line closes all OpenCV windows, effectively terminating the program and cleaning up resources.
cv2.destroyAllWindows()


"""""""""
# This Commented code is used for Measuring Image graph plotting and for the line detecting, before the mp4 format detect.
# This code can be try before the mp4 format tested, and it should be the same image from the video that gonna to test, because this functions
# def region_of_interest(image or canny): is base on the size of the image, so it's importand to test before mp4 uploade...
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_img(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    detect = cv2.Canny(blur, 50, 150)
    return detect
def display_line(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height ), (1100, height), (550, 250)]
         ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread("img2.jpg")
lane_image = np.copy(image)
detect = detect(lane_image)
cropped_image = region_of_interest(detect)
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
average_lines = average_slope(lane_image, lines)
line_image = display_line(lane_image, average_lines)
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
#plt.imshow(detect)
#plt.show()
cv2.imshow('result',combo_image)
cv2.waitKey(0)
"""""""""