# **Finding Lane Lines on the Road** 

---

**Goals**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

[original]: ./all_images.jpg "Test images"
[blur]: ./blur.jpg "Blured images"
[color_select]: ./color_select.jpg "Color select images"
[edge]: ./edge.jpg "Edge detected images"
[hough]: ./hough.jpg "Hough lines images"
[hsl_select]: ./hsl_select.jpg "HSL color select images"
[lane_drawn]: ./lane_drawn.jpg "Lane drawn images"
[overlay]: ./overlay.jpg "Overlay images"
[rgb_select]: ./rgb_select.jpg "RGB color select images"
[roi]: ./roi.jpg "ROI images"


---

### Introduction

This is a project to find lane lines on the road with python and opencv. The agenda is to first develop a pipeline which will annotate the lane lines on an image and later extend the technique to find lane line on video streams. This project explores many opencv functions to do image processing together with numpy to accomplish the task of lane detection.

### The Concept

Lane detection is a complex process. As in most computer vision problems the first thing we can do is take a detailed look at the image and find some way to differentiate the thing of interest in that image from the background. In our case our item of interest is the lane lines on the road. 

![alt text][original]

If we look into the sample image above we can clearly identify that the lane lines are usually white or yellow in color. So the step one would be as follows.

  > __ Select yellow and white color regions from the image __

The second thing we find about the lane lines in our sample images is that the lane lines have a good deal of contrast difference with the background which is the road. This means that we can do edge detection in these images to find the boundaries of lane lines. Thus the step two will be as follows.

  > __ Do edge detection to get the edges of lane lines __

Another thing to observe is that lane lines are not everywhere in the image they are in the bottom half towards the middle. This means that we can shorten our search within a trapezoid within the image. Thus the step 3 will be:

  > __ Identify a region on interest and discard other regions __

With that step we are left with identifying lines in the edge detected image. This can be done by Hough lane detection algorithm. Thus the forth step will be to find straight lines in the edge detected image using hough transform.

  > __Find lines in edge detected image__

If you have ever worked with hough line detection, then you will know that multiple lines gets detected. We just want one line on right and one on left. So we need to group and average the lines we get out of hough line detection. To identify / group lane lines we can use the fact that slope of lines will be greater than or less than zero depending on there position on the road. 

  > __Group and average lines besed on slope__

Finaly we can extend this pipeline for a video stream.

#### Summary

1. Select yellow and white color regions from the image
2. Do edge detection to get the edges of lane lines
3. Identify a region on interest and discard other regions
4. Find lines in edge detected image
5. Group and average lines based on slope

---

### Steps

#### 1. Color selection

##### RGB color space
We know that lane lines are white or yellow in color based on this information we are trying to extract regions with lane lines from the images. The images are in RGB color space and a particular range of colors can be extracted from the image using opencv's `cv2.inRange()` function. We pass in the image and two sets of RGB values and the function outputs the regions in that particular color range.

Reference: [RGB Color Code Chart](http://www.rapidtables.com/web/color/RGB_Color.htm)

```python
def rgb_color_select(image):
    """
    Seperate white and yellow color regions from
    the given image based on RGB values
    """
    # creating white mask
    lower_range = np.uint8([180,180,180])
    upper_range = np.uint8([255,255,255])
    white_image = cv2.inRange(image, lower_range, upper_range)
    
    #creating yellow mask
    lower_range = np.uint8([190, 190, 0])
    upper_range = np.uint8([255,255,255])
    yellow_image = cv2.inRange(image, lower_range, upper_range)
    
    #combine both the masks
    mask = cv2.bitwise_or(white_image, yellow_image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image
```

![alt_text][rgb_select]

This worked fine, but not for all images. The images where there are shadows, the operation did not yield good results.

##### HSL Color space
Now we can experiment with color selection operation in other color spaces. HSL is Hue Lightness Saturation color space.
Here also we can use the `cv2.inRange()` function.

```python
def hsl_color_select(image):
    """
    Seperate white and yellow color regions from
    the given image based on HSL values
    """
    img = hsl(image)
    # creating white mask
    lower_range = np.uint8([  0, 200,   0])
    upper_range = np.uint8([255,255,255])
    white_image = cv2.inRange(img, lower_range, upper_range)
    
    #creating yellow mask
    lower_range = np.uint8([10, 0, 100])
    upper_range = np.uint8([40, 255, 255])
    yellow_image = cv2.inRange(img, lower_range, upper_range)
    
    #combine both the masks
    mask = cv2.bitwise_or(white_image, yellow_image)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image
```
![alt_text][hsl_select]

This worked much better for yellow regions under the shades of trees. 

##### Combined color space

From our above result its evident that color selection in RGB space was good at identifying white regions and that in HSL space is good at identifying yellow regions. So i combined the power of both and created a single function.

```python
def color_select(image):
    # creating white mask
    lower_range = np.uint8([180,180,180])
    upper_range = np.uint8([255,255,255])
    white_image = cv2.inRange(image, lower_range, upper_range)
    
    #creating yellow mask
    lower_range = np.uint8([190, 190, 0])
    upper_range = np.uint8([255,255,255])
    yellow_image = cv2.inRange(image, lower_range, upper_range)
    
    #combine both the masks
    rgb_mask = cv2.bitwise_or(white_image, yellow_image)
    
    hsl_img = hsl(image)
    # creating white mask
    lower_range = np.uint8([  0, 200,   0])
    upper_range = np.uint8([255,255,255])
    white_image = cv2.inRange(hsl_img, lower_range, upper_range)
    
    #creating yellow mask
    lower_range = np.uint8([10, 0, 100])
    upper_range = np.uint8([40, 255, 255])
    yellow_image = cv2.inRange(hsl_img, lower_range, upper_range)
    
    #combine both the masks
    hsl_mask = cv2.bitwise_or(white_image, yellow_image)
    
    mask = cv2.bitwise_or(rgb_mask, hsl_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image
```
![alt_text][color_select]

This seems to produce very good results.

#### 2. Convert to grayscale
After color selection we are going forward with edge detection and other operations. These operations does not require color information and just a grayscale image is enough. So we convert our images into grayscale.

```python
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

#### 3. Apply blur
If we do edge detection on images with rough edge we are most likely to get a noisy image. So to get clear edges we apply a Gaussian blur to the images. We use opencv's `cv2.GaussianBlur()` function for this. This function takes an image and a kernel size. Kernal size is a possitive odd number. The greater the kernal size the more blurred the image gets. You can try different kernel sizes and find which works best for you. I am using kernel size 15. Refer the following links for more details.

- [Gaussian Filter Theory](http://docs.opencv.org/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html#gaussian-filter)
- [cv2.GaussianBlur OpenCV API Reference](http://docs.opencv.org/modules/imgproc/doc/filtering.html?highlight=gaussianblur#gaussianblur)

```python
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
```

![alt_text][blur]

This is how the images look after converting to grayscale and applying gaussian blur.

#### 4. Edge Detection
To find the edges we use canny edge detection algorithm which typically employs gradients in X and Y direction and then thins out the identified edge to a single pixel width. OpenCv has an inbuild function to do canny edge detection. This function takes a grayscale image, a low threshold and a high threshold. If a pixel gradient is higher than the upper threshold, the pixel is accepted as an edge and if it is less than the lower threshold, it is rejected. Refer the links below for more information.

- [Canny Edge Detection Theory](http://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)
- [cv2.Canny OpenCV API Reference](http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html)

```python
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
```

![alt_text][edge]
This is how an edge detected images looks like.

#### 5. Selecting Region of Interest
As we have found the edges, the next step is to find the lines. But we don’t need to do the search for lines everywhere in the image we can see that the lane lines are in the middle bottom region. So we can constrain our search to that region alone. We can use opencv's `cv2.fillPoly()` function to do this. I have identified a trapezoidal shape where i want to restrict the search.

- [cv2.fillPoly OpenCV API Reference](http://docs.opencv.org/modules/core/doc/drawing_functions.html#fillpoly)

```python
def select_region(image):
    """
    Selects a region of interest from the given image
    """
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.94]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.94]
    top_right    = [cols*0.6, rows*0.6] 
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return region_of_interest(image, vertices)
```
![alt_img][roi]

#### 6. Line Detection
After selecting the region of interest we are ready to find lines on the image. We use opencv's `cv2.HoughLinesP()` function to do that. This function has a few parameters.
- rho: Distance resolution of the accumulator in pixels.
- theta: Angle resolution of the accumulator in radians.
- threshold: Accumulator threshold parameter. Only those lines are returned that get enough votes (> `threshold`).
- minLineLength: Minimum line length. Line segments shorter than that are rejected.
- maxLineGap: Maximum allowed gap between points on the same line to link them.

This algorithm maps the result of canny edge detection to a space called hough space. In this space points will becomes lines and intersection of lines means that there is a straight line. Many lines get identified in the image. I use a pythons random function to generate random colors for each line. For more info refer

- [Hough Line Transform Theory](http://docs.opencv.org/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html)
- [cv.HoughLinesP OpenCV API Reference](http://docs.opencv.org/modules/imgproc/doc/feature_detection.html?highlight=houghlinesp#houghlinesp)

```python
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    if lines is None:
        print('no lines detected')
        print(img.shape)
        plt.imshow(img)
        plt.show()
        return None
    img_black = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(img_black, lines)
    return img_black, lines
```

![alt_img][hough]

Now i combined original image with the lines image.

![alt_img][overlay]

#### 6. Grouping, averaging and extrapolating lines
Hough line detection gives you a set of lines. We need a single line of the left as well as on the right. We do this by the fact that the lines on the left will have a negative slope and those on the right have positive slope. We group lines by their slope and finally average them so get a single line on the left as well as on the right.

```python
def get_lanes(hough_lines):
    """
    Given hough lines this function will identify lines on the left and right,
    group and average them to output single line on left and right
    """
    left_lines = []
    right_lines = []
    
    for line in hough_lines:
        slope = get_slope(line)
        if slope > 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    line_right = np.int32(np.average(right_lines, axis=0))
    line_left = np.int32(np.average(left_lines, axis=0))
    return line_right, line_left
    ```
    
This function will output a left and a right lane line. Now we need to extrapolate these lines to extend upto the top of the RIO from the bottom.

```python
def get_line_points(line, y1, y2):
    """
    Given line and top and bottm points, this function extents that line 
    to the y1, y2 and returns this longer line
    """
    if line is None:
        return None
    
    slope, length, intercept = get_slope_length_and_intercept(line)
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    
    return [x1, y1, x2, y2]
```

The above function takes a line, upper and lower y values and extrapolates the given line to those values. Thus we get a continous line in that limits.

![alt_img][lane_drawn]

** Thus we have succesfully identified lane lines on images. **

### Extending to videos

As we have successfully tested our pipeline on all our test images, lets extent our program to do the same with video streams. Video streams are just images shown at an interval called frame rate. Usually the frame rate is 24 frame per second. Which means at every second of the video there will be 24 images. We use Moviepy library to edit our test videos. `process_image()` in the following code is out pipeline function which does all the handwork for us. This function gets called every frame while processing.

Note: The problem with simply running our pipeline every frame is that the lane lines drawn looks flickering. To avoid this flickering i have added an averaging operation with a queue implementation. I have defined a queue size of 62. We add every new lane line detected to the queue and average it. The line shown is actually the average of all those lines. This brings a stabilization to our detection process.

```python
    from collections import deque

    QUEUE_SIZE=62

    class LaneProcessor:
    def __init__(self):
        self.left_lines  = deque(maxlen=QUEUE_SIZE)
        self.right_lines = deque(maxlen=QUEUE_SIZE)

    def process_image(self, image):
        rgb_masked = color_select(image)
        gray = grayscale(rgb_masked)
        blured = gaussian_blur(gray, kernel_size=15)
        edge = canny(blured, low_threshold=50, high_threshold=150)
        roi = select_region(edge)
        line_img, lines = hough_lines(roi, 1, np.pi/180, 20, 20, 300)
        
        if lines is None:
            return image
        line_right, line_left = get_lanes(lines)

        def mean_lines(line, lines):
#             print(line)
            if line is not None:
                lines.append(line)

            if len(lines)>0:
                line = np.mean(lines, axis=0, dtype=np.int32)
            return line

        line_left  = mean_lines(line_left,  self.left_lines)
        line_right = mean_lines(line_right, self.right_lines)
        
        blank_image = np.zeros_like(image)
        
        draw_line(blank_image, line_left)
        draw_line(blank_image, line_right)
        i = weighted_img(blank_image, image)

        return i
```
The outputs are as follows

- [White Lanes Video](./test_videos_output/solidWhiteRight.mp4)
- [Yellow Lanes Video](./test_videos_output/solidYellowLeft.mp4)
- [Dark Shades Video](./test_videos_output/challenge.mp4)

### Shortcomings with your current pipeline


It doesn’t work on sudden bends in the road as the region of interest is selected from the bottom middle.
This pipeline does not take into consideration the curved nature of lane lines atleast in some parts of the road.
Huge lighting variations might cause errors in detection.


### Possible improvements to the pipeline

This pipeline detects straight lines and expects the lane to be straight. One improvement might be to detect curves instead of lines and annotate the lanes as curves itself.

