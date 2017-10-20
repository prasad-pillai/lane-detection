# **Finding Lane Lines on the Road** 

---

**Goals**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Introduction

This is a project to find lane lines on the road with python and opencv. The agenda is to first develop a pipeline which will annotate the lane lines on an image and later extend the technique to find lane line on video streams. This project explores many opencv functions to do image processing togather with numpy to accomplish the task of lane detection.

### The Concept

Lane detection is a complex process. As in most computer vision problems the first thing we can do is take a detailed look at the image and find some way to differenciate the thing of intrest in that image from the background. In our case our item of interest is the lane lines in the road. 

*** Put image ***

If we look into the sample image we can clearly identify that the lane lines are usually white of yellow. So the step one would be as follows

1. Select yellow and white color regions from the image

The second thing we find about the lane lines in our sample images is that the lane lines have a good deal of contrast difference with the background which is the road. This means that we can do edge detection in these images to find the boundaries of lane lines. Thus the step two will be as follows.

2. Do edge detection to get the edges of lane lines

Another thing to observe is that lane lines are not everywere in the image they are in the bottom half towards the middle. This means that we can shorten our search within a trapiziod within the image. Thus the step 3 will be:

3. Identify a region on interest and discard other regions.

With that step we are left with identifying lines in the edge detected image. This can be done by Hough transform. Thus the forth step will be to find straight lines in the edge detected image using hough transform.

4. Find lines in edge detected image.

If you have ever worked with hough line detection, then you will know that multiple lines gets detected. We just want one line on both sides. So we need to group and average the lines we get out of hough line detection. To identify / group lane lines we can use the fact that slope of lines will be greater than or less than zero depending on there position on the road. 

5. Group and average lines besed on slope

Finaly we can extend this pipeline for a video stream.

![alt text][image1]

### Steps

#### 1. Color selection

First 

### Shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


### Possible improvements to the pipeline

This pipeline detects straight lines and expects the lane to be straight. One improvement maight be to detect curves instead of lines and annotate the lanes as curves itself.
