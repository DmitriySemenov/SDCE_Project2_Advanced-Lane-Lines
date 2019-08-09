## Advanced Lane Finding Project Writeup (Dmitriy Semenov)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/calibration1.jpg "Distorted Image"
[image1]: ./output_images/undistorted/calibration1.jpg "Undistorted Image"
[image2]: ./test_images/1.1.jpg "Distored Test Image"
[image3]: ./output_images/undistorted/1.1.jpg "Undistorted Test Image"
[image4]: ./output_images/thresholded/scaled_1.1.jpg "Thresholded Test Image"
[image5]: ./output_images/warped.png "Warped Test Image"

[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/1966/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

### Image Extraction

First step that I've decided to take before beginning work on the pipeline was to extract challenging images from optional videos.
I wrote a function `extract_frames()` that uses a clip extracted with moviepy library to save image frams captured at times `times`.  

```python
def extract_frames(clip, times, imgdir):
    for t in times:
        imgpath = os.path.join(imgdir, '{}.jpg'.format(t))
        clip.save_frame(imgpath, t)
```

The full code for this step is contained in the [first code cell](AdvancedLaneLineDetection.ipynb#ChallengeExtraction) of the IPython notebook AdvancedLaneLineDetection.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the [second code cell](AdvancedLaneLineDetection.ipynb#CameraCal) of the IPython notebook AdvancedLaneLineDetection.

First, I start by creating a directory (if it doesn't exist) where undistorted images will be saved.
I've used the `os.mkdir()` function to do that.

Second, I start by creating an `objp` variable, which contains coordinates of chessboard corners in the real world.
Third, I created `objpoints` array, which will be the (x, y, z) coordinates of the chessboard corners in the world, and variable `imgpoints`, which contains the locations of the same corners in the image space coordinate system.

Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then, for each image (converted to grayscale) in the "camera_cal" folder I call `cv2.findChessboardCorners(gray, (9,6),None)`.
This function returns the chessboard corners locations in the image space, if they are found.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration matrix `mtx` and distortion coefficients `dst` using the `cv2.calibrateCamera()` function.  

I applied this distortion correction to the test image using the `cv2.undistort()` function: 

Distorted Image            |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image0] | ![alt text][image1]

It can be seen that the camera calibration is successful as the output image looks correct.

### Pipeline (test images)

#### 1. Provide an example of a distortion-corrected image.

After the calibration matrix `mtx` and distortion coefficients `dst` were determined in the previous step, distortion correction was applied to one of the sample images to test it.

The complete code for this step can be found in the [third code cell](AdvancedLaneLineDetection.ipynb#DistCorrTest)  of the notebook.

The effect of this correction is subtle, but still noticeable if you pay attention to the hood of the vehicle 
Distorted Image            |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image2] | ![alt text][image3]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and sobel thresholds to generate a binary image.  
The `thresholding()` function is part of the [fourth code cell](AdvancedLaneLineDetection.ipynb#Thresh).

The steps that are taken in the function are:

* Copy Input image.
* Convert image from BGR to Grayscale.
* Take sobel derivative in x-direction and scale it to 8-bits.
* Create a mask using the `sx_thresh`.
* Convert image from BGR to HSV color space and extract h and v channels.
* Take an average of the v-channel in the following window [450:600,500:800]. This will approximate the lighting and/or pavement color.
* Depending on the average, adjust the l-channel threshold, v-channel threshold, and whether or not to use sobel.
* Create an h and v channel masks, which in combination will help isolate the yellow lanes from the image.
* Convert image from BGR to HLS color space and extract l channel.
* Create an l-channel mask.
* Create a final binary image by combining masks in the following way:
``` python
combined_binary[((sxbinary == 1) & (use_sobel == 1)) | (l_binary == 1) | ((h_binary == 1) & (v_binary == 1))] = 1
```

I found that you can get better lane extraction by varying the threshold levels of the l and v channels according to the conditions.

Here's an example of my output for this step.

Undistorted Image          |  Thresholded Image
:-------------------------:|:-------------------------:
![alt text][image3] | ![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes two functions `warpimg` and `unwarpimg`, found in the [fifth code cell](AdvancedLaneLineDetection.ipynb#Warp) of the IPython notebook AdvancedLaneLineDetection.

``` python
def warpimg(img, M):
    """ Warp image img using transform matrix M. """
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def unwarpimg(img, Minv):
    """ Unwarp image img using transform matrix Minv. """
    img_size = (img.shape[1], img.shape[0])
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return unwarped 
```

Both function take as inputs an image (`img`) and a transformation matrix (either `M` for warping or `Minv` for unwarping).

The matricies are generated by calling `getPerspectiveTransform()` function, with `src` and `dst` points as inputs.

``` python
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
```

The points that I chose for the transform are:

```python
x_cent_off = 60
y_cent_off = 100

src = np.float32(
[[(img_size[0] / 2) - x_cent_off, (img_size[1] / 2) + y_cent_off],
[((img_size[0] / 6) - 20), img_size[1]],
[(img_size[0] * 5 / 6) + 50, img_size[1]],
[(img_size[0] / 2) + x_cent_off, (img_size[1] / 2) + y_cent_off]])

dst = np.float32(
[[(img_size[0] / 4), 0],
[(img_size[0] / 4), img_size[1]],
[(img_size[0] * 3/4), img_size[1]],
[(img_size[0] * 3/4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 193.3, 720    | 320, 720      |
| 1116.6, 720   | 960, 720      |
| 700, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code needed to identify lane-line pixel and fit a polynomial to their positions is located in the [sixth code cell](AdvancedLaneLineDetection.ipynb#FindLane) of the notebook.

There are two approaches that are taken to find the lane lines. One is a sliding box fit approach, which uses a
function `find_lane_pixels_box()`. Function starts by creating a histogram of the cropped bottom half of the image.
The left and right 20% of the image are ignored to make finding the start of the lane lines more accurate.

``` python
bottom_half[:,0:np.int(width/5)] = 0
bottom_half[:,np.int(width*4/5):] = 0
```

The starting point of the lanes is assumed to be a point that has the most number of pixels vertically.

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

Here's a [link to my challenge video result](./output_videos/challenge_video.mp4)

My pipeline didn't work well with the harder challenge video and I'll mention why in the discussion section.
Here's [link to my harder challenge video result](./output_videos/harder_challenge_video.mp4), if that is of interest.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
