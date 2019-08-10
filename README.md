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
[image6]: ./output_images/FitImages.png "Fit Images"
[image7]: ./output_images/FinalImage.png "Final Image"

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

The code for my perspective transform includes two functions `warpimg()` and `unwarpimg()`, found in the [fifth code cell](AdvancedLaneLineDetection.ipynb#Warp) of the IPython notebook AdvancedLaneLineDetection.

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

There are two approaches that are taken to find the lane lines. 

First one is a sliding box fit approach, which uses a
function `find_lane_pixels_box()`. Function starts by creating a histogram of the cropped bottom half of the image.
The left and right 20% of the image are ignored to make finding the start of the lane lines more accurate.

``` python
bottom_half[:,0:np.int(width/5)] = 0
bottom_half[:,np.int(width*4/5):] = 0
```

The starting point of the lanes is assumed to be a point that has the most number of pixels vertically.

This starting point is used as the center of the first box that is created to detect lane pixels.
For each lane there are a total of up to 15 boxes being created of 100 pixel width. 

You can see the boxes in the left part of the image below.

![alt text][image6]

As each box is processed, if the number of pixels found in the box is above a certain minimum (150), then the next box's center gets recentered according to the mean x-axis location of the pixels.I've added a check of how much the current and previous box's centers differ. If the difference is above 40, then new center gets adjusted by this differnce as well. This is meant to help track the lane pixels in sharp turn situations.

``` python
if len(good_left_inds) > minpix:
    new_leftx = np.mean(nonzerox[good_left_inds])
    diff = new_leftx - leftx_current
    if (diff >= sharpturn):
        leftx_current = np.int(new_leftx + diff)
    elif (diff <= -sharpturn):
        leftx_current = np.int(new_leftx + diff)
    else:
        leftx_current = np.int(new_leftx)

```
As the function works its way to the top of the image, in case of very sharp turns, it's possible that the lane pixels will end somewhere below the top of the image. To prevent any new boxes from being processed once the edge of the image is reached, I've added a check for when the box's low or high x-axis index reaches outside of image boundaries. 

``` python
if win_xleft_low < 0 or win_xleft_high > binary_warped.shape[1]:
    left_inbounds = 0
if win_xright_low < 0 or win_xright_high > binary_warped.shape[1]:
    right_inbounds = 0
            
```

If there are left and right lane pixels found, function `fit_poly()` is called to calculate the 2nd order polynomial fit for those points. 
It uses numpy's polyfit function to get the fit coefficients. Those coefficients are then used to construct the lane-line's x-axis coordinates for each y-axis point. You can see the polynomial approximations in yellow on the figure below.

![alt text][image6]


The second approach to finding lane lanes, and the one that can be seen on the right part of the image above, uses a window to find the lane line pixels. The code for this can be found in the function `find_lane_pixels_window()`.

This function relies on the best fit found previously using the sliding box fit method. It will not work on it's own.
However, it is a good way to find the polynomial fit in frames following the one where the fit was found using the boxes.
This method is called the window approach because it works by searching for pixels in the window around the previous best fit's polynomial within a `margin` of 100 pixels.

``` python
left_lane_inds = np.nonzero(np.abs(left_fitx - nonzerox) < margin)[0]
right_lane_inds = np.nonzero(np.abs(right_fitx - nonzerox) < margin)[0]
```
Once the lane line pixels are identified, the next fit is calcualted using the same `fit_poly()` function.

The above calculations to find the polynomial fit were done in terms of pixels. However, to calculate the radius of the curvature of the lane, a fit in engineering units such as meters is needed. That's why there is a separate function `fit_poly_real()`, which calculates the polynomial coefficients with the following pixel to meter coefficient in mind:

``` python
ym_per_pix = 3.048/95 # meters per pixel in y dimension
xm_per_pix = 3.7/640 # meters per pixel in x dimension
```

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code needed to calculate radius of curvature and position of the vehicle with respect to center is located in the [seventh code cell](AdvancedLaneLineDetection.ipynb#FindCurve) of the notebook. It's included in the function `measure_curvature_real()`.

The curvature radius is calcualted using the formula found [on this page]https://www.intmath.com/applications-differentiation/8-radius-curvature.php

The overall curvature radius is the average of the two lane line's curvatures.

The vehicle's center offset from the center of the lane is found using the assumption that the car's center is aligned with the image (or camera's) center:
``` python
car_position = img.shape[1]/2*xm_per_pix
```
The offset is then the difference between the lane's center and the vehicle center position.

``` python
lane_center_position = (right_lane_pos + left_lane_pos) /2
center_offset = (car_position - lane_center_position)
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The drawing function `draw_lane_bounds()` is located in the [eighth code cell](AdvancedLaneLineDetection.ipynb#DrawLane) of the notebook.

It uses the lane polynomial fit information to generate the x-axis coordinates of the lane lines for each y-axis coordinate.
Then, it recasts those coordinates using numpy array manipulation and then draws a polygon in the warped perspective using the coordinates and OpenCV's `fillPoly()` function.

``` python
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
```

The transformation inverse matrix is then used to unwarp the polygon.

``` python
newwarp = cv2.warpPerspective(color_warp, Minv, (width, height)) 
result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
```

The polygon's image is then combined with undistorted image for the final output.

Information about lane curvature radius and vehicle's center offset from the middle of the lane is also drawn on the final image using `draw_data()` function found in the [ninth code cell](AdvancedLaneLineDetection.ipynb#DrawData) of the notebook. It uses OpenCV's `putText()` function.

An example of a final image output can be seen below:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

To process a video, in addition to the steps listed above, there are some santiy checks performed on the detected polynomial fits of lane lines. They are found in the function `check_fit()` found in the [ninth code cell](AdvancedLaneLineDetection.ipynb#DrawData)  of the notebook. Main checks that are performed:

* Check if the x-intercept of the left and right lane are too far apart or too close
* Check if the difference between the right and left x-coordinates at y = 0 is negative. This would mean line are crossing!
* Check if difference between the right and left x-coordinates at y = 0 is much greater than difference at x-intercept. This would mean lines are diverging!

This is a standalone check that is done on every frame's polynomial fits to filter out bad frames.

In addition, a Lane class was created in the [eleventh code cell](AdvancedLaneLineDetection.ipynb#Pipeline). This is a class that allows to keep track of lane information between frames. As a part of this class, I've created an `update_fit()` function to help smooth out the output and add additional checks for badly detected lane lines. This function uses the difference between previous best fit and current fit to identify bad lane fits:

``` python
self.diffs = abs(fit - self.best_fit)
if self.diffs[0] < 0.0004 and self.diffs[1] < 0.4 and self.diffs[2] < 300:
    self.detected = True
else:
    self.detected = False
```

The function also stores the last 10 valid fits in a variable `current_fit` and averages them to create the best fit of the lane line.
``` python
if len(self.current_fit) > 0:
    self.best_fit = np.average(self.current_fit, axis=0)
```

The same averaging is done on the curvature radius.

The final output frame is generated using each lane line's best fits.

Here's a [link to my video result](./output_videos/project_video.mp4) that shows the complete video processing pipeline output.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest amount of time spent on this project was on the thresholding function. It worked well on the main project video, but wasn't giving good results on the challenging videos. The main reason for that is the variability in road surface color and especially the lighting conditions, such as under the bridge in the challenge video. I've eventually added some logic to adjust the parameters on the fly, based on a sample of the frame to try to make the pipeline more robust. It's still not perfect, as you can see it struggle in the [challenge video result](./output_videos/challenge_video.mp4) 

The problem is you want the function to be sensitive enough to pick up enough pixels to detect lane lines in most frames, but that also causes it to be too sensitive and pick up unrelated pixels that confuse the lane line calculation function.

The pipeline struggles even more with the harder challenge video, expectedly so. There are sharp turns, high variability in lighting conditions and even frames where no human could detect lane lines because of sun shining straight into the camera's lens and making everything in the frame bright. Here's [link to my harder challenge video result](./output_videos/harder_challenge_video.mp4), if that is of interest.

The improvements that I think could be made to make it more robust are:
* More parameter tuning
* More dynmaic parameter changes according to the lighting conditions
* Additional sanity checks
* Investigation of other color spaces
