# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline.  

####As part of the description, explain how you modified the draw_lines() function.

My pipeline consists of a
* greyscale
* gaussian blur
* canny edge detector
* mask
* hough transform

for the mask, I set the quadralateral that started 60% down the image.
The base of the quadralateral is 10% in from the bottum left, 5% in from the bottum right.
the top of the quadralateral extends across the middle 10% of the image.

in the draw_lines() function I added a calculation for the line gradient and the line length.
The gradient for the left line is roughly -0.7 and the gradient for the right line is roughly 0.7.
From the line segments that fit this description, I worked out the furthest point that fit and also
the closest point to the camera that fit. If the closest line wasn't at the bottum of the screen I extrapolated
where this would be so that I could plot a solid line.

I did not plot lines that had:
* lines without length
* lines with a low < 0.3 gradient (horizontal lines)
* lines with a high > 0.9 gradient (vertical lines)


### 2. Identify potential shortcomings with your current pipeline

One potential shortcoming is bends in the road. The quadralateral cannot capture curves properly.
Another shortcoming could be with lines that have big gaps that would require a bigger max_line_gap but can also result in a more noisy result.

Another shortcoming could be with the lengthly max_line_gap in the hough transform, noise can come in and looking at the advanced video that is definitely the case. 

Another shortcoming is with the fixed size of the quadralateral, if the car sways left or right in the lane it could fall outside of the quadralateral. Also when going up a hill, the top of the quadralateral needs to be lower down and when going down a slope it needs to be high to adjust for the horizon shifting.

Another shortcoming is if the colour of the line is too similar to the road, the edge detector maynot pick this up.

### 3. Suggest possible improvements to your pipeline

A possible improvement would be to use splines rather than straight line segments to account for bends

Another potential improvement could be make the mask dynamic so that it works out the ideal horizon point based on the skyline it and also able to adjust for bends.


