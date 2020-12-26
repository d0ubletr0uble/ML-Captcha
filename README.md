# ML-Captcha
The purpose of this project is to solve captchas. Simple as that.

## Image preprocessing 
Here are some captcha examples I'll try to solve.

This is a real world problem and you see that these captchas seem to be pretty robust

![1.png](Example%20Captchas/1.png)
![2.png](Example%20Captchas/2.png)
![3.png](Example%20Captchas/3.png)
![4.png](Example%20Captchas/4.png)
![5.png](Example%20Captchas/5.png)
![6.png](Example%20Captchas/6.png)
![7.png](Example%20Captchas/7.png)
![8.png](Example%20Captchas/8.png)
![9.png](Example%20Captchas/9.png)
![10.png](Example%20Captchas/10.png)

#### Background removal
Images seem to have only 4 backgrounds.

We can extract background from image by using median pixel value of different pictures with same background.
I'll use ImageMagick tool `convert` for this task

`convert 1.png 2.png 3.png -evaluate-sequence median result.jpg`

After getting more images and grouping them by background we can finally extract backgrounds.

![Carpet.png](Backgrounds/Carpet.png)
![Glass.png](Backgrounds/Glass.png)
![LasVegas.png](Backgrounds/LasVegas.png)
![Pink.png](Backgrounds/Pink.png)

They seem to have some dead pixels, but its ok.

Lets see what happens if we compute bitwise `xor` between captcha and backgrounds.

Here is the captcha: 

![5.png](Example%20Captchas/5.png)


And `Xor` output of above captcha with all backgrounds:

![](Example%20Captchas/Xored/XorCarpet.png)
![](Example%20Captchas/Xored/XorGlass.png)
![](Example%20Captchas/Xored/XorLasVegas.png)
![](Example%20Captchas/Xored/XorPink.png)

`Xor` with matching background produces image with removed dark background.

Dark pixels have lower numerical values than coloured ones so we select image with lowest `sum` of numerical pixel values.


With this rather simple step we accomplish background recognition and removal

Next we convert image to black and white only.

![](Example%20Captchas/blackwhite.png)

## To be continued...

