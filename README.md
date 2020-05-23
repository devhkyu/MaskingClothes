MaskingClothes
==============
`MaskingClothes` is an `OpenAPI` that masks images and classifies upper, lower, whole items using Mask-RCNN(matterport).
<br>`<Reference> Smart Coordinator: https://github.com/ChoiHyungKyu/Smart_Coordinator`

Installation
------------
![python](https://img.shields.io/badge/Python-3.7.3%2B-blue.svg)
![tensorflow](https://img.shields.io/badge/Tensorflow-1.14.0%2B-blue.svg)
![keras](https://img.shields.io/badge/Keras-2.3.0%2B-blue.svg)

### Python Modules
If you want to run this code, you should install `below modules`.
```
pip install tensorflow==1.14.0
pip install keras==2.2.5
pip install "numpy<1.17"
pip install pillow, cv2
```

### Source Files
```
MaskingClothes/Source/label_descriptions.json   // Fashion Category
MaskingClothes/Source/mask_rcnn_fashion_0006.h5   // Weight File (Smart_Coordinator)
```
You can download source files(226.8Mb) at `http://naver.me/xLvqecht` <br>
If you can't download, please contact `fab@kakao.com` `(Hyungkyu Choi)`

Param & Return Values
---------
```
def __init__(self, img_size=None, threshold=None, gpu_count=None, images_per_gpu=None):
    ...
    
    - img_size(default: 512)
    - threshold(default: 0.7)
    - gpu_count(default: 1)
    - images_per_gpu(default: 1)
```
```
def run(self, IMG_DIR):
    ...
    return img, masked_image, label_type, label, score, complete
    
    - IMG_DIR = directory of image (ex: Images/mask1.jpg)
    - img = Original image (Image)
    - masked_image = Result Image (list of Image)
    - label_type = Upper, Lower, Whole (list)
    - label = Specific category name (list)
    - complete = Whether model detects items well
```

Usage with Example code
------------

### Example 1: Masking complete sets
```
import mask_clothes
model = mask_clothes.Model(img_size=512, threshold=0.7, gpu_count=1, images_per_gpu=1)
ROOT_DIR = 'Result/'

for x in range(1, 20):
    img, masked_image, label_type, label, score, complete = model.run(IMG_DIR='Images/mask' + str(x) + '.jpg')
    if complete is True:
        for y in range(len(label)):
            directory = ROOT_DIR + label_type[y] + '/' + str(x) + '_' + label[y] + '.jpeg'
            masked_image[y].save(directory)
```

![ex1](https://user-images.githubusercontent.com/44195740/82737650-7cdbd800-9d6d-11ea-935c-f0a10b9737c6.png)

### Example 2: Displaying masked images
```
import mask_clothes
model = mask_clothes.Model(img_size=512, threshold=0.7, gpu_count=1, images_per_gpu=1)

img, masked_image, label_type, label, score, complete = model.run(IMG_DIR='Images/mask1.jpg')
for x in masked_image:
    x.show()
```

![ex2](https://user-images.githubusercontent.com/44195740/82737653-85cca980-9d6d-11ea-9338-a73fa027d621.png)

License
=======
```
MIT License

Copyright (c) 2020 hky.u

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
