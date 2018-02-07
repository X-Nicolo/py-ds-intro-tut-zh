# 图像和视频分析

# 一、Python OpenCV 入门

![](https://pythonprogramming.net/static/images/opencv/opencv-intro-tutorial-python.gif)

欢迎阅读系列教程，内容涵盖 OpenCV，它是一个图像和视频处理库，包含 C ++，C，Python 和 Java 的绑定。 OpenCV 用于各种图像和视频分析，如面部识别和检测，车牌阅读，照片编辑，高级机器人视觉，光学字符识别等等。

您将需要两个主要的库，第三个可选：python-OpenCV，Numpy 和 Matplotlib。

### Windows 用户：

[python-OpenCV](https://www.lfd.uci.edu/~gohlke/pythonlibs/#opencv)：有其他的方法，但这是最简单的。 下载相应的 wheel（`.whl`）文件，然后使用`pip`进行安装。 观看视频来寻求帮助。

```
pip install numpy

pip install matplotlib
```

不熟悉使用`pip`？ 请参阅`pip`安装教程来获得帮助。

### Linux/Mac 用户


```
pip3 install numpy 
```

或者

```
apt-get install python3-numpy
```

你可能需要`apt-get`来安装`python3-pip`。

```
pip3 install matplotlib 
```

或者

```
apt-get install python3-matplotlib

apt-get install python-OpenCV
```

Matplotlib 是用于展示来自视频或图像的帧的可选选项。 我们将在这里展示几个使用它的例子。 Numpy 被用于“数值和 Python”的所有东西。 我们主要利用 Numpy 的数组功能。 最后，我们使用`python-OpenCV`，它是 Python 特定的 OpenCV 绑定。

OpenCV 有一些操作，如果没有完整安装 OpenCV （大小约 3GB），你将无法完成，但是实际上你可以用 python-OpenCV 最简安装。 我们将在本系列的后续部分中使用 OpenCV 的完整安装，如果您愿意的话，您可以随意获得它，但这三个模块将使我们忙碌一段时间！

通过运行 Python 并执行下列命令来确保您安装成功：

```py
import cv2
import matplotlib
import numpy
```

如果你没有错误，那么你已经准备好了。好了嘛？让我们下潜吧！

首先，在图像和视频分析方面，我们应该了解一些基本的假设和范式。对现在每个摄像机的记录方式来说，记录实际上是一帧一帧地显示，每秒 30-60 次。但是，它们的核心是静态帧，就像图像一样。因此，图像识别和视频分析大部分使用相同的方法。有些东西，如方向跟踪，将需要连续的图像（帧），但像面部检测或物体识别等东西，在图像和视频中代码几乎完全相同。

接下来，大量的图像和视频分析归结为尽可能简化来源。这几乎总是起始于转换为灰度，但也可以是彩色滤镜，渐变或这些的组合。从这里，我们可以对来源执行各种分析和转化。一般来说，这里发生的事情是转换完成，然后是分析，然后是任何覆盖，我们希望应用在原始来源上，这就是你可以经常看到，对象或面部识别的“成品”在全色图像或视频上显示。然而，数据实际上很少以这种原始形式处理。有一些我们可以在基本层面上做些什么的例子。所有这些都使用基本的网络摄像头来完成，没有什么特别的：

### 背景提取

![](https://pythonprogramming.net/static/images/opencv/opencv-background-subtracting.png)


### 颜色过滤

![](https://pythonprogramming.net/static/images/opencv/opencv-filtering.jpg)


### 边缘检测

![](https://pythonprogramming.net/static/images/opencv/opencv-edge-detection.png)

### 用于对象识别的特征匹配

![](https://pythonprogramming.net/static/images/opencv/opencv-feature-matching.png)

### 一般对象识别

![](https://pythonprogramming.net/static/images/opencv/opencv-object-recognition.png)

在边缘检测的情况下，黑色对应于`(0,0,0)`的像素值，而白色线条是`(255,255,255)`。视频中的每个图片和帧都会像这样分解为像素，并且像边缘检测一样，我们可以推断，边缘是基于白色与黑色像素对比的地方。然后，如果我们想看到标记边缘的原始图像，我们记录下白色像素的所有坐标位置，然后在原始图像或视频上标记这些位置。

到本教程结束时，您将能够完成上述所有操作，并且能够训练您的机器识别您想要的任何对象。就像我刚开始说的，第一步通常是转换为灰度。在此之前，我们需要加载图像。因此，我们来做吧！在整个教程中，我极力鼓励你使用你自己的数据来玩。如果你有摄像头，一定要使用它，否则找到你认为很有趣的图像。如果你有麻烦，这是一个手表的图像：

![](https://pythonprogramming.net/static/images/opencv/watch.jpg)

```py
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

首先，我们正在导入一些东西，我已经安装了这三个模块。接下来，我们将`img`定义为`cv2.read(image file, parms)`。默认值是`IMREAD_COLOR`，这是没有任何 alpha 通道的颜色。如果你不熟悉，alpha 是不透明度（与透明度相反）。如果您需要保留 Alpha 通道，也可以使用`IMREAD_UNCHANGED`。很多时候，你会读取颜色版本，然后将其转换为灰度。如果您没有网络摄像机，这将是您在本教程中使用的主要方法，即加载图像。

你可以不使用`IMREAD_COLOR` ...等，而是使用简单的数字。你应该熟悉这两种选择，以便了解某个人在做什么。对于第二个参数，可以使用`-1`，`0`或`1`。颜色为`1`，灰度为`0`，不变为`-1`。因此，对于灰度，可以执行`cv2.imread('watch.jpg', 0)`。

一旦加载完成，我们使用`cv2.imshow(title,image)`来显示图像。从这里，我们使用`cv2.waitKey(0)`来等待，直到有任何按键被按下。一旦完成，我们使用`cv2.destroyAllWindows()`来关闭所有的东西。

正如前面提到的，你也可以用 Matplotlib 显示图像，下面是一些如何实现的代码：

```py
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
plt.show()
```

请注意，您可以绘制线条，就像任何其他 Matplotlib 图表一样，使用像素位置作为坐标的。 不过，如果你想绘制你的图片，Matplotlib 不是必需的。 OpenCV 为此提供了很好的方法。 当您完成修改后，您可以保存，如下所示：

```py
cv2.imwrite('watchgray.png',img)
```

将图片导入 OpenCV 似乎很容易，加载视频源如何？ 在下一个教程中，我们将展示如何加载摄像头或视频源。

## 二、加载视频源

在这个 Python OpenCV 教程中，我们将介绍一些使用视频和摄像头的基本操作。 除了起始行，处理来自视频的帧与处理图像是一样的。 我们来举例说明一下：

```py
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
 
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

首先，我们导入`numpy`和`cv2`，没有什么特别的。 接下来，我们可以`cap = cv2.VideoCapture(0)`。 这将从您计算机上的第一个网络摄像头返回视频。 如果您正在观看视频教程，您将看到我正在使用`1`，因为我的第一个摄像头正在录制我，第二个摄像头用于实际的教程源。

```py
while(True):
    ret, frame = cap.read()
```

这段代码启动了一个无限循环（稍后将被`break`语句打破），其中`ret`和`frame`被定义为`cap.read()`。 基本上，`ret`是一个代表是否有返回的布尔值，`frame`是每个返回的帧。 如果没有帧，你不会得到错误，你会得到`None`。

```py
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

在这里，我们定义一个新的变量`gray`，作为转换为灰度的帧。 注意这个`BGR2GRAY`。 需要注意的是，OpenCV 将颜色读取为 BGR（蓝绿色红色），但大多数计算机应用程序读取为 RGB（红绿蓝）。 记住这一点。

```py
    cv2.imshow('frame',gray)
```

请注意，尽管是视频流，我们仍然使用`imshow`。 在这里，我们展示了转换为灰色的源。 如果你想同时显示，你可以对原始帧和灰度执行`imshow`，将出现两个窗口。

```py
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

这个语句每帧只运行一次。 基本上，如果我们得到一个按键，那个键是`q`，我们将退出`while`循环，然后运行：

```py
cap.release()
cv2.destroyAllWindows()
```

这将释放网络摄像头，然后关闭所有的`imshow()`窗口。

在某些情况下，您可能实际上需要录制，并将录制内容保存到新文件中。 以下是在 Windows 上执行此操作的示例：

```py
import numpy as np
import cv2

cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
```

这里主要要注意的是正在使用的编解码器，以及在`while`循环之前定义的输出信息。 然后，在`while`循环中，我们使用`out.write()`来输出帧。 最后，在`while`循环之外，在我们释放摄像头之后，我们也释放`out`。

太好了，现在我们知道如何操作图像和视频。 如果您没有网络摄像头，您可以使用图像甚至视频来跟随教程的其余部分。 如果您希望使用视频而不是网络摄像头作为源，则可以为视频指定文件路径，而不是摄像头号码。

现在我们可以使用来源了，让我们来展示如何绘制东西。 此前您已经看到，您可以使用 Matplotlib 在图片顶部绘制，但是 Matplotlib 并不真正用于此目的，特别是不能用于视频源。 幸运的是，OpenCV 提供了一些很棒的工具，来帮助我们实时绘制和标记我们的源，这就是我们将在下一个教程中讨论的内容。

## 三、在图像上绘制和写字

在这个 Python OpenCV 教程中，我们将介绍如何在图像和视频上绘制各种形状。 想要以某种方式标记检测到的对象是相当普遍的，所以我们人类可以很容易地看到我们的程序是否按照我们的希望工作。 一个例子就是之前显示的图像之一：

![](https://pythonprogramming.net/static/images/opencv/opencv-intro-tutorial-python.gif)

对于这个临时的例子，我将使用下面的图片：

![](https://pythonprogramming.net/static/images/opencv/watch.jpg)

鼓励您使用自己的图片。 像往常一样，我们的起始代码可以是这样的：

```py
import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
```

下面，我们可以开始绘制，这样：

```py
cv2.line(img,(0,0),(150,150),(255,255,255),15)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

`cv2.line()`接受以下参数：图片，开始坐标，结束坐标，颜色（bgr），线条粗细。

结果在这里：

![](https://pythonprogramming.net/static/images/opencv/opencv-line-draw-tutorial.png)

好吧，很酷，让我们绘制更多形状。 接下来是一个矩形：

```py
cv2.rectangle(img,(15,25),(200,150),(0,0,255),15)
```

这里的参数是图像，左上角坐标，右下角坐标，颜色和线条粗细。

圆怎么样？

```py
cv2.circle(img,(100,63), 55, (0,255,0), -1)
```

这里的参数是图像/帧，圆心，半径，颜色和。 注意我们粗细为`-1`。 这意味着将填充对象，所以我们会得到一个圆。

线条，矩形和圆都很酷，但是如果我们想要五边形，八边形或十八边形？ 没问题！

```py
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
# OpenCV documentation had this code, which reshapes the array to a 1 x 2. I did not 
# find this necessary, but you may:
#pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
```

首先，我们将坐标数组称为`pts`（点的简称）。 然后，我们使用`cv2.polylines`来画线。 参数如下：绘制的对象，坐标，我们应该连接终止的和起始点，颜色和粗细。

你可能想要做的最后一件事是在图像上写字。 这可以这样做：

```py
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(0,130), font, 1, (200,255,155), 2, cv2.LINE_AA)
```

目前为止的完整代码：

```py
import numpy as np
import cv2

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
cv2.line(img,(0,0),(200,300),(255,255,255),50)
cv2.rectangle(img,(500,250),(1000,500),(0,0,255),15)
cv2.circle(img,(447,63), 63, (0,255,0), -1)
pts = np.array([[100,50],[200,300],[700,200],[500,100]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img, [pts], True, (0,255,255), 3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'OpenCV Tuts!',(10,500), font, 6, (200,255,155), 13, cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

结果：

![](https://pythonprogramming.net/static/images/opencv/opencv-python-drawing-on-image-tutorial.png)

在下一个教程中，我们将介绍我们可以执行的基本图像操作。

## 四、图像操作

在 OpenCV 教程中，我们将介绍一些我们可以做的简单图像操作。 每个视频分解成帧。 然后每一帧，就像一个图像，分解成存储在行和列中的，帧/图片中的像素。 每个像素都有一个坐标位置，每个像素都由颜色值组成。 让我们列举访问不同的位的一些例子。

我们将像往常一样读取图像（如果可以，请使用自己的图像，但这里是我在这里使用的图像）：

![](https://pythonprogramming.net/static/images/opencv/watch.jpg)

```py
import cv2
import numpy as np

img = cv2.imread('watch.jpg',cv2.IMREAD_COLOR)
```

现在我们可以实际引用特定像素，像这样：

```py
px = img[55,55]
```

下面我们可以实际修改像素：

```py
img[55,55] = [255,255,255]
```

之后重新引用：

```py
px = img[55,55]
print(px)
```

现在应该不同了，下面我们可以引用 ROI，图像区域：

```py
px = img[100:150,100:150]
print(px)
```

我们也可以修改 ROI，像这样：

```py
img[100:150,100:150] = [255,255,255]
```

我们可以引用我们的图像的特定特征：

```py
print(img.shape)
print(img.size)
print(img.dtype)
```

我们可以像这样执行操作：

```py
watch_face = img[37:111,107:194]
img[0:74,0:87] = watch_face

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

这会处理我的图像，但是可能不能用于你的图像，取决于尺寸。这是我的输出：

![](https://pythonprogramming.net/static/images/opencv/opencv-python-image-oeprations-tutorial.png)

这些是一些简单的操作。 在下一个教程中，我们将介绍一些我们可以执行的更高级的图像操作。

## 五、图像算术和逻辑运算

欢迎来到另一个 Python OpenCV 教程，在本教程中，我们将介绍一些简单算术运算，我们可以在图像上执行的，并解释它们的作用。 为此，我们将需要两个相同大小的图像来开始，然后是一个较小的图像和一个较大的图像。 首先，我将使用：

![](https://pythonprogramming.net/static/images/opencv/3D-Matplotlib.png)

和

![](https://pythonprogramming.net/static/images/opencv/mainsvmimage.png)

首先，让我们看看简单的加法会做什么：

```py
import cv2
import numpy as np

# 500 x 250
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

add = img1+img2

cv2.imshow('add',add)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

结果：

![](https://pythonprogramming.net/static/images/opencv/opencv-python-image-addition-tutorial.png)

你不可能想要这种混乱的加法。 OpenCV 有一个“加法”方法，让我们替换以前的“加法”，看看是什么：

```py
add = cv2.add(img1,img2)
```

结果：

![](https://pythonprogramming.net/static/images/opencv/opencv-add-python.png)

这里可能不理想。 我们可以看到很多图像是非常“白色的”。 这是因为颜色是 0-255，其中 255 是“全亮”。 因此，例如：`(155,211,79) + (50, 170, 200) = 205, 381, 279`...转换为`(205, 255,255)`。

接下来，我们可以添加图像，并可以假设每个图像都有不同的“权重”。 这是如何工作的：

```py
import cv2
import numpy as np

img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainsvmimage.png')

weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imshow('weighted',weighted)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

对于`addWeighted`方法，参数是第一个图像，权重，第二个图像，权重，然后是伽马值，这是一个光的测量值。 我们现在就把它保留为零。

![](https://pythonprogramming.net/static/images/opencv/opencv-addWeighted-tutorial.png)

这些是一些额外的选择，但如果你真的想将一个图像添加到另一个，最新的重叠在哪里？ 在这种情况下，你会从最大的开始，然后添加较小的图像。 为此，我们将使用相同的`3D-Matplotlib.png`图像，但使用一个新的 Python 标志：

![](https://pythonprogramming.net/static/images/opencv/mainlogo.png)

现在，我们可以选取这个标志，并把它放在原始图像上。 这很容易（基本上使用我们在前一个教程中使用的相同代码，我们用一个新的东西替换了图像区域（ROI）），但是如果我们只想要标志部分而不是白色背景呢？ 我们可以使用与之前用于 ROI 替换相同的原理，但是我们需要一种方法来“去除”标志的背景，使得白色不会不必要地阻挡更多背景图像。 首先我将显示完整的代码，然后解释：

```py
import cv2
import numpy as np

# Load two images
img1 = cv2.imread('3D-Matplotlib.png')
img2 = cv2.imread('mainlogo.png')

# I want to put logo on top-left corner, So I create a ROI
rows,cols,channels = img2.shape
roi = img1[0:rows, 0:cols ]

# Now create a mask of logo and create its inverse mask
img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

# add a threshold
ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

mask_inv = cv2.bitwise_not(mask)

# Now black-out the area of logo in ROI
img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)

# Take only region of logo from logo image.
img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

dst = cv2.add(img1_bg,img2_fg)
img1[0:rows, 0:cols ] = dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

这里发生了很多事情，出现了一些新的东西。 我们首先看到的是一个新的阈值：`ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)`。

我们将在下一个教程中介绍更多的阈值，所以请继续关注具体内容，但基本上它的工作方式是根据阈值将所有像素转换为黑色或白色。 在我们的例子中，阈值是 220，但是我们可以使用其他值，或者甚至动态地选择一个，这是`ret`变量可以使用的值。 接下来，我们看到：`mask_inv = cv2.bitwise_not(mask)`。 这是一个按位操作。 基本上，这些操作符与 Python 中的典型操作符非常相似，除了一点，但我们不会在这里触及它。 在这种情况下，不可见的部分是黑色的地方。 然后，我们可以说，我们想在第一个图像中将这个区域遮住，然后将空白区域替换为图像 2 的内容。

![](https://pythonprogramming.net/static/images/opencv/opencv-bitwise-threshold-example.png)

下个教程中，我们深入讨论阈值。

## 六、阈值

欢迎阅读另一个 OpenCV 教程。在本教程中，我们将介绍图像和视频分析的阈值。阈值的思想是进一步简化视觉数据的分析。首先，你可以转换为灰度，但是你必须考虑灰度仍然有至少 255 个值。阈值可以做的事情，在最基本的层面上，是基于阈值将所有东西都转换成白色或黑色。比方说，我们希望阈值为 125（最大为 255），那么 125 以下的所有内容都将被转换为 0 或黑色，而高于 125 的所有内容都将被转换为 255 或白色。如果你像平常一样转换成灰度，你会变成白色和黑色。如果你不转换灰度，你会得到二值化的图片，但会有颜色。

虽然这听起来不错，但通常不是。我们将在这里介绍多个示例和不同类型的阈值来说明这一点。我们将使用下面的图片作为我们的示例图片，但可以随意使用您自己的图片：

![](https://pythonprogramming.net/static/images/opencv/bookpage.jpg)

这个书的图片就是个很好的例子，说明为什么一个人可能需要阈值。 首先，背景根本没有白色，一切都是暗淡的，而且一切都是变化的。 有些部分很容易阅读，另一部分则非常暗，需要相当多的注意力才能识别出来。 首先，我们尝试一个简单的阈值：

```py
retval, threshold = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
```

二元阈值是个简单的“是或不是”的阈值，其中像素为 255 或 0。在很多情况下，这是白色或黑色，但我们已经为我们的图像保留了颜色，所以它仍然是彩色的。 这里的第一个参数是图像。 下一个参数是阈值，我们选择 10。下一个是最大值，我们选择为 255。最后是阈值类型，我们选择了`THRESH_BINARY`。 通常情况下，10 的阈值会有点差。 我们选择 10，因为这是低光照的图片，所以我们选择低的数字。 通常 125-150 左右的东西可能效果最好。

```py
import cv2
import numpy as np

img = cv2.imread('bookpage.jpg')
retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

结果：

![](https://pythonprogramming.net/static/images/opencv/opencv-python-binary-threshold-tutorial.png)

现在的图片稍微更便于阅读了，但还是有点乱。 从视觉上来说，这样比较好，但是仍然难以使用程序来分析它。 让我们看看我们是否可以进一步简化。

首先，让我们灰度化图像，然后使用一个阈值：

```py
import cv2
import numpy as np

grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![](https://pythonprogramming.net/static/images/opencv/opencv-python-threshold-gray-binary-tutorial.png)

更简单，但是我们仍然在这里忽略了很多背景。 接下来，我们可以尝试自适应阈值，这将尝试改变阈值，并希望弄清楚弯曲的页面。

```py
import cv2
import numpy as np

th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
cv2.imshow('original',img)
cv2.imshow('Adaptive threshold',th)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

![](https://pythonprogramming.net/static/images/opencv/opencv-python-adaptive-gaussian-threshold-tutorial.png)


还有另一个版本的阈值，可以使用，叫做大津阈值。 它在这里并不能很好发挥作用，但是：

```py
retval2,threshold2 = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('original',img)
cv2.imshow('Otsu threshold',threshold2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
