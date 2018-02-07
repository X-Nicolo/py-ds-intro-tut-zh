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
