# Python 和 Pandas 数据分析教程

大家好，欢迎阅读 Python 和 Pandas 数据分析系列教程。 Pandas 是一个 Python 模块，Python 是我们要使用的编程语言。Pandas 模块是一个高性能，高效率，高水平的数据分析库。

它的核心就像操作一个电子表格的无头版本，比如 Excel。你使用的大多数数据集将是所谓的数据帧（`DataFrame`）。您可能已经熟悉这个术语，它也用于其他语言，但是如果没有，数据帧通常就像电子表格一样，拥有列和行，这就是它了！从这里开始，我们可以利用 Pandas 以闪电般的速度操作我们的数据集。

Pandas 还与许多其他数据分析库兼容，如用于机器学习的 Scikit-Learn，用于图形的 Matplotlib，NumPy，因为它使用 NumPy ，以及其他。这些是非常强大和宝贵的。如果你发现自己使用 Excel 或者一般电子表格来执行各种计算任务，那么他们可能需要一分钟或者一小时来运行，Pandas 将会改变你的生活。我甚至已经看到机器学习的版本，如 K-Means 聚类在 Excel 上完成。这真的很酷，但是我的 Python 会为你做得更快，这也将使你对参数要求更严格，拥有更大的数据集，并且能够完成更多的工作。

还有一个好消息。你可以很容易加载和输出`xls`或`xlsx`格式的文件，所以，即使你的老板想用旧的方式来查看，他们也可以。Pandas 还可以兼容文本文件，`csv`，`hdf`文件，`xml`，`html`等等，其 IO 非常强大。

如果您刚刚入门 Python，那么您应该可以继续学习，而不必精通 Python，这甚至可以让你入门 Python 。最重要的是，如果你有问题，问问他们！如果你为每一个困惑的领域寻找答案，并为此做好每件事，那么最终你会有一个完整的认识。你的大部分问题都可以通过 Google 解决。不要害怕 Google 你的问题，它不会嘲笑你，我保证。我仍然 Google 了我的很多目标，看看是否有人有一些示例代码，做了我想做的事情，所以不要仅仅因为你这样做了，而觉得你是个新手。

如果我还没有把 Pandas 推销给你，那么电梯演讲就是：电子表格式数据的闪电般的数据分析，具有非常强大的输入/输出机制，可以处理多种数据类型，甚至可以转换数据类型。

好的，你被推销了。现在让我们获取 Pandas！首先，我将假设有些人甚至还没有 Python。到目前为止，最简单的选择是使用预编译的 Python 发行版，比如 ActivePython，它是个快速简单的方式，将数据科学所需的所有包和依赖关系都集中在一起，而不需要一个接一个安装它们，特别是在 64 位 Windows 上。我建议获取最新版本的 64 位 Python。仅在这个系列中，我们使用 Pandas ，它需要 Numpy。我们还将使用 Matplotlib 和 Scikit-Learn，所有这些都是 ActivePython 自带的，预先编译和优化的 MKL。你可以从这里下载一个配置完整的 Python 发行版。

如果您想手动安装 Python，请转到`Python.org`，然后下载 Python 3+ 或更高版本。不要仅仅获取`2.X`。记下你下载的位版本。因为你的操作系统是 64 位的，这并是你的 Python 版本，默认总是 32 位。选择你想要的。 64 位可能有点头疼，所以如果你是新手，我不会推荐它，但 64 位是数据科学的理想选择，所以你不会被锁定在最大 2GB 的 RAM 上。如果你想装 64 位，查看`pip`安装教程可能有帮助，其中介绍了如何处理常规安装以及更棘手的 64 位软件包。如果你使用 32 位，那么现在不用担心这个教程。

所以你已经安装了 Python。接下来，转到您的终端或`cmd.exe`，然后键入：`pip install pandas`。你有没有得到`pip is not a recognized command`或类似的东西？没问题，这意味着`pip`不在你的`PATH`中。`pip`是一个程序，但是你的机器不知道它在哪里，除非它在你的`PATH`中。如果你愿意，你可以搜索如何添加一些东西到你的`PATH`中，但是你总是可以显式提供你想要执行的程序的路径。例如，在 Windows 上，Python 的`pip`位于`C:/Python34/Scripts/pip`中。 `Python34`的意思是 Python 3.4。如果你拥有 Python 3.6，那么你需要使用`Python36`，以此类推。

因此，如果常规的`pip install pandas`不起作用，那么你可以执行`C:/Python34/Scripts/pip install pandas`。

到了这里，人们争论的另一个重点是他们选择的编辑器。编辑器在事物的宏观层面中并不重要。你应该尝试多个编辑器，并选择最适合你的编辑器。无论哪个，只要你感到舒适，而且你的工作效率很高，这是最重要的。一些雇主也会迫使你最终使用编辑器 X，Y 或 Z，所以你可能不应该依赖编辑器功能。因此，我更喜欢简单的 IDLE，这就是我将用于编程的东西。再次，您可以在 Wing，emacs，Nano，Vim，PyCharm，IPython 中编程，你可以随便选一个。要打开 IDLE，只需访问开始菜单，搜索 IDLE，然后选择它。在这里，`File > New`，砰的一下，你就有了带高亮的文本编辑器和其他一些小东西。我们将在进行中介绍一些这些次要的事情。

现在，无论您使用哪种编辑器，都可以打开它，让我们编写一些简单的代码来查看数据帧。

通常，`DataFrame`最接近 Python `Dictionary` 数据结构。如果你不熟悉字典，这里有一个教程。我将在视频中注明类似的东西，并且在描述中，以及在`PythonProgramming.net`上的文本版教程中有链接。

首先，我们来做一些简单的导入：

```py
import pandas as pd
import datetime
import pandas.io.data as web
```

在这里，我们将`pandas`导入为`pd`。 这只是导入`pandas`模块时使用的常用标准。 接下来，我们导入`datetime`，我们稍后将使用它来告诉 Pandas 一些日期，我们想要拉取它们之间的数据。 最后，我们将`pandas.io.data`导入为`web`，因为我们将使用它来从互联网上获取数据。 接下来：

```py
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 8, 22)
```
在这里，我们创建`start`和`end`变量，这些变量是`datetime`对象，获取 2010 年 1 月 1 日到 2015 年 8 月 22 日的数据。现在，我们可以像这样创建数据帧：

```py
df = web.DataReader("XOM", "yahoo", start, end)
```

这从雅虎财经 API 获取 Exxon 的数据，存储到我们的`df`变量。 将你的数据帧命名为`df`不是必需的，但是它页是用于 Pandas 的非常主流的标准。 它只是帮助人们立即识别活动数据帧，而无需追溯代码。

所以这给了我们一个数据帧，我们怎么查看它？ 那么，可以打印它，就像这样：

```py
print(df)
```

所以这是很大一个空间。 数据集的中间被忽略，但仍然是大量输出。 相反，大多数人只会这样做：

```py
print(df.head())
```

输出：

```py
                 Open       High        Low      Close    Volume  Adj Close
Date                                                                       
2010-01-04  68.720001  69.260002  68.190002  69.150002  27809100  59.215446
2010-01-05  69.190002  69.449997  68.800003  69.419998  30174700  59.446653
2010-01-06  69.449997  70.599998  69.339996  70.019997  35044700  59.960452
2010-01-07  69.900002  70.059998  69.419998  69.800003  27192100  59.772064
2010-01-08  69.690002  69.750000  69.220001  69.519997  24891800  59.532285
```

这打印了数据帧的前 5 行，并且对于调试很有用，只查看了数据帧的外观。 当你执行分析等，看看你想要的东西是否实际发生了，就很有用。 不过，我们稍后会深入它。

我们可以在这里停止介绍，但还有一件事：数据可视化。 我之前说过，Pandas 和其他模块配合的很好，Matplotlib 就是其中之一。 让我们来看看！ 打开你的终端或`cmd.exe`，并执行`pip install matplotlib`。 你安装完 Pandas，我确信你应该已经获取了它，但我们要证实一下。 现在，在脚本的顶部，和其他导入一起，添加：

```py
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
```

Pyplot 是 matplotlib 的基本绘图模块。 Style  帮助我们快速美化图形，`style.use`让我们选择风格。 有兴趣了解 Matplotlib 的更多信息吗？ 查看 Matplotlib 的深入系列教程！

接下来，在我们的`print(df.head())`下方，我们可以执行如下操作：

```py
df['High'].plot()
plt.legend()
plt.show()
```

![](https://pythonprogramming.net/static/images/pandas/pandas-graph-example.png)

很酷！ 这里有个 pandas  的快速介绍，但一点也不可用。 在这个系列中，我们将会涉及更多 Pandas 的基础知识，然后转到导航和处理数据帧。 从这里开始，我们将更多地介绍可视化，多种数据格式的输入和输出，基本和进阶数据分析和操作，合并和组合数据帧，重复取样等等。

如果您迷茫，困惑，或需要澄清，请不要犹豫，给对应的视频提问。

## 二、Pandas 基础

在这个 Python 和 Pandas 数据分析教程中，我们将弄清一些 Pandas 的基础知识。 加载到 Pandas 数据帧之前，数据可能有多种形式，但通常需要是以行和列组成的数据集。 所以也许是这样的字典：

```py
web_stats = {'Day':[1,2,3,4,5,6],
             'Visitors':[43,34,65,56,29,76],
             'Bounce Rate':[65,67,78,65,45,52]}
```

我们可以将这个字典转换成数据帧，通过这样：

```py
import pandas as pd

web_stats = {'Day':[1,2,3,4,5,6],
             'Visitors':[43,34,65,56,29,76],
             'Bounce Rate':[65,67,78,65,45,52]}

df = pd.DataFrame(web_stats)
```

现在我们可以做什么？之前看到，你可以通过这样来查看简单的起始片段：

```py
print(df.head())
   Bounce Rate  Day  Visitors
0           65    1        43
1           67    2        34
2           78    3        65
3           65    4        56
4           45    5        29
```

你也可以查看后几行。为此，你需要这样做：

```py
print(df.tail())
   Bounce Rate  Day  Visitors
1           67    2        34
2           78    3        65
3           65    4        56
4           45    5        29
5           52    6        76
```

最后，你也可以传入头部和尾部数量，像这样：

```py
print(df.tail(2))
   Bounce Rate  Day  Visitors
4           45    5        29
5           52    6        76
```

你可以在这里看到左边有这些数字，`0,1,2,3,4,5`等等，就像行号一样。 这些数字实际上是你的“索引”。 数据帧的索引是数据相关，或者数据按它排序的东西。 一般来说，这将是连接所有数据的变量。 这里，我们从来没有为此目的定义任何东西，知道这个变量是什么，对于 Pandas 是个挑战。 因此，当你没有定义索引时，Pandas 会像这样为你生成一个。 现在看数据集，你能看到连接其他列的列吗？

`Day`列适合这个东西！ 一般来说，如果您有任何日期数据，日期将成为“索引”，因为这就是所有数据点的关联方式。 有很多方法可以识别索引，更改索引等等。 我们将在这里介绍一些。 首先，在任何现有的数据帧上，我们可以像这样设置一个新的索引：

```py
df.set_index('Day', inplace=True)
```

输出：

```py
     Bounce Rate  Visitors
Day                       
1             65        43
2             67        34
3             78        65
4             65        56
5             45        29
```

现在你可以看到这些行号已经消失了，同时也注意到`Day`比其他列标题更低，这是为了表示索引。 有一点需要注意的是`inplace = True`的使用。 这允许我们原地修改数据帧，意味着我们实际上修改了变量本身。 没有`inplace = True`，我们需要做一些事情：

```py
df = df.set_index('Day')
```

您也可以设置多个索引，但这是以后的更高级的主题。 你可以很容易做到这一点，但它的原因相当合理。

一旦你有了合理的索引，是一个日期时间或数字，那么它将作为一个 X 轴。 如果其他列也是数值数据，那么您可以轻松绘图。 就像我们之前做的那样，继续并执行：

```py
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
```

然后，在底部，你可以绘图。 还记得我们之前引用了特定的列嘛？也许你注意到了，但是我们可以像这样，引用数据帧中的特定项目：

```py
print(df['Visitors'])
Day
1    43
2    34
3    65
4    56
5    29
6    76
Name: Visitors, dtype: int64
```

您也可以像对象一样引用数据帧的部分，只要没有空格，就可以这样做：

```py
print(df.Visitors)
Day
1    43
2    34
3    65
4    56
5    29
6    76
Name: Visitors, dtype: int64
```

所以我们可以像这样绘制单列：

```py
df['Visitors'].plot()
plt.show()
```

我们也可以绘制整个数据帧。 只要数据是规范化的或者在相同的刻度上，效果会很好。 这是一个例子：

```py
df.plot()
plt.show()
```

注意图例如何自动添加。 你可能会喜欢的另一个很好的功能是，图例也自动为实际绘制的直线让路。 如果你是 Python 和 Matplotlib 的新手，这可能对你来说并不重要，但这不是一个正常的事情。

最后，在我们离开之前，你也可以一次引用多个列，就像这样（我们只有两列，但是多列相同）：

```py
print(df[['Visitors','Bounce Rate']])
```

所以这是括起来的列标题列表。 你也可以绘制这个。

这些是一些方法，您可以直接与数据帧进行交互，引用数据框的各个方面，带有一个示例，绘制了这些特定的方面。

## 三、IO 基础

欢迎阅读 Pandas 和 Python 数据分析第三部分。在本教程中，我们将开始讨论 Pandas IO 即输入/输出，并从一个实际的用例开始。为了得到充分的实践，一个非常有用的网站是 Quandl。 Quandl 包含大量的免费和付费数据源。这个站点的好处在于数据通常是标准化的，全部在一个地方，提取数据的方法是一样的。如果您使用的是 Python，并且通过它们的简单模块访问 Quandl 数据，那么数据将自动以数据帧返回。出于本教程的目的，我们将仅仅出于学习的目的而手动下载一个 CSV 文件，因为并不是每个数据源都会有一个完美的模块用于提取数据集。

假设我们有兴趣，在德克萨斯州的奥斯汀购买或出售房屋。那里的邮政编码是 77006。我们可以访问当地的房源清单，看看目前的价格是多少，但这并不能真正为我们提供任何真实的历史信息，所以我们只是试图获得一些数据。让我们来查询“房屋价值指数 77006”。果然，我们可以在这里看到一个索引。有顶层，中层，下层，三居室，等等。比方说，当然，我们有一个三居室的房子。我们来检查一下。原来 Quandl 已经提供了图表，但是我们还是要抓取数据集，制作自己的图表，或者做一些其他的分析。访问“下载”，并选择 CSV。Pandas 的 IO 兼容 csv，excel 数据，hdf，sql，json，msgpack，html，gbq，stata，剪贴板和 pickle 数据，并且列表不断增长。查看 IO 工具文档的当前列表。将该 CSV 文件移动到本地目录（您正在使用的目录/这个`.py`脚本所在的目录）。

以这个代码开始，将 CSV 加载进数据帧就是这样简单：

```py
import pandas as pd

df = pd.read_csv('ZILL-Z77006_3B.csv')
print(df.head())
```

输出：

```py
         Date   Value
0  2015-06-30  502300
1  2015-05-31  501500
2  2015-04-30  500100
3  2015-03-31  495800
4  2015-02-28  492700
```

注意我们又没有了合适的索引。我们可以首先这样做来修复：

```py
df.set_index('Date', inplace = True)
```

现在，让我们假设，我们打算将它转回 CSV，我们可以：

```py
df.to_csv('newcsv2.csv')
```

我们仅仅有了一列，但是如果你有很多列，并且仅仅打算转换一列，你可以：

```py
df['Value'].to_csv('newcsv2.csv')
```

要记住我们如何绘制多列，但是并不是所有列。看看你能不能猜出如何保存多列，但不是所有列。

现在，让我们读取新的 CSV：

```py
df = pd.read_csv('newcsv2.csv')
print(df.head())
```

输出：

```py
         Date   Value
0  2015-06-30  502300
1  2015-05-31  501500
2  2015-04-30  500100
3  2015-03-31  495800
4  2015-02-28  492700
```

该死，我们的索引又没了！ 这是因为 CSV 没有像我们的数据帧那样的“索引”属性。 我们可以做的是，在导入时设置索引，而不是导入之后设置索引。 像这样：

```py
df = pd.read_csv('newcsv2.csv', index_col=0)
print(df.head())
```

输出：

```py
             Value
Date              
2015-06-30  502300
2015-05-31  501500
2015-04-30  500100
2015-03-31  495800
2015-02-28  492700
```

现在，我不了解你，但“价值”这个名字是毫无价值的。 我们可以改变这个吗？ 当然，有很多方法来改变列名，一种方法是：

```py
df.columns = ['House_Prices']
print(df.head())
```

输出：

```py
            House_Prices
Date                    
2015-06-30        502300
2015-05-31        501500
2015-04-30        500100
2015-03-31        495800
2015-02-28        492700
```
下面，我们可以尝试这样保存为 CSV：

```py
df.to_csv('newcsv3.csv')
```

如果你看看 CSV，你应该看到它拥有标题。如果不想要标题怎么办呢？没问题！

```py
df.to_csv('newcsv4.csv', header=False)
```

如果文件没有标题呢？没问题！

```py
df = pd.read_csv('newcsv4.csv', names = ['Date','House_Price'], index_col=0)
print(df.head())
```

输出：

```py
            House_Price
Date                   
2015-06-30       502300
2015-05-31       501500
2015-04-30       500100
2015-03-31       495800
2015-02-28       492700
```

这些是IO的基本知识，在输入和输出时有一些选项。

一个有趣的事情是使用 Pandas 进行转换。 所以，也许你是从 CSV 输入数据，但你真的希望在你的网站上，将这些数据展示为 HTML。 由于 HTML 是数据类型之一，我们可以将其导出为 HTML，如下所示：

```py
df.to_html('example.html')
```

现在我们有了 HTML 文件。打开它，然后你就有了 HTML 中的一个表格：

|  | House_Prices |
| --- | --- |
| Date |  |
| 2015-06-30 | 502300 |
| 2015-05-31 | 501500 |
| 2015-04-30 | 500100 |
| 2015-03-31 | 495800 |
| 2015-02-28 | 492700 |
| 2015-01-31 | 493000 |
| 2014-12-31 | 494200 |
| 2014-11-30 | 490900 |
| 2014-10-31 | 486000 |
| 2014-09-30 | 479800 |
| 2014-08-31 | 473900 |
| 2014-07-31 | 467100 |
| 2014-06-30 | 461400 |
| 2014-05-31 | 455400 |
| 2014-04-30 | 450500 |
| 2014-03-31 | 450300 |

注意，这个表自动分配了`dataframe`类。 这意味着你可以自定义 CSS 来处理数据帧特定的表！

当我有用数据的 SQL 转储时，我特别喜欢使用 Pandas。 我倾向于将数据库数据直接倒入 Pandas 数据帧中，执行我想要执行的操作，然后将数据显示在图表中，或者以某种方式提供数据。

最后，如果我们想重新命名其中一列，该怎么办？ 之前，你已经看到了如何命名所有列，但是也许你只是想改变一个列，而不必输入所有的列。 足够简单：

```py
print(df.head())

df.rename(columns={'House_Price':'Prices'}, inplace=True)
print(df.head())
```

输出：

```py
         Date  House_Price
0  2015-06-30       502300
1  2015-05-31       501500
2  2015-04-30       500100
3  2015-03-31       495800
4  2015-02-28       492700
         Date  Prices
0  2015-06-30  502300
1  2015-05-31  501500
2  2015-04-30  500100
3  2015-03-31  495800
4  2015-02-28  492700
```

所以在这里，我们首先导入了无头文件，提供了列名`Date`和`House_Price`。 然后，我们决定，我们打算用`Price`代替`House_Price`。 因此，我们使用`df.rename`，指定我们要重命名的列，然后在字典形式中，键是原始名称，值是新名称。 我们最终使用`inplace = True`，以便修改原始对象。

## 四、构件数据集

在 Python 和 Pandas 数据分析系列教程的这一部分中，我们将扩展一些东西。让我们想想，我们是亿万富豪，还是千万富豪，但成为亿万富豪则更有趣，我们正在努力使我们的投资组合尽可能多样化。我们希望拥有所有类型的资产类别，所以我们有股票，债券，也许是一个货币市场帐户，现在我们正在寻找坚实的不动产。你们都看过广告了吗？你买了 60 美元的 CD，参加了 500 美元的研讨会，你开始把你的 6 位数字投资到房地产，对吧？

好吧，也许不是，但是我们肯定要做一些研究，并有一些购买房地产的策略。那么，什么统治了房价，我们是否需要进行研究才能找到答案？一般来说，不，你并不需要那么做，我们知道这些因素。房价的因素受经济，利率和人口统计的影响。这是房地产价格总体上的三大影响。现在当然，如果你买土地，其他的事情很重要，它的水平如何，我们是否需要在土地上做一些工作，才能真正奠定基础，如何排水等等。那么我们还有更多的因素，比如屋顶，窗户，暖气/空调，地板，地基等等。我们可以稍后考虑这些因素，但首先我们要从宏观层面开始。你会看到我们的数据集在这里膨胀得有多快，它会爆炸式增长。

所以，我们的第一步是收集数据。 Quandl 仍然是良好的起始位置，但是这一次让我们自动化数据抓取。我们将首先抓取 50 个州的住房数据，但是我们也试图收集其他数据。我们绝对不想手动抓取这个数据。首先，如果你还没有帐户，你需要得到一个帐户。这将给你一个 API 密钥和免费数据的无限的 API 请求，这真棒。

一旦你创建了一个账户，访问`your account / me`，不管他们这个时候叫什么，然后找到标有 API 密钥的部分。这是你所需的密钥。接下来，我们要获取 Quandl 模块。我们实际上并不需要模块来生成请求，但它是一个非常小的模块，他能给我们带来一些小便利，所以不妨试试。打开你的终端或`cmd.exe`并且执行`pip install quandl`（再一次，如果`pip`不能识别，记得指定`pip`的完整路径）。

接下来，我们做好了开始的准备，打开一个新的编辑器。开始：

```py
import Quandl

# Not necessary, I just do this so I do not show my API key.
api_key = open('quandlapikey.txt','r').read()

df = Quandl.get("FMAC/HPI_TX", authtoken=api_key)

print(df.head())
```

如果你愿意的话，你可以只存储你的密钥的纯文本版本，我只隐藏了我的密钥，因为它是我发布的教程。这是我们需要做的，来获得德克萨斯州的房价指数。我们抓取的实际指标可以在任何页面上找到，无论你什么时候访问，只要在网站上点击你使用的库，我们这里是 Python，然后需要输入的查询就会弹出。

随着您的数据科学事业的发展，您将学习到各种常数，因为人们是合乎逻辑和合理的。我们这里，我们需要获取所有州的数据。我们如何做到呢？我们是否需要手动抓取每个指标？不，看看这个代码，我们看到`FMAC/HPI_TX`。我们可以很容易地把这个解码为`FMAC = Freddie Mac`。 `HPI = House Price Index`（房价指数）。`TX`是德克萨斯州，它的常用两字母缩写。从这里，我们可以安全地假设所有的代码都是这样构建的，所以现在我们只需要一个州缩写的列表。我们搜索它，作出选择，就像这个 50 个州的列表。怎么办？

我们可以通过多种方式提取这些数据。这是一个 Pandas 教程，所以如果我们可以 Pandas 熊猫，我们就这样。让我们来看看 Pandas 的`read_html`。它不再被称为“实验性”的，但我仍然会将其标记为实验性的。其他 IO 模块的标准和质量非常高并且可靠。`read_html`并不是很好，但我仍然说这是非常令人印象深刻有用的代码，而且很酷。它的工作方式就是简单地输入一个 URL，Pandas 会从表中将有价值的数据提取到数据帧中。这意味着，与其他常用的方法不同，`read_html`最终会读入一些列数据帧。这不是唯一不同点，但它是不同的。首先，为了使用`read_html`，我们需要`html5lib`。打开`cmd.exe`或您的终端，并执行：`pip install html5lib`。现在，我们可以做我们的第一次尝试：

```py
fiddy_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
print(fiddy_states)
```

它的输出比我要在这里发布的更多，但你明白了。 这些数据中至少有一部分是我们想要的，看起来第一个数据帧是一个很好的开始。 那么让我们执行：

```py
print(fiddy_states[0])
               0               1               2                  3
0   Abbreviation      State name         Capital     Became a state
1             AL         Alabama      Montgomery  December 14, 1819
2             AK          Alaska          Juneau    January 3, 1959
3             AZ         Arizona         Phoenix  February 14, 1912
4             AR        Arkansas     Little Rock      June 15, 1836
5             CA      California      Sacramento  September 9, 1850
6             CO        Colorado          Denver     August 1, 1876
7             CT     Connecticut        Hartford    January 9, 1788
8             DE        Delaware           Dover   December 7, 1787
9             FL         Florida     Tallahassee      March 3, 1845
10            GA         Georgia         Atlanta    January 2, 1788
11            HI          Hawaii        Honolulu    August 21, 1959
12            ID           Idaho           Boise       July 3, 1890
13            IL        Illinois     Springfield   December 3, 1818
14            IN         Indiana    Indianapolis  December 11, 1816
15            IA            Iowa      Des Moines  December 28, 1846
16            KS          Kansas          Topeka   January 29, 1861
17            KY        Kentucky       Frankfort       June 1, 1792
18            LA       Louisiana     Baton Rouge     April 30, 1812
19            ME           Maine         Augusta     March 15, 1820
20            MD        Maryland       Annapolis     April 28, 1788
21            MA   Massachusetts          Boston   February 6, 1788
22            MI        Michigan         Lansing   January 26, 1837
23            MN       Minnesota      Saint Paul       May 11, 1858
24            MS     Mississippi         Jackson  December 10, 1817
25            MO        Missouri  Jefferson City    August 10, 1821
26            MT         Montana          Helena   November 8, 1889
27            NE        Nebraska         Lincoln      March 1, 1867
28            NV          Nevada     Carson City   October 31, 1864
29            NH   New Hampshire         Concord      June 21, 1788
30            NJ      New Jersey         Trenton  December 18, 1787
31            NM      New Mexico        Santa Fe    January 6, 1912
32            NY        New York          Albany      July 26, 1788
33            NC  North Carolina         Raleigh  November 21, 1789
34            ND    North Dakota        Bismarck   November 2, 1889
35            OH            Ohio        Columbus      March 1, 1803
36            OK        Oklahoma   Oklahoma City  November 16, 1907
37            OR          Oregon           Salem  February 14, 1859
38            PA    Pennsylvania      Harrisburg  December 12, 1787
39            RI    Rhode Island      Providence       May 19, 1790
40            SC  South Carolina        Columbia       May 23, 1788
41            SD    South Dakota          Pierre   November 2, 1889
42            TN       Tennessee       Nashville       June 1, 1796
43            TX           Texas          Austin  December 29, 1845
44            UT            Utah  Salt Lake City    January 4, 1896
45            VT         Vermont      Montpelier      March 4, 1791
46            VA        Virginia        Richmond      June 25, 1788
47            WA      Washington         Olympia  November 11, 1889
48            WV   West Virginia      Charleston      June 20, 1863
49            WI       Wisconsin         Madison       May 29, 1848
50            WY         Wyoming        Cheyenne      July 10, 1890
```

是的，这看起来不错，我们想要第零列。所以，我们要遍历`fiddy_states[0]`的第零列。 请记住，现在`fiddy_states`是一个数帧列表，而`fiddy_states[0]`是第一个数据帧。 为了引用第零列，我们执行`fiddy_states[0][0]`。 一个是列表索引，它返回一个数据帧。 另一个是数据帧中的一列。 接下来，我们注意到第零列中的第一项是`abbreviation`，我们不想要它。 当我们遍历第零列中的所有项目时，我们可以使用`[1:]`排除掉它。 因此，我们的缩写列表是`fiddy_states[0][0][1:]`，我们可以像这样迭代：

```py
for abbv in fiddy_states[0][0][1:]:
    print(abbv)
AL
AK
AZ
AR
CA
CO
CT
DE
FL
GA
HI
ID
IL
IN
IA
KS
KY
LA
ME
MD
MA
MI
MN
MS
MO
MT
NE
NV
NH
NJ
NM
NY
NC
ND
OH
OK
OR
PA
RI
SC
SD
TN
TX
UT
VT
VA
WA
WV
WI
WY
```

完美！ 现在，我们回忆这样做的原因：我们正在试图用州名缩写建立指标，来获得每个州的房价指数。 好的，我们可以建立指标：

```py
for abbv in fiddy_states[0][0][1:]:
    #print(abbv)
    print("FMAC/HPI_"+str(abbv))
FMAC/HPI_AL
FMAC/HPI_AK
FMAC/HPI_AZ
FMAC/HPI_AR
FMAC/HPI_CA
FMAC/HPI_CO
FMAC/HPI_CT
FMAC/HPI_DE
FMAC/HPI_FL
FMAC/HPI_GA
FMAC/HPI_HI
FMAC/HPI_ID
FMAC/HPI_IL
FMAC/HPI_IN
FMAC/HPI_IA
FMAC/HPI_KS
FMAC/HPI_KY
FMAC/HPI_LA
FMAC/HPI_ME
FMAC/HPI_MD
FMAC/HPI_MA
FMAC/HPI_MI
FMAC/HPI_MN
FMAC/HPI_MS
FMAC/HPI_MO
FMAC/HPI_MT
FMAC/HPI_NE
FMAC/HPI_NV
FMAC/HPI_NH
FMAC/HPI_NJ
FMAC/HPI_NM
FMAC/HPI_NY
FMAC/HPI_NC
FMAC/HPI_ND
FMAC/HPI_OH
FMAC/HPI_OK
FMAC/HPI_OR
FMAC/HPI_PA
FMAC/HPI_RI
FMAC/HPI_SC
FMAC/HPI_SD
FMAC/HPI_TN
FMAC/HPI_TX
FMAC/HPI_UT
FMAC/HPI_VT
FMAC/HPI_VA
FMAC/HPI_WA
FMAC/HPI_WV
FMAC/HPI_WI
FMAC/HPI_WY
```

我们已经得到了指标，现在我们已经准备好提取数据帧了。 但是，一旦我们拿到他们，我们会做什么？ 我们将使用 50 个独立的数据帧？ 听起来像一个愚蠢的想法，我们需要一些方法来组合他们。 Pandas 背后的优秀人才看到了这一点，并为我们提供了多种组合数据帧的方法。 我们将在下一个教程中讨论这个问题。

## 五、连接和附加数据帧

欢迎阅读 Python 和 Pandas 数据分析系列教程第五部分。在本教程中，我们将介绍如何以各种方式组合数据帧。

在我们的房地产投资案例中，我们希望使用房屋数据获取 50 个数据帧，然后把它们全部合并成一个数据帧。我们这样做有很多原因。首先，将这些组合起来更容易，更有意义，也会减少使用的内存。每个数据帧都有日期和值列。这个日期列在所有数据帧中重复出现，但实际上它们应该全部共用一个，实际上几乎减半了我们的总列数。

在组合数据帧时，您可能会考虑相当多的目标。例如，你可能想“附加”到他们，你可能会添加到最后，基本上就是添加更多的行。或者，也许你想添加更多的列，就像我们的情况一样。有四种主要的数据帧组合方式，我们现在开始介绍。四种主要的方式是：连接（Concatenation），连接（Join），合并和附加。我们将从第一种开始。这里有一些初始数据帧：

```py
df1 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2001, 2002, 2003, 2004])

df2 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'US_GDP_Thousands':[50, 55, 65, 55]},
                   index = [2005, 2006, 2007, 2008])

df3 = pd.DataFrame({'HPI':[80,85,88,85],
                    'Int_rate':[2, 3, 2, 2],
                    'Low_tier_HPI':[50, 52, 50, 53]},
                   index = [2001, 2002, 2003, 2004])
```

注意这些之间有两个主要的变化。 `df1`和`df3`具有相同的索引，但它们有一些不同的列。 `df2`和`df3`有不同的索引和一些不同的列。 通过连接（concat），我们可以讨论将它们结合在一起的各种方法。 我们来试一下简单的连接（concat）：

```py
concat = pd.concat([df1,df2])
print(concat)


      HPI  Int_rate  US_GDP_Thousands
2001   80         2                50
2002   85         3                55
2003   88         2                65
2004   85         2                55
2005   80         2                50
2006   85         3                55
2007   88         2                65
2008   85         2                55
```

很简单。 这两者之间的主要区别仅仅是索引的延续，但是它们共享同一列。 现在他们已经成为单个数据帧。 然而我们这里，我们对添加列而不是行感到好奇。 当我们将一些共有的和一些新列组合起来：

```py
concat = pd.concat([df1,df2,df3])
print(concat)


      HPI  Int_rate  Low_tier_HPI  US_GDP_Thousands
2001   80         2           NaN                50
2002   85         3           NaN                55
2003   88         2           NaN                65
2004   85         2           NaN                55
2005   80         2           NaN                50
2006   85         3           NaN                55
2007   88         2           NaN                65
2008   85         2           NaN                55
2001   80         2            50               NaN
2002   85         3            52               NaN
2003   88         2            50               NaN
2004   85         2            53               NaN
```

不错，我们有一些`NaN`（不是数字），因为那个索引处不存在数据，但是我们所有的数据确实在这里。

这些就是基本的连接（concat），接下来，我们将讨论附加。 附加就像连接的第一个例子，只是更加强大一些，因为数据帧会简单地追加到行上。 我们通过一个例子来展示它的工作原理，同时也展示它可能出错的地方：

```py
df4 = df1.append(df2)
print(df4)


      HPI  Int_rate  US_GDP_Thousands
2001   80         2                50
2002   85         3                55
2003   88         2                65
2004   85         2                55
2005   80         2                50
2006   85         3                55
2007   88         2                65
2008   85         2                55
```

这就是我们期望的附加。 在大多数情况下，你将要做这样的事情，就像在数据库中插入新行一样。 我们并没有真正有效地附加数据帧，它们更像是根据它们的起始数据来操作，但是如果你需要，你可以附加。 当我们附加索引相同的数据时会发生什么？

```py
df4 = df1.append(df3)
print(df4)


      HPI  Int_rate  Low_tier_HPI  US_GDP_Thousands
2001   80         2           NaN                50
2002   85         3           NaN                55
2003   88         2           NaN                65
2004   85         2           NaN                55
2001   80         2            50               NaN
2002   85         3            52               NaN
2003   88         2            50               NaN
2004   85         2            53               NaN
```

好吧，这很不幸。 有人问为什么连接（concat ）和附加都退出了。 这就是原因。 因为共有列包含相同的数据和相同的索引，所以组合这些数据帧要高效得多。 一个另外的例子是附加一个序列。 鉴于`append`的性质，您可能会附加一个序列而不是一个数据帧。 至此我们还没有谈到序列。 序列基本上是单列的数据帧。 序列确实有索引，但是，如果你把它转换成一个列表，它将仅仅是这些值。 每当我们调用`df ['column']`时，返回值就是一个序列。

```py
s = pd.Series([80,2,50], index=['HPI','Int_rate','US_GDP_Thousands'])
df4 = df1.append(s, ignore_index=True)
print(df4)

   HPI  Int_rate  US_GDP_Thousands
0   80         2                50
1   85         3                55
2   88         2                65
3   85         2                55
4   80         2                50
```

在附加序列时，我们必须忽略索引，因为这是规则，除非序列拥有名称。

在这里，我们已经介绍了 Pandas 中的连接（concat）和附加数据帧。 接下来，我们将讨论如何连接（join）和合并数据帧。
