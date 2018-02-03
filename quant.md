# Python 金融教程

## 一、入门和获取股票数据

您好，欢迎来到 Python 金融系列教程。在本系列中，我们将使用 Pandas 框架来介绍将金融（股票）数据导入 Python 的基础知识。从这里开始，我们将操纵数据，试图搞出一些公司的投资系统，应用一些机器学习，甚至是一些深度学习，然后学习如何回溯测试一个策略。我假设你知道 Python 基础。如果您不确定，请点击基础链接，查看系列中的一些主题，并进行判断。如果在任何时候你卡在这个系列中，或者对某个主题或概念感到困惑，请随时寻求帮助，我将尽我所能提供帮助。

我被问到的一个常见问题是，我是否使用这些技术投资或交易获利。我主要是为了娱乐，并且练习数据分析技巧而玩财务数据，但实际上这也影响了我今天的投资决策。在写这篇文章的时候，我并没有用编程来进行实时算法交易，但是我已经有了实际的盈利，但是在算法交易方面还有很多工作要做。最后，如何操作和分析财务数据，以及如何测试交易状态的知识已经为我节省了大量的金钱。

这里提出的策略都不会使你成为一个超富有的人。如果他们愿意，我可能会把它们留给自己！然而，知识本身可以为你节省金钱，甚至可以使你赚钱。

好吧，让我们开始吧。首先，我正在使用 Python 3.5，但你应该能够获取更高版本。我会假设你已经安装了Python。如果你没有 64 位的 Python，但有 64 位的操作系统，去获取 64 位的 Python，稍后会帮助你。如果你使用的是 32 位操作系统，那么我对你的情况感到抱歉，不过你应该没问题。

用于启动的所需模块：

1.  NumPy
1.  Matplotlib
1.  Pandas
1.  Pandas-datareader
1.  BeautifulSoup4
1.  scikit-learn / sklearn

这些是现在做的，我们会在其他模块出现时处理它们。 首先，让我们介绍一下如何使用 pandas，matplotlib 和 Python 处理股票数据。

如果您想了解 Matplotlib 的更多信息，请查看 Matplotlib 数据可视化系列教程。

如果您想了解 Pandas 的更多信息，请查看 Pandas 数据分析系列教程。

首先，我们将执行以下导入：

```py
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
```

`Datetime`让我们很容易处理日期，`matplotlib`用于绘图，Pandas 用于操纵数据，`pandas_datareader`是我写这篇文章时最新的 Pandas io 库。

现在进行一些启动配置：

```py
style.use('ggplot')

start = dt.datetime(2000, 1, 1)
end = dt.datetime(2016, 12, 31)
```

我们正在设置一个风格，所以我们的图表看起来并不糟糕。 在金融领域，即使你亏本，你的图表也是非常重要的。 接下来，我们设置一个开始和结束`datetime `对象，这将是我们要获取股票价格信息的日期范围。

现在，我们可以从这些数据中创建一个数据帧：

```py
df = web.DataReader('TSLA', "yahoo", start, end)
```

如果您目前不熟悉`DataFrame`对象，可以查看 Pandas 的教程，或者只是将其想象为电子表格或存储器/ RAM 中的数据库表。 这只是一些行和列，并带有一个索引和列名乘。 在我们的这里，我们的索引可能是日期。 索引应该是与所有列相关的东西。

`web.DataReader('TSLA', "yahoo", start, end)`这一行，使用`pandas_datareader`包，寻找股票代码`TSLA`（特斯拉），从 yahoo 获取信息，从我们选择的起始和结束日期起始或结束。 以防你不知道，股票是公司所有权的一部分，代码是用来在证券交易所引用公司的“符号”。 大多数代码是 1-4 个字母。

所以现在我们有一个`Pandas.DataFrame`对象，它包含特斯拉的股票交易信息。 让我们看看我们在这里有啥：

```py
print(df.head())
```

```
                 Open   High        Low      Close    Volume  Adj Close
Date                                                                   
2010-06-29  19.000000  25.00  17.540001  23.889999  18766300  23.889999
2010-06-30  25.790001  30.42  23.299999  23.830000  17187100  23.830000
2010-07-01  25.000000  25.92  20.270000  21.959999   8218800  21.959999
2010-07-02  23.000000  23.10  18.709999  19.200001   5139800  19.200001
2010-07-06  20.000000  20.00  15.830000  16.110001   6866900  16.110001
```

`.head()`是可以用`Pandas DataFrames`做的事情，它会输出前`n`行​​，其中`n`是你传递的可选参数。如果不传递参数，则默认值为 5。我们绝对会使用`.head()`来快速浏览一下我们的数据，以确保我们在正路上。看起来很棒！

以防你不知道：

+   开盘价 - 当股市开盘交易时，一股的价格是多少？
+   最高价 - 在交易日的过程中，那一天的最高价是多少？
+   最低价 - 在交易日的过程中，那一天的最低价是多少？
+   收盘价 - 当交易日结束时，最终的价格是多少？
+   成交量 - 那一天有多少股交易？

调整收盘价 - 这一个稍微复杂一些，但是随着时间的推移，公司可能决定做一个叫做股票拆分的事情。例如，苹果一旦股价超过 1000 美元就做了一次。由于在大多数情况下，人们不能购买股票的一小部分，股票价格 1000 美元相当限制投资者。公司可以做股票拆分，他们说每股现在是 2 股，价格是一半。任何人如果以 1,000 美元买入 1 股苹果股份，在拆分之后，苹果的股票翻倍，他们将拥有 2 股苹果（AAPL），每股价值 500 美元。调整收盘价是有帮助的，因为它解释了未来的股票分割，并给出分割的相对价格。出于这个原因，调整价格是你最有可能处理的价格。

## 二、处理数据和绘图

欢迎阅读 Python 金融系列教程的第 2 部分。 在本教程中，我们将使用我们的股票数据进一步拆分一些基本的数据操作和可视化。 我们将使用的起始代码（在前面的教程中已经介绍过）是：

```py
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)
df = web.DataReader('TSLA', 'yahoo', start, end)
```

我们可以用这些`DataFrame`做些什么？ 首先，我们可以很容易地将它们保存到各种数据类型中。 一个选项是`csv`：

```py
df.to_csv('TSLA.csv')
```

我们也可以将数据从 CSV 文件读取到`DataFrame`中，而不是将数据从 Yahoo 财经 API 读取到`DataFrame`中：

```py
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
```

现在，我们可以绘制它：

```py
df.plot()
plt.show()
```

![](https://pythonprogramming.net/static/images/finance/initial_graph_volume.png)

很酷，尽管我们真正能看到的唯一的东西就是成交量，因为它比股票价格大得多。 我们怎么可能仅仅绘制我们感兴趣的东西？

```py
df['Adj Close'].plot()
plt.show()
```

![](https://pythonprogramming.net/static/images/finance/stock_data_graph.png)

你可以看到，你可以在`DataFrame`中引用特定的列，如：`df['Adj Close']`，但是你也可以一次引用多个，如下所示：

```py
df[['High','Low']]
```

在下一个教程中，我们将介绍这些数据的一些基本操作，以及一些更基本的可视化。

## 三、基本的股票数据操作

欢迎阅读 Python 金融系列教程的第 3 部分。 在本教程中，我们将使用我们的股票数据进一步拆分一些基本的数据操作和可视化。 我们将要使用的起始代码（在前面的教程中已经介绍过）是：

```py
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
```

Pandas 模块配备了一堆可用的内置函数，以及创建自定义 Pandas 函数的方法。 稍后我们将介绍一些自定义函数，但现在让我们对这些数据执行一个非常常见的操作：移动均值。

简单移动均值的想法是选取时间窗口，并计算该窗口内的均值。 然后我们把这个窗口移动一个周期，然后再做一次。 在我们这里，我们将计算 100 天滚动均值。 因此，这将选取当前价格和过去 99 天的价格，加起来，除以 100，之后就是当前的 100 天移动均值。 然后我们把窗口移动一天，然后再做同样的事情。 在 Pandas 中这样做很简单：

```py
df['100ma'] = df['Adj Close'].rolling(window=100).mean()
```


如果我们有一列叫做`100ma`，执行`df['100ma']`允许我们重新定义包含现有列的内容，否则创建一个新列，这就是我们在这里做的。 我们说`df['100ma']`列等同于应用滚动方法的`df['Adj Close']`列，窗口为 100，这个窗口将是` mean()`（均值）操作。

现在，我们执行：

```py
print(df.head())
```

```
                  Date       Open   High        Low      Close    Volume  \
Date                                                                       
2010-06-29  2010-06-29  19.000000  25.00  17.540001  23.889999  18766300   
2010-06-30  2010-06-30  25.790001  30.42  23.299999  23.830000  17187100   
2010-07-01  2010-07-01  25.000000  25.92  20.270000  21.959999   8218800   
2010-07-02  2010-07-02  23.000000  23.10  18.709999  19.200001   5139800   
2010-07-06  2010-07-06  20.000000  20.00  15.830000  16.110001   6866900   

            Adj Close  100ma  
Date                          
2010-06-29  23.889999    NaN  
2010-06-30  23.830000    NaN  
2010-07-01  21.959999    NaN  
2010-07-02  19.200001    NaN  
2010-07-06  16.110001    NaN  
```

发生了什么？ 在`100ma`列中，我们只看到`NaN`。 我们选择了 100 移动均值，理论上需要 100 个之前的数据点进行计算，所以我们在这里没有任何前 100 行的数据。 `NaN`的意思是“不是一个数字”。 有了 Pandas，你可以决定对缺失数据做很多事情，但现在，我们只需要改变最小周期参数：

```
                  Date       Open   High        Low      Close    Volume  \
Date                                                                       
2010-06-29  2010-06-29  19.000000  25.00  17.540001  23.889999  18766300   
2010-06-30  2010-06-30  25.790001  30.42  23.299999  23.830000  17187100   
2010-07-01  2010-07-01  25.000000  25.92  20.270000  21.959999   8218800   
2010-07-02  2010-07-02  23.000000  23.10  18.709999  19.200001   5139800   
2010-07-06  2010-07-06  20.000000  20.00  15.830000  16.110001   6866900   

            Adj Close      100ma  
Date                              
2010-06-29  23.889999  23.889999  
2010-06-30  23.830000  23.860000  
2010-07-01  21.959999  23.226666  
2010-07-02  19.200001  22.220000  
2010-07-06  16.110001  20.998000 
```

好吧，可以用，现在我们想看看它！ 但是我们已经看到了简单的图表，那么稍微复杂一些呢？

```py
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
```

如果你想了解`subplot2grid`的更多信息，请查看 Matplotlib 教程的子图部分。

基本上，我们说我们想要创建两个子图，而这两个子图都在`6x1`的网格中，我们有 6 行 1 列。 第一个子图从该网格上的`(0,0)`开始，跨越 5 行，并跨越 1 列。 下一个子图也在`6x1`网格上，但是从`(5,0)`开始，跨越 1 行和 1 列。 第二个子图带有`sharex = ax1`，这意味着`ax2`的`x`轴将始终与`ax1`的`x`轴对齐，反之亦然。 现在我们只是绘制我们的图形：

```py
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()
```

在上面，我们在第一个子图中绘制了的`close`和`100ma`，第二个图中绘制`volume`。 我们的结果：

![](https://pythonprogramming.net/static/images/finance/price_ma_and_volume_stock_graph_python.png)

到这里的完整代码：

```py
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean()
print(df.head())

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])

plt.show()
```

在接下来的几个教程中，我们将学习如何通过 Pandas 数据重采样制作烛台图，并学习更多使用 Matplotlib 的知识。

## 四、更多股票操作

欢迎阅读 Python 金融教程系列的第 4 部分。 在本教程中，我们将基于`Adj Close`列创建烛台/  OHLC 图，我将介绍重新采样和其他一些数据可视化概念。

名为烛台图的 OHLC 图是一个图表，将开盘价，最高价，最低价和收盘价都汇总成很好的格式。 并且它使用漂亮的颜色，还记得我告诉你有关漂亮的图表的事情嘛？

之前的教程中，目前为止的起始代码：

```py
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)
```

不幸的是，即使创建 OHLC 数据是这样，Pandas 没有内置制作烛台图的功能。 有一天，我确信这个图表类型将会可用，但是，现在不是。 没关系，我们会实现它！ 首先，我们需要做两个新的导入：

```py
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
```

第一个导入是来自 matplotlib 的 OHLC 图形类型，第二个导入是特殊的`mdates`类型，它在对接中是个麻烦，但这是 matplotlib 图形的日期类型。 Pandas 自动为你处理，但正如我所说，我们没有那么方便的烛台。

首先，我们需要适当的 OHLC 数据。 我们目前的数据确实有 OHLC 值，除非我错了，特斯拉从未有过拆分，但是你不会总是这么幸运。 因此，我们将创建我们自己的 OHLC 数据，这也将使我们能够展示来自 Pandas 的另一个数据转换：

```py
df_ohlc = df['Adj Close'].resample('10D').ohlc()
```

我们在这里所做的是，创建一个新的数据帧，基于`df ['Adj Close']`列，使用 10 天窗口重采样，并且重采样是一个 OHLC（开高低关）。我们也可以用`.mean()`或`.sum()`计算 10 天的均值，或 10 天的总和。请记住，这 10 天的均值是 10 天均值，而不是滚动均值。由于我们的数据是每日数据，重采样到 10 天的数据有效地缩小了我们的数据大小。这就是你规范多个数据集的方式。有时候，您可能会在每个月的第一天记录一次数据，在每个月末记录其他数据，最后每周记录一些数据。您可以将该数据帧重新采样到月底，并有效地规范化所有东西！这是一个更先进的 Padas 功能，如果你喜欢，你可以更多了解 Pandas 的序列。

我们想要绘制烛台数据以及成交量数据。我们不需要将成交量数据重采样，但是我们应该这样做，因为与我们的`10D`价格数据相比，这个数据太细致了。

```py
df_volume = df['Volume'].resample('10D').sum()
```

我们在这里使用`sum`，因为我们真的想知道在这 10 天内交易总量，但也可以用平均值。 现在如果我们这样做：

```py
print(df_ohlc.head())
```

```
                 open       high        low      close
Date                                                  
2010-06-29  23.889999  23.889999  15.800000  17.459999
2010-07-09  17.400000  20.639999  17.049999  20.639999
2010-07-19  21.910000  21.910000  20.219999  20.719999
2010-07-29  20.350000  21.950001  19.590000  19.590000
2010-08-08  19.600000  19.600000  17.600000  19.150000
```

这是预期，但是，我们现在要将这些信息移动到 matplotlib，并将日期转换为`mdates`版本。 由于我们只是要在 Matplotlib 中绘制列，我们实际上不希望日期成为索引，所以我们可以这样做：

```py
df_ohlc = df_ohlc.reset_index()
```

现在`dates `只是一个普通的列。 接下来，我们要转换它：

```py
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
```

现在我们打算配置图形：

```py
fig = plt.figure()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1,sharex=ax1)
ax1.xaxis_date()
```

除了`ax1.xaxis_date()`之外，你已经看到了一切。 这对我们来说，是把轴从原始的`mdate`数字转换成日期。

现在我们可以绘制烛台图：

```py
candlestick_ohlc(ax1, df_ohlc.values, width=2, colorup='g')
```

之后是成交量：

```py
ax2.fill_between(df_volume.index.map(mdates.date2num),df_volume.values,0)
```

`fill_between`函数将绘制`x`，`y`，然后填充之间的内容。 在我们的例子中，我们选择 0。

```py
plt.show()
```

![](https://pythonprogramming.net/static/images/finance/candlestick_and_volume_graph_matplotlib.png)

这个教程的完整代码：

```py
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader.data as web
style.use('ggplot')

df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

df_ohlc = df['Adj Close'].resample('10D').ohlc()
df_volume = df['Volume'].resample('10D').sum()

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.xaxis_date()

candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0)
plt.show()

```

在接下来的几个教程中，我们将把可视化留到后面一些，然后专注于获取并处理数据。

## 五、自动获取 SP500 列表

欢迎阅读 Python 金融教程系列的第 5 部分。在本教程和接下来的几章中，我们将着手研究如何能够获取大量价格信息，以及如何一次处理所有这些数据。

首先，我们需要一个公司名单。我可以给你一个清单，但实际上获得股票清单可能只是你可能遇到的许多挑战之一。在我们的案例中，我们需要一个 SP500 公司的 Python 列表。

无论您是在寻找道琼斯公司，SP500 指数还是罗素 3000 指数，这些公司的信息都有可能在某个地方发布。您需要确保它是最新的，但是它可能还不是完美的格式。在我们的例子中，我们将从维基百科获取这个列表：`http://en.wikipedia.org/wiki/List_of_S%26P_500_companies`。

维基百科中的代码/符号组织在一张表里面。为了解决这个问题，我们将使用 HTML 解析库，Beautiful Soup。如果你想了解更多，我有一个使用 Beautiful Soup 进行网页抓取的简短的四部分教程。

首先，我们从一些导入开始：

```py
import bs4 as bs
import pickle
import requests
```

`bs4`是 Beautiful Soup，`pickle `是为了我们可以很容易保存这个公司的名单，而不是每次我们运行时都访问维基百科（但要记住，你需要及时更新这个名单！），我们将使用 `requests `从维基百科页面获取源代码。

这是我们函数的开始：

```py
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
```

首先，我们访问维基百科页面，并获得响应，其中包含我们的源代码。 为了处理源代码，我们想要访问`.text`属性，我们使用 BeautifulSoup 将其转为`soup`。 如果您不熟悉 BeautifulSoup 为您所做的工作，它基本上将源代码转换为一个 BeautifulSoup 对象，马上就可以看做一个典型的 Python 对象。

有一次维基百科试图拒绝 Python 的访问。 目前，在我写这篇文章的时候，代码不改变协议头也能工作。 如果您发现原始源代码（`resp.text`）似乎不返回相同的页面，像您在家用计算机上看到的那样，请添加以下内容并更改`resp var`代码：

```py
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.17 (KHTML, like Gecko) Chrome/24.0.1312.27 Safari/537.17'}
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
                        headers=headers)
```

一旦我们有了`soup`，我们可以通过简单地搜索`wikitable sortable`类来找到股票数据表。 我知道指定这个表的唯一原因是，因为我之前在浏览器中查看了源代码。 可能会有这样的情况，你想解析一个不同的网站的股票列表，也许它是在一个表中，也可能是一个列表，或者可能是一些`div`标签。 这都是一个非常具体的解决方案。 从这里开始，我们仅仅遍历表格：

```py
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
```

对于每一行，在标题行之后（这就是为什么我们要执行`[1:]`），我们说股票是“表格数据”（`td`），我们抓取它的`.text`， 将此代码添加到我们的列表中。

现在，如果我们可以保存这个列表，那就好了。 我们将使用`pickle`模块来为我们序列化 Python 对象。

```py
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)

    return tickers
```

我们希望继续并保存它，因此我们无需每天多次请求维基百科。 在任何时候，我们可以更新这个清单，或者我们可以编程一个月检查一次...等等。

目前为止的完整代码：

```py
import bs4 as bs
import pickle
import requests

def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

save_sp500_tickers()
```

现在我们已经知道了代码，我们已经准备好提取所有的信息，这是我们将在下一个教程中做的事情。

## 六、获取 SP500 中所有公司的价格数据

欢迎阅读 Python 金融教程系列的第 6 部分。 在之前的 Python 教程中，我们介绍了如何获取我们感兴趣的公司名单（在我们的案例中是 SP500），现在我们将获取所有这些公司的股票价格数据。

目前为止的代码：

```py
def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers
```

我们打算添加一些新的导入：

```py
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
```

我们将使用`datetime`为 Pandas `datareader`指定日期，`os`用于检查并创建目录。 你已经知道 Pandas 干什么了！

我们的新函数的开始：

```py
def get_data_from_yahoo(reload_sp500=False):
    
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
```

在这里，我将展示一个简单示例，可以处理是否重新加载 SP500 列表。 如果我们让它这样，这个程序将重新抓取 SP500，否则将只使用我们的`pickle`。 现在我们准备抓取数据。

现在我们需要决定我们要处理的数据。 我倾向于尝试解析网站一次，并在本地存储数据。 我不会事先知道我可能用数据做的所有事情，但是我知道如果我不止一次地抓取它，我还可以保存它（除非它是一个巨大的数据集，但不是）。 因此，对于每一种股票，我们抓取所有雅虎可以返回给我们的东西，并保存下来。 为此，我们将创建一个新目录，并在那里存储每个公司的股票数据。 首先，我们需要这个初始目录：

```py
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')
```

您可以将这些数据集存储在与您的脚本相同的目录中，但在我看来，这会变得非常混乱。 现在我们准备好提取数据了。 你已经知道如何实现，我们在第一个教程中完成了！

```py
    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)
    
    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))
```

你可能想要为这个函数传入`force_data_update`参数，因为现在它不会重新提取它已经访问的数据。 由于我们正在提取每日数据，所以您最好至少重新提取最新的数据。 也就是说，如果是这样的话，最好对每个公司使用数据库而不是表格，然后从 Yahoo 数据库中提取最新的值。 但是现在我们会保持简单！

目前为止的代码：

```py
import bs4 as bs
import datetime as dt
import os
import pandas as pd
import pandas_datareader.data as web
import pickle
import requests


def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        
    return tickers

#save_sp500_tickers()


def get_data_from_yahoo(reload_sp500=False):
    
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
    
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime(2016, 12, 31)
    
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, "yahoo", start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

get_data_from_yahoo()
```

如果雅虎阻拦你的话。 在我写这篇文章的时候，雅虎并没有阻拦我，我能够毫无问题地完成这个任务。 但是这可能需要你一段时间，尤其取决于你的机器。 好消息是，我们不需要再做一遍！ 同样在实践中，因为这是每日数据，但是您可能每天都执行一次。

另外，如果你的互联网速度很慢，你不需要获取所有的代码，即使只有 10 个就足够了，所以你可以用`ticker [:10]`或者类似的东西来加快速度。

在下一个教程中，一旦你下载了数据，我们将把我们感兴趣的数据编译成一个大的 Pandas`DataFrame`。
