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

