# TensorFlow 聊天机器人

## 一、使用深度学习创建聊天机器人

您好，欢迎阅读 Python 聊天机器人系列教程。 在本系列中，我们将介绍如何使用 Python 和 TensorFlow 创建一个能用的聊天机器人。 以下是一些 chatbot 的实例：

> I use Google and it works.
> 
> — Charles the AI (@Charles_the_AI) November 24, 2017

> I prefer cheese.
> 
> — Charles the AI (@Charles_the_AI) November 24, 2017

> The internet
> 
> — Charles the AI (@Charles_the_AI) November 24, 2017

> I'm not sure . I'm just a little drunk.
> 
> — Charles the AI (@Charles_the_AI) November 24, 2017

我的目标是创建一个聊天机器人，可以实时与 Twitch Stream 上的人交谈，而不是听起来像个白痴。为了创建一个聊天机器人，或者真的做任何机器学习任务，当然，您的第一个任务就是获取训练数据，之后您需要构建并准备，将其格式化为“输入”和“输出”形式，机器学习算法可以消化它。可以说，这就是做任何机器学习时的实际工作。建立模型和训练/测试步骤简单的部分！

为了获得聊天训练数据，您可以查看相当多的资源。例如，[康奈尔电影对话语料库](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)似乎是最受欢迎的语料之一。还有很多其他来源，但我想要的东西更加......原始。有些没有美化的东西，有一些带有为其准备的特征。自然，这把我带到了 Reddit。起初，我认为我会使用 Python Reddit API 包装器，但 Reddit 对抓取的限制并不是最友好的。为了收集大量的数据，你必须打破一些规则。相反，我发现了一个 [17 亿个 Reddit 评论的数据转储](https://www.reddit.com/r/datasets/comments/3bxlg7/i_have_every_publicly_available_reddit_comment/?st=j9udbxta&sh=69e4fee7)。那么，应该使用它！

Reddit 的结构是树形的，不像论坛，一切都是线性的。父评论是线性的，但父评论的回复是个分支。以防有些人不熟悉：

```
-Top level reply 1
--Reply to top level reply 1
--Reply to top level reply 1
---Reply to reply...
-Top level reply 2
--Reply to top level reply 1
-Top level reply 3	
```

我们需要用于深度学习的结构是输入输出。 所以我们实际上通过评论和回复偶对的方式，试图获得更多的东西。 在上面的例子中，我们可以使用以下作为评论回复偶对：

```
-Top level reply 1 and --Reply to top level reply 1

--Reply to top level reply 1 and ---Reply to reply...
```

所以，我们需要做的是获取这个 Reddit 转储，并产生这些偶对。 接下来我们需要考虑的是，每个评论应该只有 1 个回复。 尽管许多单独的评论可能会有很多回复，但我们应该只用一个。 我们可以只用第一个，或者我们可以用最顶上那个。 稍后再说。 我们的第一个任务是获取数据。 如果你有存储限制，你可以查看一个月的 Reddit 评论，这是 2015 年 1 月。否则，你可以获取整个转储：

```
magnet:?xt=urn:btih:7690f71ea949b868080401c749e878f98de34d3d&dn=reddit%5Fdata&tr=http%3A%2F%2Ftracker.pushshift.io%3A6969%2Fannounce&tr=udp%3A%2F%2Ftracker.openbittorrent.com%3A80

```

我只下载过两次这个种子，但根据种子和对等的不同，下载速度可能会有很大差异。

最后，您还可以通过 [Google BigQuery](https://www.reddit.com/r/bigquery/comments/3cej2b/17_billion_reddit_comments_loaded_on_bigquery/?st=j9xmvats&sh=5843d18e) 查看所有 Reddit 评论。 BigQuery 表似乎随着时间的推移而更新，而 torrent 不是，所以这也是一个不错的选择。 我个人将会使用 torrent，因为它是完全免费的，所以，如果你想完全遵循它，就需要这样做，但如果你愿意的话，可以随意改变主意，使用 Google BigQuery 的东西！

由于数据下载可能需要相当长的时间，我会在这里中断。 一旦你下载了数据，继续下一个教程。 您可以仅仅下载`2015-01`文件来跟随整个系列教程，您不需要整个 17 亿个评论转储。 一个月的就足够了。

## 二、聊天数据结构

欢迎阅读 Python 和 TensorFlow 聊天机器人系列教程的第二部分。现在，我假设你已经下载了数据，或者你只是在这里观看。对于大多数机器学习，您需要获取数据，并且某些时候需要输入和输出。对于神经网络，这表示实际神经网络的输入层和输出层。对于聊天机器人来说，这意味着我们需要将东西拆成评论和回复。评论是输入，回复是所需的输出。现在使用 Reddit，并不是所有的评论都有回复，然后很多评论会有很多回复！我们需要挑一个。

我们需要考虑的另一件事是，当我们遍历这个文件时，我们可能会发现一个回复，但随后我们可能会找到更好的回复。我们可以使用一种方法是看看得票最高的。我们可能也只想要得票最高的回应。我们可以考虑在这里很多事情，按照你的希望随意调整！

首先，我们的数据格式，如果我们走了 torrent  路线：

```json
{"author":"Arve","link_id":"t3_5yba3","score":0,"body":"Can we please deprecate the word \"Ajax\" now? \r\n\r\n(But yeah, this _is_ much nicer)","score_hidden":false,"author_flair_text":null,"gilded":0,"subreddit":"reddit.com","edited":false,"author_flair_css_class":null,"retrieved_on":1427426409,"name":"t1_c0299ap","created_utc":"1192450643","parent_id":"t1_c02999p","controversiality":0,"ups":0,"distinguished":null,"id":"c0299ap","subreddit_id":"t5_6","downs":0,"archived":true}

```

每一行就像上面那样。我们并不需要这些数据的全部，但是我们肯定需要`body`，`comment_id`和`parent_id`。如果您下载完整的 torrent 文件，或者正在使用 BigQuery 数据库，那么可以使用样例数据，所以我也将使用`score`。我们可以为分数设定限制。我们也可以处理特定的`subreddit`，来创建一个说话风格像特定 subreddit 的 AI。现在，我会处理所有 subreddit。

现在，即使一个月的评论也可能超过 32GB，我也无法将其纳入 RAM，我们需要通过数据进行缓冲。我的想法是继续并缓冲评论文件，然后将我们感兴趣的数据存储到 SQLite 数据库中。这里的想法是我们可以将评论数据插入到这个数据库中。所有评论将按时间顺序排列，所有评论最初都是“父节点”，自己并没有父节点。随着时间的推移，会有回复，然后我们可以存储这个“回复”，它将在数据库中有父节点，我们也可以按照 ID 拉取，然后我们可以检索一些行，其中我们拥有父评论和回复。

然后，随着时间的推移，我们可能会发现父评论的回复，这些回复的投票数高于目前在那里的回复。发生这种情况时，我们可以使用新信息更新该行，以便我们可以最终得到通常投票数较高的回复。

无论如何，有很多方法可以实现，让我们开始吧！首先，让我们进行一些导入：

```py
import sqlite3
import json
from datetime import datetime
```

我们将为我们的数据库使用`sqlite3`，`json`用于从`datadump`加载行，然后`datetime`实际只是为了记录。 这不完全必要。

所以 torrent  转储带有一大堆目录，其中包含实际的`json`数据转储，按年和月（YYYY-MM）命名。 他们压缩为`.bz2`。 确保你提取你打算使用的那些。 我们不打算编写代码来做，所以请确保你完成了！

下面，我们以一些变量开始：

```py
timeframe = '2015-05'
sql_transaction = []

connection = sqlite3.connect('{}.db'.format(timeframe))
c = connection.cursor()
```

`timeframe`值将成为我们将要使用的数据的年份和月份。 你也可以把它列在这里，然后如果你喜欢，可以遍历它们。 现在，我将只用 2015 年 5 月的文件。 接下来，我们有`sql_transaction`。 所以在 SQL 中的“提交”是更昂贵的操作。 如果你知道你将要插入数百万行，你也应该知道你*真的*不应该一一提交。 相反，您只需在单个事务中构建语句，然后执行全部操作，然后提交。 接下来，我们要创建我们的表。 使用 SQLite，如果数据库尚不存在，连接时会创建数据库。

```py
def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

```

在这里，我们正在准备存储`parent_id`，`comment_id`，父评论，回复（评论），subreddit，时间，然后最后是评论的评分（得票）。

接下来，我们可以开始我们的主代码块：

```py
if __name__ == '__main__':
    create_table()
```

目前为止的完整代码：

```py
import sqlite3
import json
from datetime import datetime

timeframe = '2015-05'
sql_transaction = []

connection = sqlite3.connect('{}2.db'.format(timeframe))
c = connection.cursor()

def create_table():
    c.execute("CREATE TABLE IF NOT EXISTS parent_reply(parent_id TEXT PRIMARY KEY, comment_id TEXT UNIQUE, parent TEXT, comment TEXT, subreddit TEXT, unix INT, score INT)")

if __name__ == '__main__':
    create_table()
```

一旦我们建立完成，我们就可以开始遍历我们的数据文件并存储这些信息。 我们将在下一个教程中开始这样做！
