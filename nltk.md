# 自然语言处理教程

# 一、使用 NLTK 分析单词和句子

欢迎阅读自然语言处理系列教程，使用 Python 的自然语言工具包 NLTK 模块。

NLTK 模块是一个巨大的工具包，目的是在整个自然语言处理（NLP）方法上帮助您。 NLTK 将为您提供一切，从将段落拆分为句子，拆分词语，识别这些词语的词性，高亮主题，甚至帮助您的机器了解文本关于什么。在这个系列中，我们将要解决意见挖掘或情感分析的领域。

在我们学习如何使用 NLTK 进行情感分析的过程中，我们将学习以下内容：

+   分词 - 将文本正文分割为句子和单词。
+   词性标注
+   机器学习与朴素贝叶斯分类器
+   如何一起使用 Scikit Learn（sklearn）与 NLTK
+   用数据集训练分类器
+   用 Twitter 进行实时的流式情感分析。
+   ...以及更多。

为了开始，你需要 NLTK 模块，以及 Python。

如果您还没有 Python，请转到`python.org`并下载最新版本的 Python（如果您在 Windows上）。如果你在 Mac 或 Linux 上，你应该可以运行`apt-get install python3`。

接下来，您需要 NLTK 3。安装 NLTK 模块的最简单方法是使用`pip`。

对于所有的用户来说，这通过打开`cmd.exe`，bash，或者你使用的任何 shell，并键入以下命令来完成：

```
pip install nltk
```


接下来，我们需要为 NLTK 安装一些组件。通过你的任何常用方式打开 python，然后键入：

```py
    import nltk
    nltk.download()
```

除非你正在操作无头版本，否则一个 GUI 会弹出来，可能只有红色而不是绿色：

![](https://pythonprogramming.net/static/images/nltk/nltk-download-gui.png)

为所有软件包选择下载“全部”，然后单击“下载”。 这会给你所有分词器，分块器，其他算法和所有的语料库。 如果空间是个问题，您可以选择手动选择性下载所有内容。 NLTK 模块将占用大约 7MB，整个`nltk_data`目录将占用大约 1.8GB，其中包括您的分块器，解析器和语料库。

如果您正在使用 VPS 运行无头版本，您可以通过运行 Python ，并执行以下操作来安装所有内容：

```py
import nltk

nltk.download()

d (for download)

all (for download everything)
```

这将为你下载一切东西。

现在你已经拥有了所有你需要的东西，让我们敲一些简单的词汇：

+   语料库（Corpus） - 文本的正文，单数。Corpora 是它的复数。示例：`A collection of medical journals`。
+   词库（Lexicon） - 词汇及其含义。例如：英文字典。但是，考虑到各个领域会有不同的词库。例如：对于金融投资者来说，`Bull`（牛市）这个词的第一个含义是对市场充满信心的人，与“普通英语词汇”相比，这个词的第一个含义是动物。因此，金融投资者，医生，儿童，机械师等都有一个特殊的词库。
+   标记（Token） - 每个“实体”都是根据规则分割的一部分。例如，当一个句子被“拆分”成单词时，每个单词都是一个标记。如果您将段落拆分为句子，则每个句子也可以是一个标记。

这些是在进入自然语言处理（NLP）领域时，最常听到的词语，但是我们将及时涵盖更多的词汇。以此，我们来展示一个例子，说明如何用 NLTK 模块将某些东西拆分为标记。

```py
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT))
```

起初，你可能会认为按照词或句子来分词，是一件相当微不足道的事情。 对于很多句子来说，它可能是。 第一步可能是执行一个简单的`.split('. ')`，或按照句号，然后是空格分割。 之后也许你会引入一些正则表达式，来按照句号，空格，然后是大写字母分割。 问题是像`Mr. Smith`这样的事情，还有很多其他的事情会给你带来麻烦。 按照词分割也是一个挑战，特别是在考虑缩写的时候，例如`we`和`we're`。 NLTK 用这个看起来简单但非常复杂的操作为您节省大量的时间。

上面的代码会输出句子，分成一个句子列表，你可以用`for`循环来遍历。

```py
['Hello Mr. Smith, how are you doing today?', 'The weather is great, and Python is awesome.', 'The sky is pinkish-blue.', "You shouldn't eat cardboard."]
```

所以这里，我们创建了标记，它们都是句子。让我们这次按照词来分词。

```py
print(word_tokenize(EXAMPLE_TEXT))

['Hello', 'Mr.', 'Smith', ',', 'how', 'are', 'you', 'doing', 'today', '?', 'The', 'weather', 'is', 'great', ',', 'and', 'Python', 'is', 'awesome', '.', 'The', 'sky', 'is', 'pinkish-blue', '.', 'You', 'should', "n't", 'eat', 'cardboard', '.']
```

这里有几件事要注意。 首先，注意标点符号被视为一个单独的标记。 另外，注意单词`shouldn't`分隔为`should`和`n't`。 最后要注意的是，`pinkish-blue`确实被当作“一个词”来对待，本来就是这样。很酷！

现在，看着这些分词后的单词，我们必须开始思考我们的下一步可能是什么。 我们开始思考如何通过观察这些词汇来获得含义。 我们可以想清楚，如何把价值放在许多单词上，但我们也看到一些基本上毫无价值的单词。 这是一种“停止词”的形式，我们也可以处理。 这就是我们将在下一个教程中讨论的内容。

## 二、NLTK 与停止词

自然语言处理的思想，是进行某种形式的分析或处理，机器至少可以在某种程度上理解文本的含义，表述或暗示。

这显然是一个巨大的挑战，但是有一些任何人都能遵循的步骤。然而，主要思想是电脑根本不会直接理解单词。令人震惊的是，人类也不会。在人类中，记忆被分解成大脑中的电信号，以发射模式的神经组的形式。对于大脑还有很多未知的事情，但是我们越是把人脑分解成基本的元素，我们就会发现基本的元素。那么，事实证明，计算机以非常相似的方式存储信息！如果我们要模仿人类如何阅读和理解文本，我们需要一种尽可能接近的方法。一般来说，计算机使用数字来表示一切事物，但是我们经常直接在编程中看到使用二进制信号（`True`或`False`，可以直接转换为 1 或 0，直接来源于电信号存在`(True, 1)`或不存在`(False, 0)`）。为此，我们需要一种方法,将单词转换为数值或信号模式。将数据转换成计算机可以理解的东西，这个过程称为“预处理”。预处理的主要形式之一就是过滤掉无用的数据。在自然语言处理中，无用词（数据）被称为停止词。

我们可以立即认识到，有些词语比其他词语更有意义。我们也可以看到，有些单词是无用的，是填充词。例如，我们在英语中使用它们来填充句子，这样就没有那么奇怪的声音了。一个最常见的，非官方的，无用词的例子是单词`umm`。人们经常用`umm`来填充，比别的词多一些。这个词毫无意义，除非我们正在寻找一个可能缺乏自信，困惑，或者说没有太多话的人。我们都这样做，有...呃...很多时候，你可以在视频中听到我说`umm`或`uhh`。对于大多数分析而言，这些词是无用的。

我们不希望这些词占用我们数据库的空间，或占用宝贵的处理时间。因此，我们称这些词为“无用词”，因为它们是无用的，我们希望对它们不做处理。 “停止词”这个词的另一个版本可以更书面一些：我们停在上面的单词。

例如，如果您发现通常用于讽刺的词语，可能希望立即停止。讽刺的单词或短语将因词库和语料库而异。就目前而言，我们将把停止词当作不含任何含义的词，我们要把它们删除。

您可以轻松地实现它，通过存储您认为是停止词的单词列表。 NLTK 用一堆他们认为是停止词的单词，来让你起步，你可以通过 NLTK 语料库来访问它：

```py
from nltk.corpus import stopwords
```

这里是这个列表：

```py
>>> set(stopwords.words('english'))
{'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than'}
```

以下是结合使用`stop_words`集合，从文本中删除停止词的方法：

```py
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example_sent)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

filtered_sentence = []

for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

print(word_tokens)
print(filtered_sentence)
```

我们的输出是：

```py
['This', 'is', 'a', 'sample', 'sentence', ',', 'showing', 'off', 'the', 'stop', 'words', 'filtration', '.']
['This', 'sample', 'sentence', ',', 'showing', 'stop', 'words', 'filtration', '.']
```

我们的数据库感谢了我们。数据预处理的另一种形式是“词干提取（Stemming）”，这就是我们接下来要讨论的内容。

## 三、NLTK 词干提取

词干的概念是一种规范化方法。 除涉及时态之外，许多词语的变体都具有相同的含义。

我们提取词干的原因是为了缩短查找的时间，使句子正常化。

考虑：

```
I was taking a ride in the car.
I was riding in the car.
```

这两句话意味着同样的事情。 `in the car`（在车上）是一样的。 `I`（我）是一样的。 在这两种情况下，`ing`都明确表示过去式，所以在试图弄清这个过去式活动的含义的情况下，是否真的有必要区分`riding`和`taking a ride`？

不，并没有。

这只是一个小例子，但想象英语中的每个单词，可以放在单词上的每个可能的时态和词缀。 每个版本有单独的字典条目，将非常冗余和低效，特别是因为一旦我们转换为数字，“价值”将是相同的。

最流行的瓷感提取算法之一是 Porter，1979 年就存在了。

首先，我们要抓取并定义我们的词干：

```py
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
```

现在让我们选择一些带有相似词干的单词，例如：

```py
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
```

下面，我们可以这样做来轻易提取词干：

```py
for w in example_words:
    print(ps.stem(w))
```

我们的输出：

```py
python
python
python
python
pythonli
```

现在让我们尝试对一个典型的句子，而不是一些单词提取词干：

```py
new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words = word_tokenize(new_text)

for w in words:
    print(ps.stem(w))
```

现在我们的结果为：

```
It
is
import
to
by
veri
pythonli
while
you
are
python
with
python
.
All
python
have
python
poorli
at
least
onc
.
```

接下来，我们将讨论 NLTK 模块中一些更高级的内容，词性标注，其中我们可以使用 NLTK 模块来识别句子中每个单词的词性。

## 四、NLTK 词性标注

NLTK模块的一个更强大的方面是，它可以为你做词性标注。 意思是把一个句子中的单词标注为名词，形容词，动词等。 更令人印象深刻的是，它也可以按照时态来标记，以及其他。 这是一列标签，它们的含义和一些例子：

```py
POS tag list:

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there (like: "there is" ... think of it like "there exists")
FW	foreign word
IN	preposition/subordinating conjunction
JJ	adjective	'big'
JJR	adjective, comparative	'bigger'
JJS	adjective, superlative	'biggest'
LS	list marker	1)
MD	modal	could, will
NN	noun, singular 'desk'
NNS	noun plural	'desks'
NNP	proper noun, singular	'Harrison'
NNPS	proper noun, plural	'Americans'
PDT	predeterminer	'all the kids'
POS	possessive ending	parent's
PRP	personal pronoun	I, he, she
PRP$	possessive pronoun	my, his, hers
RB	adverb	very, silently,
RBR	adverb, comparative	better
RBS	adverb, superlative	best
RP	particle	give up
TO	to	go 'to' the store.
UH	interjection	errrrrrrrm
VB	verb, base form	take
VBD	verb, past tense	took
VBG	verb, gerund/present participle	taking
VBN	verb, past participle	taken
VBP	verb, sing. present, non-3d	take
VBZ	verb, 3rd person sing. present	takes
WDT	wh-determiner	which
WP	wh-pronoun	who, what
WP$	possessive wh-pronoun	whose
WRB	wh-abverb	where, when
```

我们如何使用这个？ 当我们处理它的时候，我们要讲解一个新的句子标记器，叫做`PunktSentenceTokenizer`。 这个标记器能够无监督地进行机器学习，所以你可以在你使用的任何文本上进行实际的训练。 首先，让我们获取一些我们打算使用的导入：

```py
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
```

现在让我们创建训练和测试数据：

```py
train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")
```

一个是 2005 年以来的国情咨文演说，另一个是 2006 年以来的乔治·W·布什总统的演讲。

接下来，我们可以训练 Punkt 标记器，如下所示：

```py
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
```

之后我们可以实际分词，使用：

```py
tokenized = custom_sent_tokenizer.tokenize(sample_text)
```

现在我们可以通过创建一个函数，来完成这个词性标注脚本，该函数将遍历并标记每个句子的词性，如下所示：

```py
def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)

    except Exception as e:
        print(str(e))


process_content()
```

输出应该是元组列表，元组中的第一个元素是单词，第二个元素是词性标签。 它应该看起来像：

```py
[('PRESIDENT', 'NNP'), ('GEORGE', 'NNP'), ('W.', 'NNP'), ('BUSH', 'NNP'), ("'S", 'POS'), ('ADDRESS', 'NNP'), ('BEFORE', 'NNP'), ('A', 'NNP'), ('JOINT', 'NNP'), ('SESSION', 'NNP'), ('OF', 'NNP'), ('THE', 'NNP'), ('CONGRESS', 'NNP'), ('ON', 'NNP'), ('THE', 'NNP'), ('STATE', 'NNP'), ('OF', 'NNP'), ('THE', 'NNP'), ('UNION', 'NNP'), ('January', 'NNP'), ('31', 'CD'), (',', ','), ('2006', 'CD'), ('THE', 'DT'), ('PRESIDENT', 'NNP'), (':', ':'), ('Thank', 'NNP'), ('you', 'PRP'), ('all', 'DT'), ('.', '.')] [('Mr.', 'NNP'), ('Speaker', 'NNP'), (',', ','), ('Vice', 'NNP'), ('President', 'NNP'), ('Cheney', 'NNP'), (',', ','), ('members', 'NNS'), ('of', 'IN'), ('Congress', 'NNP'), (',', ','), ('members', 'NNS'), ('of', 'IN'), ('the', 'DT'), ('Supreme', 'NNP'), ('Court', 'NNP'), ('and', 'CC'), ('diplomatic', 'JJ'), ('corps', 'NNS'), (',', ','), ('distinguished', 'VBD'), ('guests', 'NNS'), (',', ','), ('and', 'CC'), ('fellow', 'JJ'), ('citizens', 'NNS'), (':', ':'), ('Today', 'NN'), ('our', 'PRP$'), ('nation', 'NN'), ('lost', 'VBD'), ('a', 'DT'), ('beloved', 'VBN'), (',', ','), ('graceful', 'JJ'), (',', ','), ('courageous', 'JJ'), ('woman', 'NN'), ('who', 'WP'), ('called', 'VBN'), ('America', 'NNP'), ('to', 'TO'), ('its', 'PRP$'), ('founding', 'NN'), ('ideals', 'NNS'), ('and', 'CC'), ('carried', 'VBD'), ('on', 'IN'), ('a', 'DT'), ('noble', 'JJ'), ('dream', 'NN'), ('.', '.')] [('Tonight', 'NNP'), ('we', 'PRP'), ('are', 'VBP'), ('comforted', 'VBN'), ('by', 'IN'), ('the', 'DT'), ('hope', 'NN'), ('of', 'IN'), ('a', 'DT'), ('glad', 'NN'), ('reunion', 'NN'), ('with', 'IN'), ('the', 'DT'), ('husband', 'NN'), ('who', 'WP'), ('was', 'VBD'), ('taken', 'VBN'), ('so', 'RB'), ('long', 'RB'), ('ago', 'RB'), (',', ','), ('and', 'CC'), ('we', 'PRP'), ('are', 'VBP'), ('grateful', 'JJ'), ('for', 'IN'), ('the', 'DT'), ('good', 'NN'), ('life', 'NN'), ('of', 'IN'), ('Coretta', 'NNP'), ('Scott', 'NNP'), ('King', 'NNP'), ('.', '.')] [('(', 'NN'), ('Applause', 'NNP'), ('.', '.'), (')', ':')] [('President', 'NNP'), ('George', 'NNP'), ('W.', 'NNP'), ('Bush', 'NNP'), ('reacts', 'VBZ'), ('to', 'TO'), ('applause', 'VB'), ('during', 'IN'), ('his', 'PRP$'), ('State', 'NNP'), ('of', 'IN'), ('the', 'DT'), ('Union', 'NNP'), ('Address', 'NNP'), ('at', 'IN'), ('the', 'DT'), ('Capitol', 'NNP'), (',', ','), ('Tuesday', 'NNP'), (',', ','), ('Jan', 'NNP'), ('.', '.')]
```

到了这里，我们可以开始获得含义，但是还有一些工作要做。 我们将要讨论的下一个话题是分块（chunking），其中我们跟句单词的词性，将单词分到，有意义的分组中。

## 五、NLTK 分块

现在我们知道了词性，我们可以注意所谓的分块，把词汇分成有意义的块。 分块的主要目标之一是将所谓的“名词短语”分组。 这些是包含一个名词的一个或多个单词的短语，可能是一些描述性词语，也可能是一个动词，也可能是一个副词。 这个想法是把名词和与它们有关的词组合在一起。

为了分块，我们将词性标签与正则表达式结合起来。 主要从正则表达式中，我们要利用这些东西：

```
+ = match 1 or more
? = match 0 or 1 repetitions.
* = match 0 or MORE repetitions	  
. = Any character except a new line
```

如果您需要正则表达式的帮助，请参阅上面链接的教程。 最后需要注意的是，词性标签中用`<`和`>`表示，我们也可以在标签本身中放置正则表达式，来表达“全部名词”（`<N.*>`）。

```py
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            chunked.draw()     

    except Exception as e:
        print(str(e))

process_content()
```

结果是这样的：

![](https://pythonprogramming.net/static/images/nltk/nltk_chunking.png)

这里的主要一行是：

```py
chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
```

把这一行拆分开：

`<RB.?>*`：零个或多个任何时态的副词，后面是：

`<VB.?>*`：零个或多个任何时态的动词，后面是：

`<NNP>+`：一个或多个合理的名词，后面是：

`<NN>?`：零个或一个名词单数。

尝试玩转组合来对各种实例进行分组，直到您觉得熟悉了。

视频中没有涉及，但是也有个合理的任务是实际访问具体的块。 这是很少被提及的，但根据你在做的事情，这可能是一个重要的步骤。 假设你把块打印出来，你会看到如下输出：

```
(S
  (Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP)
  'S/POS
  (Chunk
    ADDRESS/NNP
    BEFORE/NNP
    A/NNP
    JOINT/NNP
    SESSION/NNP
    OF/NNP
    THE/NNP
    CONGRESS/NNP
    ON/NNP
    THE/NNP
    STATE/NNP
    OF/NNP
    THE/NNP
    UNION/NNP
    January/NNP)
  31/CD
  ,/,
  2006/CD
  THE/DT
  (Chunk PRESIDENT/NNP)
  :/:
  (Chunk Thank/NNP)
  you/PRP
  all/DT
  ./.)
```

很酷，这可以帮助我们可视化，但如果我们想通过我们的程序访问这些数据呢？ 那么，这里发生的是我们的“分块”变量是一个 NLTK 树。 每个“块”和“非块”是树的“子树”。 我们可以通过像`chunked.subtrees`的东西来引用它们。 然后我们可以像这样遍历这些子树：

```py
            for subtree in chunked.subtrees():
                print(subtree)
```

接下来，我们可能只关心获得这些块，忽略其余部分。 我们可以在`chunked.subtrees()`调用中使用`filter`参数。

```py
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)
```

现在，我们执行过滤，来显示标签为“块”的子树。 请记住，这不是 NLTK 块属性中的“块”...这是字面上的“块”，因为这是我们给它的标签：`chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""`。

如果我们写了一些东西，类似`chunkGram = r"""Pythons: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""`，那么我们可以通过`"Pythons."`标签来过滤。 结果应该是这样的：

```
-
(Chunk PRESIDENT/NNP GEORGE/NNP W./NNP BUSH/NNP)
(Chunk
  ADDRESS/NNP
  BEFORE/NNP
  A/NNP
  JOINT/NNP
  SESSION/NNP
  OF/NNP
  THE/NNP
  CONGRESS/NNP
  ON/NNP
  THE/NNP
  STATE/NNP
  OF/NNP
  THE/NNP
  UNION/NNP
  January/NNP)
(Chunk PRESIDENT/NNP)
(Chunk Thank/NNP)
```

完整的代码是：

```py
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            print(chunked)
            for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk'):
                print(subtree)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
```

## 六、 NLTK 添加缝隙（Chinking）

你可能会发现，经过大量的分块之后，你的块中还有一些你不想要的单词，但是你不知道如何通过分块来摆脱它们。 你可能会发现添加缝隙是你的解决方案。

添加缝隙与分块很像，它基本上是一种从块中删除块的方法。 你从块中删除的块就是你的缝隙。

代码非常相似，你只需要用`}{`来代码缝隙，在块后面，而不是块的`{}`。

```py
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            chunkGram = r"""Chunk: {<.*>+}
                                    }<VB.?|IN|DT|TO>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)

            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
```

使用它，你得到了一些东西：

![](https://pythonprogramming.net/static/images/nltk/chinking.png)

现在，主要的区别是：

```
}<VB.?|IN|DT|TO>+{
```

这意味着我们要从缝隙中删除一个或多个动词，介词，限定词或`to`这个词。

现在我们已经学会了，如何执行一些自定义的分块和添加缝隙，我们来讨论一下 NLTK 自带的分块形式，这就是命名实体识别。

## 七、NLTK 命名实体识别

自然语言处理中最主要的分块形式之一被称为“命名实体识别”。 这个想法是让机器立即能够拉出“实体”，例如人物，地点，事物，位置，货币等等。

这可能是一个挑战，但 NLTK 是为我们内置了它。 NLTK 的命名实体识别有两个主要选项：识别所有命名实体，或将命名实体识别为它们各自的类型，如人物，地点，位置等。

这是一个例子：

```py
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEnt = nltk.ne_chunk(tagged, binary=True)
            namedEnt.draw()
    except Exception as e:
        print(str(e))


process_content()
```

在这里，选择`binary = True`，这意味着一个东西要么是命名实体，要么不是。 将不会有进一步的细节。 结果是：

![](https://pythonprogramming.net/static/images/nltk/named-entity-recognition-binary-true.png)

如果你设置了`binary = False`，结果为：

![](https://pythonprogramming.net/static/images/nltk/named-entity-recognition-binary-false.png)

你可以马上看到一些事情。 当`binary`是假的时候，它也选取了同样的东西，但是把`White House`这样的术语分解成`White`和`House`，就好像它们是不同的，而我们可以在`binary = True`的选项中看到，命名实体的识别 说`White House`是相同命名实体的一部分，这是正确的。

根据你的目标，你可以使用`binary `选项。 如果您的`binary `为`false`，这里是你可以得到的，命名实体的类型：

```
NE Type and Examples
ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
```

无论哪种方式，你可能会发现，你需要做更多的工作才能做到恰到好处，但是这个功能非常强大。

在接下来的教程中，我们将讨论类似于词干提取的东西，叫做“词形还原”（lemmatizing）。

## 八、NLTK 词形还原

与词干提权非常类似的操作称为词形还原。 这两者之间的主要区别是，你之前看到了，词干提权经常可能创造出不存在的词汇，而词形是实际的词汇。

所以，你的词干，也就是你最终得到的词，不是你可以在字典中查找的东西，但你可以查找一个词形。

有时你最后会得到非常相似的词语，但有时候，你会得到完全不同的词语。 我们来看一些例子。

```py
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better", pos="a"))
print(lemmatizer.lemmatize("best", pos="a"))
print(lemmatizer.lemmatize("run"))
print(lemmatizer.lemmatize("run",'v'))
```

在这里，我们有一些我们使用的词的词形的例子。 唯一要注意的是，`lemmatize `接受词性参数`pos`。 如果没有提供，默认是“名词”。 这意味着，它将尝试找到最接近的名词，这可能会给你造成麻烦。 如果你使用词形还原，请记住！

在接下来的教程中，我们将深入模块附带的 NTLK 语料库，查看所有优秀文档，他们在那里等待着我们。

## 九、 NLTK 语料库

在本教程的这一部分，我想花一点时间来深入我们全部下载的语料库！ NLTK 语料库是各种自然语言数据集，绝对值得一看。

NLTK 语料库中的几乎所有文件都遵循相同的规则，通过使用 NLTK 模块来访问它们，但是它们没什么神奇的。 这些文件大部分都是纯文本文件，其中一些是 XML 文件，另一些是其他格式文件，但都可以通过手动或模块和 Python 访问。 让我们来谈谈手动查看它们。

根据您的安装，您的`nltk_data`目录可能隐藏在多个位置。 为了找出它的位置，请转到您的 Python 目录，也就是 NLTK 模块所在的位置。 如果您不知道在哪里，请使用以下代码：

```py
import nltk
print(nltk.__file__)
```

运行它，输出将是 NLTK 模块`__init__.py`的位置。 进入 NLTK 目录，然后查找`data.py`文件。

代码的重要部分是：

```py
if sys.platform.startswith('win'):
    # Common locations on Windows:
    path += [
        str(r'C:\nltk_data'), str(r'D:\nltk_data'), str(r'E:\nltk_data'),
        os.path.join(sys.prefix, str('nltk_data')),
        os.path.join(sys.prefix, str('lib'), str('nltk_data')),
        os.path.join(os.environ.get(str('APPDATA'), str('C:\\')), str('nltk_data'))
    ]
else:
    # Common locations on UNIX & OS X:
    path += [
        str('/usr/share/nltk_data'),
        str('/usr/local/share/nltk_data'),
        str('/usr/lib/nltk_data'),
        str('/usr/local/lib/nltk_data')
    ]
```

在那里，你可以看到`nltk_data`的各种可能的目录。 如果你在 Windows 上，它很可能是在你的`appdata`中，在本地目录中。 为此，你需要打开你的文件浏览器，到顶部，然后输入`%appdata%`。

接下来点击`roaming`，然后找到`nltk_data`目录。 在那里，你将找到你的语料库文件。 完整的路径是这样的：

```
C:\Users\yourname\AppData\Roaming\nltk_data\corpora
```

在这里，你有所有可用的语料库，包括书籍，聊天记录，电影评论等等。

现在，我们将讨论通过 NLTK 访问这些文档。 正如你所看到的，这些主要是文本文档，所以你可以使用普通的 Python 代码来打开和阅读文档。 也就是说，NLTK 模块有一些很好的处理语料库的方法，所以你可能会发现使用他们的方法是实用的。 下面是我们打开“古腾堡圣经”，并阅读前几行的例子：

```py
from nltk.tokenize import sent_tokenize, PunktSentenceTokenizer
from nltk.corpus import gutenberg

# sample text
sample = gutenberg.raw("bible-kjv.txt")

tok = sent_tokenize(sample)

for x in range(5):
    print(tok[x])
```

其中一个更高级的数据集是`wordnet`。 Wordnet 是一个单词，定义，他们使用的例子，同义词，反义词，等等的集合。 接下来我们将深入使用 wordnet。

## 十、 NLTK 和 Wordnet

WordNet 是英语的词汇数据库，由普林斯顿创建，是 NLTK 语料库的一部分。

您可以一起使用 WordNet 和 NLTK 模块来查找单词含义，同义词，反义词等。 我们来介绍一些例子。

首先，你将需要导入`wordnet`：

```py
from nltk.corpus import wordnet
```

之后我们打算使用单词`program`来寻找同义词：

```py
syns = wordnet.synsets("program")
```

一个同义词的例子：

```py
print(syns[0].name())

# plan.n.01
```

只是单词：

```py
print(syns[0].lemmas()[0].name())

# plan
```

第一个同义词的定义：

```py
print(syns[0].definition())

# a series of steps to be carried out or goals to be accomplished
```

单词的使用示例：

```py
print(syns[0].examples())

# ['they drew up a six-step plan', 'they discussed plans for a new bond issue']
```

接下来，我们如何辨别一个词的同义词和反义词？ 这些词形是同义词，然后你可以使用`.antonyms`找到词形的反义词。 因此，我们可以填充一些列表，如：

```py
synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
print(set(antonyms))

'''
{'beneficial', 'just', 'upright', 'thoroughly', 'in_force', 'well', 'skilful', 'skillful', 'sound', 'unspoiled', 'expert', 'proficient', 'in_effect', 'honorable', 'adept', 'secure', 'commodity', 'estimable', 'soundly', 'right', 'respectable', 'good', 'serious', 'ripe', 'salutary', 'dear', 'practiced', 'goodness', 'safe', 'effective', 'unspoilt', 'dependable', 'undecomposed', 'honest', 'full', 'near', 'trade_good'} {'evil', 'evilness', 'bad', 'badness', 'ill'}
'''
```

你可以看到，我们的同义词比反义词更多，因为我们只是查找了第一个词形的反义词，但是你可以很容易地平衡这个，通过也为`bad`这个词执行完全相同的过程。

接下来，我们还可以很容易地使用 WordNet 来比较两个词的相似性和他们的时态，把 Wu 和 Palmer 方法结合起来用于语义相关性。

我们来比较名词`ship`和`boat`：

```py
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')
print(w1.wup_similarity(w2))

# 0.9090909090909091

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')
print(w1.wup_similarity(w2))

# 0.6956521739130435

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('cat.n.01')
print(w1.wup_similarity(w2))

# 0.38095238095238093
```

接下来，我们将讨论一些问题并开始讨论文本分类的主题。

## 十一、NLTK 文本分类

现在我们熟悉 NLTK 了，我们来尝试处理文本分类。 文本分类的目标可能相当宽泛。 也许我们试图将文本分类为政治或军事。 也许我们试图按照作者的性别来分类。 一个相当受欢迎的文本分类任务是，将文本的正文识别为垃圾邮件或非垃圾邮件，例如电子邮件过滤器。 在我们的例子中，我们将尝试创建一个情感分析算法。

为此，我们首先尝试使用属于 NLTK 语料库的电影评论数据库。 从那里，我们将尝试使用词汇作为“特征”，这是“正面”或“负面”电影评论的一部分。 NLTK 语料库`movie_reviews`数据集拥有评论，他们被标记为正面或负面。 这意味着我们可以训练和测试这些数据。 首先，让我们来预处理我们的数据。

```py
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words["stupid"])
```

运行此脚本可能需要一些时间，因为电影评论数据集有点大。 我们来介绍一下这里发生的事情。

导入我们想要的数据集后，您会看到：

```py
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
```

基本上，用简单的英文，上面的代码被翻译成：在每个类别（我们有正向和独享），选取所有的文件 ID（每个评论有自己的 ID），然后对文件 ID存储`word_tokenized`版本（单词列表），后面是一个大列表中的正面或负面标签。

接下来，我们用`random `来打乱我们的文件。这是因为我们将要进行训练和测试。如果我们把他们按序排列，我们可能会训练所有的负面评论，和一些正面评论，然后在所有正面评论上测试。我们不想这样，所以我们打乱了数据。

然后，为了你能看到你正在使用的数据，我们打印出`documents[1]`，这是一个大列表，其中第一个元素是一列单词，第二个元素是`pos`或`neg`标签。

接下来，我们要收集我们找到的所有单词，所以我们可以有一个巨大的典型单词列表。从这里，我们可以执行一个频率分布，然后找出最常见的单词。正如你所看到的，最受欢迎的“词语”其实就是标点符号，`the`，`a`等等，但是很快我们就会得到有效词汇。我们打算存储几千个最流行的单词，所以这不应该是一个问题。

```py
print(all_words.most_common(15))
```

以上给出了15个最常用的单词。 你也可以通过下面的步骤找出一个单词的出现次数：

```py
print(all_words["stupid"])
```

接下来，我们开始将我们的单词，储存为正面或负面的电影评论的特征。

## 十二、使用 NLTK 将单词转换为特征

在本教程中，我们在以前的视频基础上构建，并编撰正面评论和负面评论中的单词的特征列表，来看到正面或负面评论中特定类型单词的趋势。

最初，我们的代码：

```py
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]
```

几乎和以前一样，只是现在有一个新的变量，`word_features`，它包含了前 3000 个最常用的单词。 接下来，我们将建立一个简单的函数，在我们的正面和负面的文档中找到这些前 3000 个单词，将他们的存在标记为是或否：

```py
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features
```

下面，我们可以打印出特征集：

```py
print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
```

之后我们可以为我们所有的文档做这件事情，通过做下列事情，保存特征存在性布尔值，以及它们各自的正面或负面的类别：

```py
featuresets = [(find_features(rev), category) for (rev, category) in documents]
```

真棒，现在我们有了特征和标签，接下来是什么？ 通常，下一步是继续并训练算法，然后对其进行测试。 所以，让我们继续这样做，从下一个教程中的朴素贝叶斯分类器开始！

## 十三、NLTK 朴素贝叶斯分类器

现在是时候选择一个算法，将我们的数据分成训练和测试集，然后启动！我们首先要使用的算法是朴素贝叶斯分类器。这是一个非常受欢迎的文本分类算法，所以我们只能先试一试。然而，在我们可以训练和测试我们的算法之前，我们需要先把数据分解成训练集和测试集。

你可以训练和测试同一个数据集，但是这会给你带来一些严重的偏差问题，所以你不应该训练和测试完全相同的数据。为此，由于我们已经打乱了数据集，因此我们将首先将包含正面和负面评论的 1900 个乱序评论作为训练集。然后，我们可以在最后的 100 个上测试，看看我们有多准确。

这被称为监督机器学习，因为我们正在向机器展示数据，并告诉它“这个数据是正面的”，或者“这个数据是负面的”。然后，在完成训练之后，我们向机器展示一些新的数据，并根据我们之前教过计算机的内容询问计算机，计算机认为新数据的类别是什么。

我们可以用以下方式分割数据：

```py
# set that we'll train our classifier with
training_set = featuresets[:1900]

# set that we'll test against.
testing_set = featuresets[1900:]
```

下面，我们可以定义并训练我们的分类器：

```py
classifier = nltk.NaiveBayesClassifier.train(training_set)
```

首先，我们只是简单调用朴素贝叶斯分类器，然后在一行中使用`.train()`进行训练。

足够简单，现在它得到了训练。 接下来，我们可以测试它：

```py
print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testing_set))*100)
```

砰，你得到了你的答案。 如果你错过了，我们可以“测试”数据的原因是，我们仍然有正确的答案。 因此，在测试中，我们向计算机展示数据，而不提供正确的答案。 如果它正确猜测我们所知的答案，那么计算机是正确的。 考虑到我们所做的打乱，你和我可能准确度不同，但你应该看到准确度平均为 60-75%。

接下来，我们可以进一步了解正面或负面评论中最有价值的词汇：

```py
classifier.show_most_informative_features(15)
```

这对于每个人都不一样，但是你应该看到这样的东西：

```
Most Informative Features
insulting = True neg : pos = 10.6 : 1.0
ludicrous = True neg : pos = 10.1 : 1.0
winslet = True pos : neg = 9.0 : 1.0
detract = True pos : neg = 8.4 : 1.0
breathtaking = True pos : neg = 8.1 : 1.0
silverstone = True neg : pos = 7.6 : 1.0
excruciatingly = True neg : pos = 7.6 : 1.0
warns = True pos : neg = 7.0 : 1.0
tracy = True pos : neg = 7.0 : 1.0
insipid = True neg : pos = 7.0 : 1.0
freddie = True neg : pos = 7.0 : 1.0
damon = True pos : neg = 5.9 : 1.0
debate = True pos : neg = 5.9 : 1.0
ordered = True pos : neg = 5.8 : 1.0
lang = True pos : neg = 5.7 : 1.0
```

这个告诉你的是，每一个词的负面到正面的出现几率，或相反。 因此，在这里，我们可以看到，负面评论中的`insulting`一词比正面评论多出现 10.6 倍。`Ludicrous`是 10.1。

现在，让我们假设，你完全满意你的结果，你想要继续，也许使用这个分类器来预测现在的事情。 训练分类器，并且每当你需要使用分类器时，都要重新训练，是非常不切实际的。 因此，您可以使用`pickle`模块保存分类器。 我们接下来做。

## 十四、使用 NLTK 保存分类器

训练分类器和机器学习算法可能需要很长时间，特别是如果您在更大的数据集上训练。 我们的其实很小。 你可以想象，每次你想开始使用分类器的时候，都要训练分类器吗？ 这么恐怖！ 相反，我们可以使用`pickle`模块，并序列化我们的分类器对象，这样我们所需要做的就是简单加载该文件。

那么，我们该怎么做呢？ 第一步是保存对象。 为此，首先需要在脚本的顶部导入`pickle`，然后在使用`.train()`分类器进行训练后，可以调用以下几行：

```py
save_classifier = open("naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()
```

这打开了一个`pickle`文件，准备按字节写入一些数据。 然后，我们使用`pickle.dump()`来转储数据。 `pickle.dump()`的第一个参数是你写入的东西，第二个参数是你写入它的地方。

之后，我们按照我们的要求关闭文件，这就是说，我们现在在脚本的目录中保存了一个`pickle`或序列化的对象！

接下来，我们如何开始使用这个分类器？ `.pickle`文件是序列化的对象，我们现在需要做的就是将其读入内存，这与读取任何其他普通文件一样简单。 这样做：

```py
classifier_f = open("naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()
```

在这里，我们执行了非常相似的过程。 我们打开文件来读取字节。 然后，我们使用`pickle.load()`来加载文件，并将数据保存到分类器变量中。 然后我们关闭文件，就是这样。 我们现在有了和以前一样的分类器对象！

现在，我们可以使用这个对象，每当我们想用它来分类时，我们不再需要训练我们的分类器。

虽然这一切都很好，但是我们可能不太满意我们所获得的 60-75% 的准确度。 其他分类器呢？ 其实，有很多分类器，但我们需要 scikit-learn（sklearn）模块。 幸运的是，NLTK 的员工认识到将 sklearn 模块纳入 NLTK 的价值，他们为我们构建了一个小 API。 这就是我们将在下一个教程中做的事情。

## 十五、NLTK 和 Sklearn

现在我们已经看到，使用分类器是多么容易，现在我们想尝试更多东西！ Python 的最好的模块是 Scikit-learn（sklearn）模块。

如果您想了解 Scikit-learn 模块的更多信息，我有一些关于 Scikit-Learn 机器学习的教程。

幸运的是，对于我们来说，NLTK 背后的人们更看重将 sklearn 模块纳入NLTK分类器方法的价值。 就这样，他们创建了各种`SklearnClassifier` API。 要使用它，你只需要像下面这样导入它：

```py
from nltk.classify.scikitlearn import SklearnClassifier
```

从这里开始，你可以使用任何`sklearn`分类器。 例如，让我们引入更多的朴素贝叶斯算法的变体：

```py
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
```

之后，如何使用它们？结果是，这非常简单。

```py
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MultinomialNB accuracy percent:",nltk.classify.accuracy(MNB_classifier, testing_set))

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BernoulliNB accuracy percent:",nltk.classify.accuracy(BNB_classifier, testing_set))

```

就是这么简单。让我们引入更多东西：

```py
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
```

现在，我们所有分类器应该是这样：

```py
print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
```

运行它的结果应该是这样：

```
Original Naive Bayes Algo accuracy percent: 63.0
Most Informative Features
                thematic = True              pos : neg    =      9.1 : 1.0
                secondly = True              pos : neg    =      8.5 : 1.0
                narrates = True              pos : neg    =      7.8 : 1.0
                 rounded = True              pos : neg    =      7.1 : 1.0
                 supreme = True              pos : neg    =      7.1 : 1.0
                 layered = True              pos : neg    =      7.1 : 1.0
                  crappy = True              neg : pos    =      6.9 : 1.0
               uplifting = True              pos : neg    =      6.2 : 1.0
                     ugh = True              neg : pos    =      5.3 : 1.0
                   mamet = True              pos : neg    =      5.1 : 1.0
                 gaining = True              pos : neg    =      5.1 : 1.0
                   wanda = True              neg : pos    =      4.9 : 1.0
                   onset = True              neg : pos    =      4.9 : 1.0
               fantastic = True              pos : neg    =      4.5 : 1.0
                kentucky = True              pos : neg    =      4.4 : 1.0
MNB_classifier accuracy percent: 66.0
BernoulliNB_classifier accuracy percent: 72.0
LogisticRegression_classifier accuracy percent: 64.0
SGDClassifier_classifier accuracy percent: 61.0
SVC_classifier accuracy percent: 45.0
LinearSVC_classifier accuracy percent: 68.0
NuSVC_classifier accuracy percent: 59.0
```

所以，我们可以看到，SVC 的错误比正确更常见，所以我们可能应该丢弃它。 但是呢？ 接下来我们可以尝试一次使用所有这些算法。 一个算法的算法！ 为此，我们可以创建另一个分类器，并根据其他算法的结果来生成分类器的结果。 有点像投票系统，所以我们只需要奇数数量的算法。 这就是我们将在下一个教程中讨论的内容。

## 十六、使用 NLTK 组合算法

现在我们知道如何使用一堆算法分类器，就像糖果岛上的一个孩子，告诉他们只能选择一个，我们可能会发现很难只选择一个分类器。 好消息是，你不必这样！ 组合分类器算法是一种常用的技术，通过创建一种投票系统来实现，每个算法拥有一票，选择得票最多分类。

为此，我们希望我们的新分类器的工作方式像典型的 NLTK 分类器，并拥有所有方法。 很简单，使用面向对象编程，我们可以确保从 NLTK 分类器类继承。 为此，我们将导入它：

```py
from nltk.classify import ClassifierI
from statistics import mode
```

我们也导入`mode`（众数），因为这将是我们选择最大计数的方法。

现在，我们来建立我们的分类器类：

```py
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
```

我们把我们的类叫做`VoteClassifier`，我们继承了 NLTK 的`ClassifierI`。 接下来，我们将传递给我们的类的分类器列表赋给`self._classifiers`。

接下来，我们要继续创建我们自己的分类方法。 我们打算把它称为`.classify`，以便我们可以稍后调用`.classify`，就像传统的 NLTK 分类器那样。

```py
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
```

很简单，我们在这里所做的就是，遍历我们的分类器对象列表。 然后，对于每一个，我们要求它基于特征分类。 分类被视为投票。 遍历完成后，我们返回`mode(votes)`，这只是返回投票的众数。

这是我们真正需要的，但是我认为另一个参数，置信度是有用的。 由于我们有了投票算法，所以我们也可以统计支持和反对票数，并称之为“置信度”。 例如，3/5 票的置信度弱于 5/5 票。 因此，我们可以从字面上返回投票比例，作为一种置信度指标。 这是我们的置信度方法：

```py
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
```

现在，让我们把东西放到一起：

```py
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
        
training_set = featuresets[:1900]
testing_set =  featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open("naivebayes.pickle","rb")
classifier = pickle.load(classifier_f)
classifier_f.close()




print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

##SVC_classifier = SklearnClassifier(SVC())
##SVC_classifier.train(training_set)
##print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)


voted_classifier = VoteClassifier(classifier,
                                  NuSVC_classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

```

所以到了最后，我们对文本运行一些分类器示例。我们所有输出：

```
Original Naive Bayes Algo accuracy percent: 66.0
Most Informative Features
                thematic = True              pos : neg    =      9.1 : 1.0
                secondly = True              pos : neg    =      8.5 : 1.0
                narrates = True              pos : neg    =      7.8 : 1.0
                 layered = True              pos : neg    =      7.1 : 1.0
                 rounded = True              pos : neg    =      7.1 : 1.0
                 supreme = True              pos : neg    =      7.1 : 1.0
                  crappy = True              neg : pos    =      6.9 : 1.0
               uplifting = True              pos : neg    =      6.2 : 1.0
                     ugh = True              neg : pos    =      5.3 : 1.0
                 gaining = True              pos : neg    =      5.1 : 1.0
                   mamet = True              pos : neg    =      5.1 : 1.0
                   wanda = True              neg : pos    =      4.9 : 1.0
                   onset = True              neg : pos    =      4.9 : 1.0
               fantastic = True              pos : neg    =      4.5 : 1.0
                   milos = True              pos : neg    =      4.4 : 1.0
MNB_classifier accuracy percent: 67.0
BernoulliNB_classifier accuracy percent: 67.0
LogisticRegression_classifier accuracy percent: 68.0
SGDClassifier_classifier accuracy percent: 57.99999999999999
LinearSVC_classifier accuracy percent: 67.0
NuSVC_classifier accuracy percent: 65.0
voted_classifier accuracy percent: 65.0
Classification: neg Confidence %: 100.0
Classification: pos Confidence %: 57.14285714285714
Classification: neg Confidence %: 57.14285714285714
Classification: neg Confidence %: 57.14285714285714
Classification: pos Confidence %: 57.14285714285714
Classification: pos Confidence %: 85.71428571428571
```

