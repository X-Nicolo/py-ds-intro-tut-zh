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

现在我们已经学会了，如何执行一些自定义的分块和添加缝隙，我们来讨论一下 NLTK 自带的分块形式，这叫做实体识别。
