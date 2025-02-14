# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
    """Checks whether the casing config is consistent with the checkpoint name."""

    # The casing has to be passed in by the user and there is no explicit check
    # as to whether it matches the checkpoint. The casing information probably
    # should have been stored in the bert_config.json file, but it's not, so
    # we have to heuristically detect it to validate.

    if not init_checkpoint:
        return

    m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
    if m is None:
        return

    model_name = m.group(1)

    lower_models = [
        "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
        "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
    ]

    cased_models = [
        "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
        "multi_cased_L-12_H-768_A-12"
    ]

    is_bad_config = False
    if model_name in lower_models and not do_lower_case:
        is_bad_config = True
        actual_flag = "False"
        case_name = "lowercased"
        opposite_flag = "True"

    if model_name in cased_models and do_lower_case:
        is_bad_config = True
        actual_flag = "True"
        case_name = "cased"
        opposite_flag = "False"

    if is_bad_config:
        raise ValueError(
            "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
            "However, `%s` seems to be a %s model, so you "
            "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
            "how the model was pre-training. If this error is wrong, please "
            "just comment out this check." % (actual_flag, init_checkpoint,
                                              model_name, case_name, opposite_flag))


###################################################################################################
#
# 将输入 `text` 转化为 unicode 字符串：
# 1. 如果它已经是unicode字符串，则直接返回;
# 2. 否则，假设它为UTF-8编码，解码为unicode字符串.
#
###################################################################################################
def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


###################################################################################################
#
# 从文件中加载vocab，其中文件的格式为：每一行一个token，返回一个vocab字典{token: id}
#
###################################################################################################
def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with tf.gfile.GFile(vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


###################################################################################################
#
# 将输入的token序列转化为对应的id序列, 或者将输入的id序列转化为token序列，取决于输入的内容.
#
###################################################################################################
def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output

###################################################################################################
#
# 将输入的token序列转化为对应id序列，通过查词典的方式.
#
###################################################################################################
def convert_tokens_to_ids(vocab, tokens):
    return convert_by_vocab(vocab, tokens)


###################################################################################################
#
# 将输入的id序列转化为对应token序列，通过查逆向词典的方式.
#
###################################################################################################
def convert_ids_to_tokens(inv_vocab, ids):
    return convert_by_vocab(inv_vocab, ids)


###################################################################################################
#
# 以空格为分隔符将输入文本 `text` tokenize化.
#
###################################################################################################
def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


###################################################################################################
#
# 对输入的 `text` 文本进行端到端的tokenize, 步骤如下:
# 1. 加载wordpiece的词表vocab
# 2. 首先对输入text进行基本的tokenize，得到char粒度的token序列（中文为单个汉字，英文为单个word)
# 3. 对步骤2中的每个token通过查表vocab得到wordpiece
#
###################################################################################################
class FullTokenizer(object):
    """Runs end-to-end tokenziation."""

    def __init__(self, vocab_file, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


###################################################################################################
#
# BasicTokenizer: 对一段文本进行基本的数据清洗后，按照word粒度（中文按照char)粒度进行分割，不涉及词表vocab问题.
#
###################################################################################################
class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
        self.do_lower_case = do_lower_case

    ###############################################################################################
    #
    # Basic Tokenizer:
    # 1. 将文本转化成unicode文本，文本清洗(移除控制字符，空白字符\t\r\n统一为空格)
    # 2. 在中文字符左右两侧增加空格 ("S中国人E" ==> "S 中 国 人 E")
    # 3. 按照空格进行split得到原始的token序列orig_tokens; 如果do_lower_case=True，转入步骤4；否则转入步骤5.
    # 4. 如果设置do_lower_case=True，则将orig_tokens中的每个token进行小写化，并移除重音; 转入步骤5.
    # 5. 将origin_tokens中的每个token按照标点符号进行split (例如: "hello," ==> ["hello", ","]), 追加到tokens序列
    #
    ###############################################################################################
    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)

        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    # 将输入的token按照标点符号分割为token序列，例如: "world," ==> ["world", ","]
    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    # 在单个中文字符的两侧增加空格字符. 例如: "S我是中国人E" --> "S 我  是  中  国  人 E"
    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    # 判断一个字符是否为中文字符，这里中文字符定义为CJK Unicode Block内的所有字符.
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    # 移除基本的控制字符，对所有的空白字符进行归一化，保持普通字符和标点符号不变.
    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


###################################################################################################
#
# 对一个token序列进行wordpiece化，将每个token按照给定的vocab进行最长匹配查找其对应的wordpiece.
#
###################################################################################################
class WordpieceTokenizer(object):
    """Runs WordPiece tokenziation."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

        text = convert_to_unicode(text)

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            # 如果当前token的长度已经大于max_input_chars_per_word，则将该token看作是unk，加入output_tokens.
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            ########################################################################################
            #
            # 单个token的WordPiece:
            # 1. 初始化: start指向token的起始位置，end指向token的最后一个字符的下一个位置
            # 2. 固定start指针不变，end指针不断前移，判断[start, end)之间的子串substr是否位于vocab内；如果不在
            #    vocab内则，end指针减一，直到start==end，说明vocab不存在这样的substr，因此是一个bad token，置is_bad=True
            # 3. 如果[start,end)之间的substr在vocab内，则将该追加到最终的output_tokens列表;
            # 4. 然后让start指针指向end位置, 如果start < len(chars)，则回到步骤2继续执行
            # 5. 直到start==len(chars)，则遍历完成整个token中的所有chars，返回output_tokens
            #
            ########################################################################################
            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens


###################################################################################################
#
# 判断一个字符是否是空白字符, 其中空白字符定义为:
# " ", "\t", "\r", "\n", 以及其他Unicode标准中General_Category为Zs的空白字符.
#
###################################################################################################
def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


###################################################################################################
#
# 判断一个字符是否是控制字符, 其中控制字符定义为:
# Unicode标准中General_Category为Control(Cc)、Format(Cf)的字符.
#
###################################################################################################
def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat in ("Cc", "Cf"):
        return True
    return False


###################################################################################################
#
# 判断一个字符是否是标点符号, 其中标点符号定义为:
# 1. ASCII中所有的非字母、数字符号，包括 "^", "$", "`"等
# 2. Unicode标准中General_Category以"_Punctuation"结尾的字符.
#
###################################################################################################
def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False
