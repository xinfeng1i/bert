# coding: utf-8

# Author: xinfengli
# Date: 2019/07/24

"""
该文件用于从输入文件列表中产生能用于BERT训练的wordpiece的词表, 其中输入文件的格式为：
one sentence a line.
"""

import re
import os
import six
import unicodedata
import sentencepiece as spm
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
logger = logging.getLogger("create_wordpiece_vocab")

SENTENCEPIECE_EXCLUDE_WORDS = set(["<unk>", "<s>", "</s>"])


def train_sentencepiece_model(input_file, output_dir, sentencepiece_vocab_size,
                              character_coverage=0.995, model_type="unigram"):
    """
    训练wordpiece模型
    :param input_file: 输入文件，应该是经过basictoken后的文件，可以是单个文件或者以 `,` 隔开的文件列表
    :param sentencepiece_model_prefix:  输出模型路径
    :param sentencepiece_vocab_size: 输出的sentencepiece 词表大小
    :param character_coverage: 字符覆盖率
    :param model_type: 模型类型，支持参数 "unigram", "bpe", "char", "word"
    :return:
    """
    output_prefix = os.path.join(output_dir, "sentencepiece")
    logger.info("sentencepiece model prefix: %s" % output_prefix)

    cmd_str = '--input=%s --model_prefix=%s --vocab_size=%d, --character_coverage=%f, --model_type=%s --input_sentence_size=10000000 --shuffle_input_sentence=true' % (
        input_file, output_prefix, sentencepiece_vocab_size, character_coverage, model_type)
    spm.SentencePieceTrainer.Train(cmd_str)


def load_sentencepiece_vocab(sentencepiece_vocab_file):
    sp_vocab = []

    with open(sentencepiece_vocab_file, "r", encoding="utf-8") as fr:
        for i, line in enumerate(fr, 1):
            line = line.rstrip("\r\n")
            cols = line.split("\t")

            if len(cols) != 2:
                logger.warning("Line %d: wrong fields" %i)
                continue

            word = cols[0].strip()
            if word not in SENTENCEPIECE_EXCLUDE_WORDS:
                sp_vocab.append(word)

    return sp_vocab


def run_basic_tokenization(input_file, output_file, do_lower=False):
    basic_tokenizer = BasicTokenizer(do_lower)
    START_WITH_NUM_LETTER_PATTERN = "^[0-9a-zA-Z]+.*"

    with open(input_file, "r", encoding="utf-8") as fr, open(output_file, "w", encoding="utf-8") as fw:
        for i, line in enumerate(fr, 1):
            if line.strip() == "":
                continue
            else:
                output_tokens = basic_tokenizer.tokenize(line.strip())
                outline = " ".join(output_tokens).strip()
                flag = re.match(START_WITH_NUM_LETTER_PATTERN, outline)
                if flag:
                    # 如果以字母数字开头，则增加空格，保证sentencepiece算法能正确识别句首词语为whole word
                    outline = " " + outline + "\n"
                else:
                    outline = outline + "\n"

                fw.write(outline)

                if i % 50000 == 0:
                    logger.info("run basic token progress on file [%s] line: %d" % (input_file, i))


def convert_sentencepiece_to_wordpiece(sentencepiece_vocab, unused_token_num, do_lower=False):
    _basic_tokenizer = BasicTokenizer(do_lower)

    first_list = ["[PAD]"]
    unused_list = ["[unused%d]" % i for i in range(1, unused_token_num + 1)]
    third_list = ["[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"]
    NUM_LIST = [str(i) for i in range(0, 10)]
    LOWER_LETTERS = [chr(i) for i in range(97, 97 + 26)]
    UPPER_LETTERS = [chr(i) for i in range(65, 65 + 26)]

    NON_SUBWORD_PATTERN = "[\u4E00-\u9FFF\u3400-\u4DBF\uF900-\uFAFF]+"

    # 将sentencepiece的token转化为wordpiece的token
    clean_vocab = []
    for word in sentencepiece_vocab:
        if word.startswith("▁"):
            word = word[1:]
        else:
            flag = re.match(NON_SUBWORD_PATTERN, word)
            if flag or (len(word) == 1 and _is_punctuation(word)):
                word = word  # 如果是中文或者标点符号，则不算作subword
            else:
                word = "##" + word

        if word.strip() != "":
            clean_vocab.append(word.strip())

    # 预留字符vocab
    seen = set([])
    reserved_vocab = first_list + unused_list + third_list + NUM_LIST + LOWER_LETTERS + UPPER_LETTERS

    # 最终vocab
    result_vocab = []
    for word in reserved_vocab:
        if word not in seen:
            result_vocab.append(word)
            seen.add(word)

    for word in clean_vocab:
        if word not in seen:
            result_vocab.append(word)
            seen.add(word)

    return result_vocab


def save_vocab_to_file(vocab, output_file):
    with open(output_file, "w", encoding="utf-8") as fw:
        for word in vocab:
            outline = word + "\n"
            fw.write(outline)


def stat_token_distribution(input_files, output_file):
    input_file_list = input_files.split(",")
    input_file_list = [fn.strip() for fn in input_file_list if fn.strip() != ""]

    logger.info("*** read from files ***")
    for fn in input_file_list:
        logger.info("  %s" % fn)

    basic_tokenizer = BasicTokenizer(False)

    total_sentence_num = 0
    total_doc_num = 0
    total_tokens_num = 0
    for fn in input_file_list:
        with open(fn, "r", encoding="utf-8") as fr:
            for i, line in enumerate(fr, 1):
                line = line.strip()
                if line != "":
                    total_sentence_num += 1
                    output_tokens = basic_tokenizer.tokenize(line)
                    total_tokens_num += len(output_tokens)
                else:
                    total_doc_num += 1

                if i % 50000 == 0:
                    logger.info("Progress: file %s, line %d" % (fn, i))

    with open(output_file, "w", encoding="utf-8") as fw:
        fw.write("total_sentence_num: %d\n" % total_sentence_num)
        fw.write("total_doc_num: %d\n" % total_doc_num)
        fw.write("total_tokens_num: %d\n" % total_tokens_num)


###################################################################################################
#
# The following code is copied from Google BERT source code file `tokenization.py`
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


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


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


def run(input_files, output_dir, vocab_size, do_lower,
        number_of_placeholder=100, character_coverage=0.995, model_type="unigram"):
    input_file_names = input_files.split(",")
    input_file_names = [fn.strip() for fn in input_file_names if fn.strip() != ""]

    logger.info("*** input files ***")
    for name in input_file_names:
        logger.info(" %s" % name)

    assert(os.path.exists(output_dir))
    base_name_list = [os.path.basename(ph) for ph in input_file_names]
    base_name_without_ext = [os.path.splitext(name)[0] for name in base_name_list]
    output_file_names = [os.path.join(output_dir, name + "_tokenized.txt") for name in base_name_without_ext]

    logger.info("*** output files ***")
    for name in output_file_names:
        logger.info(" %s" % name)

    assert(len(input_file_names) == len(output_file_names))

    logger.info("*** run basic tokenizer ***")
    for i in range(len(input_file_names)):
        infile = input_file_names[i]
        outfile = output_file_names[i]
        run_basic_tokenization(infile, outfile, do_lower)

    logger.info("*** run sentencepiece algorithm ***")
    train_sentencepiece_model(",".join(output_file_names), output_dir, vocab_size, character_coverage, model_type)

    logger.info("*** convert sentencepiece vocab to wordpiece vocab ***")
    sentencepiece_vocab_name = os.path.join(output_dir, "sentencepiece.vocab")
    wordpiece_vocab_name = os.path.join(output_dir, "wordpiece.vocab")

    sp_vocab = load_sentencepiece_vocab(sentencepiece_vocab_name)
    wp_vocab = convert_sentencepiece_to_wordpiece(sp_vocab, number_of_placeholder, do_lower)
    save_vocab_to_file(wp_vocab, wordpiece_vocab_name)

    logger.info("*** All finished. ***")


if __name__ == "__main__":
    # run("dataset/full_data/bert_input_part1.txt,dataset/full_data/bert_input_part2.txt,"
    #     "dataset/full_data/bert_input_part3.txt,dataset/full_data/bert_input_part4.txt,"
    #     "dataset/full_data/bert_input_part5.txt,dataset/full_data/bert_input_part6.txt",
    #     "output_vocab/", 60000, False, 1000)

    # train_sentencepiece_model("output_vocab/bert_input_part1_tokenized.txt,"
    #                           "output_vocab/bert_input_part2_tokenized.txt,"
    #                           "output_vocab/bert_input_part3_tokenized.txt,"
    #                           "output_vocab/bert_input_part4_tokenized.txt,"
    #                           "output_vocab/bert_input_part5_tokenized.txt",
    #                           "output_vocab/", 60000, 0.9995, "unigram")
    pass
