# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

"""Tokenization classes for LLaMA."""

import os
from shutil import copyfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import json
import sentencepiece as spm
import transformers
from transformers.convert_slow_tokenizer import import_protobuf
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


if TYPE_CHECKING:
    from transformers.tokenization_utils_base import TextInput

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.model"}

SPIECE_UNDERLINE = "▁"

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# fmt: off
DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your \
answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure\
 that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not \
correct. If you don't know the answer to a question, please don't share false information."""
# fmt: on

def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self, data):
        self.head = None
        self.tail = None
        self._build_list(data)

    def _build_list(self, data):
        """Build a doubly linked list from the provided data (list of characters)."""
        for char in data:
            self.append(char)

    def append(self, value):
        """Append a new node to the linked list."""
        new_node = Node(value)
        if self.tail:
            self.tail.next = new_node
            new_node.prev = self.tail
        else:
            self.head = new_node
        self.tail = new_node

    def search_and_replace(self, word):
        """Search for the word in the linked list and replace its occurrence."""
        current = self.head
        while current:
            # Check if we can match the word starting from 'current'
            match_start = current
            match_node = current
            i = 0
            while match_node and i < len(word) and match_node.value == word[i]:
                match_node = match_node.next
                i += 1
            
            if i == len(word):  # Full match found
                # If match_node is None, we have reached the end of the list
                self._replace(match_start, match_node, word)
            current = current.next

    def _replace(self, start_node, end_node, word):
        """Replace the nodes from start_node to end_node with a single node containing 'word'."""
        # Create a new node for the word
        new_node = Node(word)
        
        # Connect new_node in place of the found sequence
        if start_node.prev:
            start_node.prev.next = new_node
            new_node.prev = start_node.prev
        else:
            self.head = new_node
        
        if end_node:
            end_node.prev = new_node
            new_node.next = end_node
        else:
            self.tail = new_node

    def to_list(self):
        """Convert the linked list back to a regular list."""
        result = []
        current = self.head
        while current:
            result.append(current.value)
            current = current.next
        return result

class LlamaTokenizer(PreTrainedTokenizer):
    """
    Construct a Llama tokenizer. Based on byte-level Byte-Pair-Encoding. The default padding token is unset as there is
    no padding token in the original model.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<s>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"</s>"`):
            The end of sequence token.
        pad_token (`str` or `tokenizers.AddedToken`, *optional*):
            A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by
            attention mechanisms or loss computation.
        sp_model_kwargs (`Dict[str, Any]`, `Optional`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other things,
            to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is infinite and samples from the all hypothesis (lattice)
                using forward-filtering-and-backward-sampling algorithm.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merge operations for
              BPE-dropout.

        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Llama should be used.
        spaces_between_special_tokens (`bool`, *optional*, defaults to `False`):
            Whether or not to add spaces between special tokens.
        legacy (`bool`, *optional*):
            Whether or not the `legacy` behavior of the tokenizer should be used. Legacy is before the merge of #24622
            and #25224 which includes fixes to properly handle tokens that appear after special tokens.
            Make sure to also set `from_slow` to `True`.
            A simple example:

            - `legacy=True`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=True, from_slow=True)
            >>> tokenizer.encode("Hello <s>.") # 869 is '▁.'
            [1, 15043, 29871, 1, 869]
            ```
            - `legacy=False`:
            ```python
            >>> from transformers import LlamaTokenizerFast

            >>> tokenizer = LlamaTokenizerFast.from_pretrained("huggyllama/llama-7b", legacy=False, from_slow=True)
            >>> tokenizer.encode("Hello <s>.")  # 29889 is '.'
            [1, 15043, 29871, 1, 29889]
            ```
            Checkout the [pull request](https://github.com/huggingface/transformers/pull/24565) for more details.
        add_prefix_space (`bool`, *optional*, defaults to `True`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. Again, this should be set with `from_slow=True` to make sure it's taken into account.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        added_vocab=None,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token=None,
        sp_model_kwargs: Optional[Dict[str, Any]] = None,
        add_bos_token=True,
        add_eos_token=False,
        clean_up_tokenization_spaces=False,
        use_default_system_prompt=False,
        spaces_between_special_tokens=False,
        legacy=None,
        add_prefix_space=True,
        **kwargs,
    ):
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs
        bos_token = AddedToken(bos_token, normalized=False, special=True) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, normalized=False, special=True) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, normalized=False, special=True) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, normalized=False, special=True) if isinstance(pad_token, str) else pad_token

        if legacy is None:
            logger.warning_once(
                f"You are using the default legacy behaviour of the {self.__class__}. This is"
                " expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you."
                " If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it"
                " means, and thoroughly read the reason why this was added as explained in"
                " https://github.com/huggingface/transformers/pull/24565 - if you loaded a llama tokenizer from a GGUF file"
                " you can ignore this message"
            )
            legacy = True

        self.legacy = legacy
        self.vocab_file = vocab_file
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.use_default_system_prompt = use_default_system_prompt
        self.sp_model = self.get_spm_processor(kwargs.pop("from_slow", False))
        self.add_prefix_space = add_prefix_space
        
        self.added_vocab = {line.strip():32000+idx for idx,line in enumerate(open(added_vocab, 'r').readlines())}
        for i in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
            self.added_vocab[i] = len(self.added_vocab)
        self.added_vocab[SPIECE_UNDERLINE] = len(self.added_vocab)
        
        self.byte_encoder = bytes_to_unicode()
        merges_file = os.path.join(os.path.dirname(vocab_file), 'merges.txt')
        if not os.path.exists(merges_file):
            tokenizer_file = os.path.join(os.path.dirname(vocab_file), 'tokenizer.json')
            with open(tokenizer_file, 'r') as f:
                data = json.load(f)
            f.close()
            with open(merges_file, 'w') as f:
                for d in data['model']['merges']:
                    f.write(d + '\n')
            del data
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            sp_model_kwargs=self.sp_model_kwargs,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            use_default_system_prompt=use_default_system_prompt,
            spaces_between_special_tokens=spaces_between_special_tokens,
            legacy=legacy,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.get_spm_processor
    def get_spm_processor(self, from_slow=False):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        if self.legacy or from_slow:  # no dependency on protobuf
            tokenizer.Load(self.vocab_file)
            return tokenizer

        with open(self.vocab_file, "rb") as f:
            sp_model = f.read()
            model_pb2 = import_protobuf(f"The new behaviour of {self.__class__.__name__} (with `self.legacy = False`)")
            model = model_pb2.ModelProto.FromString(sp_model)
            normalizer_spec = model_pb2.NormalizerSpec()
            normalizer_spec.add_dummy_prefix = False
            model.normalizer_spec.MergeFrom(normalizer_spec)
            sp_model = model.SerializeToString()
            tokenizer.LoadFromSerializedProto(sp_model)
        return tokenizer

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        state["sp_model_proto"] = self.sp_model.serialized_model_proto()
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.LoadFromSerializedProto(self.sp_model_proto)

    @property
    def vocab_size(self):
        """Returns vocab size"""
        return self.sp_model.get_piece_size()

    def get_vocab(self):
        """Returns vocab as a dict"""
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.tokenize
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Converts a string to a list of tokens. If `self.legacy` is set to `False`, a prefix token is added unless the
        first token is special.
        """
        if self.legacy or len(text) == 0:
            return super().tokenize(text, **kwargs)

        text = text.replace(SPIECE_UNDERLINE, " ")
        if self.add_prefix_space:
            text = SPIECE_UNDERLINE + text
        tokens = super().tokenize(text, **kwargs)
        # print(tokens)

        if len(tokens) > 1 and tokens[0] == SPIECE_UNDERLINE and tokens[1] in self.all_special_tokens:
            tokens = tokens[1:]
        return tokens

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer._tokenize
    def _tokenize(self, text, **kwargs):
        """
        Returns a tokenized string.

        Using Basic Bpe instead of SentencePiece for AdaptBPE
        """
        bpe_tokens = []
        for token in text.split():
            token = SPIECE_UNDERLINE + token
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens


    # O(n^2)
    def merge_added_vocab(self, token_list, added_vocab):
        result = []
        if added_vocab and os.path.exists(added_vocab):
            ext = added_vocab.split('.')[-1]
            if ext=='txt':
                with open(added_vocab, 'r') as f:
                    vocabulary= [line.strip() for line in f]
            elif ext=='json':
                with open(added_vocab, 'r') as f:
                    vocabulary = json.load(f)['vocab']
            else:
                raise Exception
        else:
            vocabulary = []
        vocab_set = set(vocabulary)  # Convert vocabulary list to a set for faster lookup
        i = 0
        while i < len(token_list):
            match_found = False
            # Try to find the longest match starting from the current index
            for j in range(len(token_list), i, -1):
                candidate = ''.join(token_list[i:j])
                if candidate in vocab_set:
                    result.append(candidate)
                    i = j  # Move the index past the matched tokens
                    match_found = True
                    break  # Longest match found, break the inner loop
            if not match_found:
                # No match found; add the current token as is
                result.append(token_list[i])
                i += 1  # Move to the next token
        return tuple(result)
    
    # O(n^2) without the exception
    def _merge_added_vocab(self,token_list, added_vocab):
        if added_vocab and os.path.exists(added_vocab):
            ext = added_vocab.split('.')[-1]
            if ext=='txt':
                with open(added_vocab, 'r') as f:
                    vocabulary= [line.strip() for line in f]
            elif ext=='json':
                with open(added_vocab, 'r') as f:
                    vocabulary = json.load(f)['vocab']
            else:
                raise Exception
        else:
            vocabulary = []
        vocabulary = sorted(vocabulary, key=len, reverse=True)
        linked_list = DoublyLinkedList(token_list)

        for word in vocabulary:
            linked_list.search_and_replace(word)

        return linked_list.to_list()
    
    def max_subword_split(self, w):     
        for l in range(len(w), 0, -1):
            for i in range(0, len(w) - l + 1):
                if w[i] == '-':
                    continue
                subword = w[i:i + l]
                if subword in self.added_vocab: return subword, w[:i] + l * '-' + w[i + l:], i
        return None, None, None
    
    '''
    this functino calls max_subword_split recursively to get the longest match from the vocab also 
    replacing the match with hyphens so that the matche part is not considered for further search.
    '''
    def get_recursive_dict(self, w, k=100):
        max_subword, rest, i = self.max_subword_split(w)
        if max_subword is None:
            return dict()
        if k == 1 or rest == len(rest) * '-':
            adapt_dict = {i: max_subword}
            return adapt_dict
        adapt_dict = self.get_recursive_dict(rest, k - 1)
        adapt_dict[i] = max_subword
        return adapt_dict

    '''
    this functino calls get_recursive_dict to get the longest match from the vocab and returns
    the subwords in the order of their occurence in the word.
    '''
    def added_tokenize(self, w):
        adapt_dict = self.get_recursive_dict(w,k=100)
        if len(adapt_dict) > 0: 
            if len(''.join([subword for i, subword in sorted(adapt_dict.items())])) == len(w): 
                return [subword for i, subword in sorted(adapt_dict.items())]
            else: return tuple(w)
        else: return tuple(w)

    '''
    The three functions above togther are used to change the intialzation step of BPE 
    with that of added_tokenize
    '''
    def bpe(self, token):
        # print(token)
        if token in self.cache:
            return self.cache[token]
        
        word = self.added_tokenize(token) #this is the replacement for initializtion of BPE  
        # word = tuple(token)  this was how BPE used to initialize. Uncomment this line and comment the above line to use default BPE.             
    
        pairs = get_pairs(word)
        
        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            # print(first,second)
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    
    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self.sp_model.IdToPiece(index)
        return token

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        # since we manually add the prefix space, we have to remove it when decoding
        if tokens[0].startswith(SPIECE_UNDERLINE) and self.add_prefix_space:
            tokens[0] = tokens[0][1:]

        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for i, token in enumerate(tokens):
            # make sure that special tokens are not decoded using sentencepiece model
            if token in self.all_special_tokens:
                if not prev_is_special and i != 0 and self.legacy:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                if prev_is_special and i == 1 and self.add_prefix_space and not token.startswith(SPIECE_UNDERLINE):
                    out_string += " "
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        return out_string

    def save_vocabulary(self, save_directory, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the files saved.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                content_spiece_model = self.sp_model.serialized_model_proto()
                fi.write(content_spiece_model)

        return (out_vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None, already_has_special_tokens: bool = False
    ) -> List[int]:
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        bos_token_id = [1] if self.add_bos_token else []
        eos_token_id = [1] if self.add_eos_token else []

        if token_ids_1 is None:
            return bos_token_id + ([0] * len(token_ids_0)) + eos_token_id
        return (
            bos_token_id
            + ([0] * len(token_ids_0))
            + eos_token_id
            + bos_token_id
            + ([0] * len(token_ids_1))
            + eos_token_id
        )

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |
        ```

        if token_ids_1 is None, only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of ids.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = [0] * len(bos_token_id + token_ids_0 + eos_token_id)

        if token_ids_1 is not None:
            output += [1] * len(bos_token_id + token_ids_1 + eos_token_id)

        return output
