import json
import regex as re
from typing import Iterable, Iterator
from cs336_basics.pretokenization_example import find_chunk_boundaries
import os
import multiprocessing
from io import BytesIO
from tqdm import tqdm

def _encode_chunk_wrapper(args):
    return args[0].encode_big_string(*args[1:])


class BPETokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        # create a tokenizer which can encode sentences and also can decode token ids back to sentences
        self.vocab = vocab  # (id: bytes)
        self.vocab_reverse = {v: k for k,v in vocab.items()} # (bytes: id)
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        # 提前计算merge pair的id
        merged_ids = {}
        pairs_ids = {}
        for i,merger in enumerate(merges):
            pairs = (self.vocab_reverse[merger[0]],self.vocab_reverse[merger[1]])
            pairs_ids[pairs] = i
            merged_bytes = merger[0] + merger[1]
            merged_ids[i] = self.vocab_reverse[merged_bytes]
        self.pairs_ids = pairs_ids
        self.merged_ids = merged_ids
        

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # load vocab and merges from files and return a BPETokenizer, this is a class method
        # read vocab, notice that vocab file is josn line,actually a dict
        vocab = {}
        file_ext = os.path.splitext(vocab_filepath)[1].lower()
        with open(vocab_filepath, 'r',encoding='utf-8') as f:
            if file_ext == '.json':
                j = json.load(f)   # notice format is [token_bytes: token_id]
                for token_bytes, token_id in j.items():
                    vocab[token_id] = token_bytes.encode('utf-8')
            elif file_ext == '.txt':
                for line in f:
                    token_id, token = line.strip().split('\t')
                    vocab[int(token_id)] = token.encode('utf-8')
        # read merges, each line is a merge operation
        merges = []
        with open(merges_filepath, 'r',encoding='utf-8') as f:
            j = json.load(f)
            for merge in j:
                first_byte = merge[0].encode('utf-8')
                second_byte = merge[1].encode('utf-8')
                merges.append((first_byte, second_byte))
        return cls(vocab, merges, special_tokens)
            

    def encode_big_string(self, start: int, end: int, file_like) -> list[int]:
        """
        处理大字符串并将其划分为块，然后并行地编码每个块。
        `start` 和 `end` 是文本块的起始和结束位置，`file_like` 是文件句柄。
        """
        file_like.seek(start)
        chunk_bytes = file_like.read(end - start)
        chunk_text = chunk_bytes.decode('utf-8', errors='ignore')

        return self.encode_chunk(chunk_text)

    def encode_chunk(self, text: str) -> list[int]:
        """
        对文本块进行编码，保持原有编码逻辑。
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        token_ids = []
        chunks = [text]

        # 如果有特殊token，进行特殊token分割
        if hasattr(self, 'special_tokens') and self.special_tokens:
            escaped_tokens = [re.escape(token) for token in self.special_tokens]
            escaped_tokens.sort(key=len, reverse=True)
            pattern = '(' + '|'.join(escaped_tokens) + ')'
            chunks = re.split(pattern, text)
        
        for chunk in chunks:
            if not chunk:
                continue

            # 对于特殊token直接编码
            if hasattr(self, 'special_tokens') and chunk in self.special_tokens:
                chunk_bytes = chunk.encode('utf-8')
                if chunk_bytes in self.vocab_reverse:
                    token_id = self.vocab_reverse[chunk_bytes]
                    token_ids.append(token_id)
                continue
            
            # 正常的文本处理
            tokens = re.finditer(PAT, chunk)
            words = [token.group(0) for token in tokens]

            for word in words:
                word_bytes = word.encode('utf-8')
                token_seq = []

                for byte in word_bytes:
                    byte_as_bytes = bytes([byte])
                    if byte_as_bytes in self.vocab_reverse:
                        token_seq.append(self.vocab_reverse[byte_as_bytes])

                # BPE合并过程
                while len(token_seq) > 1:
                    first_idx = -1
                    j = 0
                    while j < len(token_seq) - 1:
                        pair = (token_seq[j], token_seq[j+1])
                        if pair in self.pairs_ids:
                            idx = self.pairs_ids[pair]
                            if idx < first_idx or first_idx == -1:
                                first_idx = idx
                        j += 1

                    if first_idx == -1:
                        break
                    else:
                        j = 0
                        while j < len(token_seq) - 1:
                            pair = (token_seq[j], token_seq[j+1])
                            if pair in self.pairs_ids and self.pairs_ids[pair] == first_idx:
                                token_seq[j] = self.merged_ids[first_idx]
                                token_seq.pop(j+1)
                            j += 1
                token_ids.extend(token_seq)

        return token_ids


    def encode(self, text: str) -> list[int]:
        """
        处理并行编码大文本。
        """
        if len(text) > 100000:
            # 启用并行化
            num_processes = min(multiprocessing.cpu_count(), 32)
            text_bytes = text.encode('utf-8')
            file_like = BytesIO(text_bytes)

            # 寻找切分边界
            boundaries = find_chunk_boundaries(file_like, num_processes, b'<|endoftext|>')

            # 使用进程池并行处理
            pool_args = []
            for i in range(len(boundaries) - 1):
                pool_args.append((boundaries[i], boundaries[i+1], file_like))

            # 开启进程池并行处理
            with multiprocessing.Pool(processes=num_processes) as pool:
                print(f"进程池已创建，主进程PID: {os.getpid()},启用 {num_processes} 个子进程进行编码")
                if len(pool_args) > 0:
                     # Add self to args for the wrapper
                    tasks = [(self, *args) for args in pool_args]
                    batch_results = list(tqdm(pool.imap(_encode_chunk_wrapper, tasks), total=len(tasks), desc="Encoding chunks"))
                else:
                    batch_results = []

            # 合并每个子进程的结果
            token_ids = []
            for result in batch_results:
                token_ids.extend(result)

        else:
            # 如果文本较小，直接调用常规编码方法
            token_ids = self.encode_chunk(text)

        return token_ids

        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # encode an iterable of strings into an iterator of token ids
        # we do this at last 
        for string in iterable:
            token_ids = self.encode(string)
            for token_id in token_ids:
                yield token_id
    def decode(self, ids: list[int]) -> str:
        # decode a list of token ids back into a string
        output = b''
        for token_id in ids:
            token_bytes = self.vocab.get(token_id,b'')
            output += token_bytes
        return output.decode('utf-8',errors='replace')
        
def tokenize():
    #input_path = "/data/share/hw1-data/TinyStoriesV2-GPT4-train.txt"
    input_path = "/data/share/hw1-data/owt_train.txt"
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="./data/bpe_train_owt_vocab.json",
        merges_filepath="./data/bpe_train_owt_merges.json",
        special_tokens=["<|endoftext|>"],
    )
    # 将整个文件都转化为token ids，方便后面的训练
    # 并行处理

    with open(input_path, "r",encoding="utf-8") as f:
        # 将最终的id保存为numpy的uint16数组
        text = f.read()
        token_ids = tokenizer.encode(text)
    import numpy as np
    token_ids_array = np.array(token_ids,dtype=np.uint16)

    tokenlist = token_ids_array[:100]
    wordlist = tokenizer.decode(tokenlist)
    print(f"First 100 token ids decode to: {wordlist}")
    np.save("./data/owt_train_bpe_token_ids.npy",token_ids_array)

if __name__ == "__main__":
    tokenize()