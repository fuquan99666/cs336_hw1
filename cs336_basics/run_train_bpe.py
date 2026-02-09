import regex as re
import os


import time
import tracemalloc

import multiprocessing
from collections import Counter, defaultdict
from cs336_basics.pretokenization_example import find_chunk_boundaries
import heapq
from tqdm import tqdm

class bpe_max_heap:
    def __init__(self, pairs_freq, vocab):
        """
        初始化最大堆。
        :param pairs_freq: 字典，键是元组，值是频率。
        :param lt: 自定义的比较函数，用于决定堆的排序规则。
        """
        self.pairs = []
        self.vocab = vocab
        self._build_heap(pairs_freq)


    def _build_heap(self, pairs_freq):
        """
        根据给定的频率字典初始化堆。
        :param pairs_freq: 字典，键是元组，值是频率。
        """
        for pair, freq in pairs_freq.items():
            self.push(pair, freq)


    def push(self, pair, freq):
        """
        向堆中添加一个新的元素。
        :param pair: 要添加的元素对。
        :param freq: 元素的频率。
        """
        # 放在最后一个位置
        self.pairs.append((pair, freq))
        # up操作
        self._heapify_up(len(self.pairs) - 1)


    def pop_with_freq(self):
        """
        弹出堆顶元素并返回。
        """
        if len(self.pairs) > 1:
            self.pairs[0], self.pairs[-1] = self.pairs[-1], self.pairs[0]
            item = self.pairs.pop()
            self._heapify_down(0)
        elif self.pairs:
            item = self.pairs.pop()
        else:
            raise IndexError("pop from empty heap")
        return item[0], item[1]


    def top(self):
        """
        查看堆顶元素。
        """
        return self.pairs[0] if self.pairs else None
    

    def size(self):
        """
        返回堆的大小。
        """
        return len(self.pairs)


    def _heapify_up(self, index):
        """
        确保堆的上升操作符合自定义的比较规则。
        :param index: 插入元素的索引。
        """
        while index > 0:
            parent = (index - 1) // 2
            if self.lt(self.pairs[parent], self.pairs[index]):
                self.pairs[index], self.pairs[parent] = self.pairs[parent], self.pairs[index]
                index = parent
            else:
                break


    def _heapify_down(self, index):
        """
        确保堆的下降操作符合自定义的比较规则。
        :param index: 被下沉元素的索引。
        """
        size = len(self.pairs)
        while 2 * index + 1 < size:
            left = 2 * index + 1
            right = 2 * index + 2
            largest = index


            if left < size and self.lt(self.pairs[largest], self.pairs[left]):
                largest = left
            if right < size and self.lt(self.pairs[largest], self.pairs[right]):
                largest = right


            if largest != index:
                self.pairs[index], self.pairs[largest] = self.pairs[largest], self.pairs[index]
                index = largest
            else:
                break


    # 示例：定义一个自定义的比较函数 lt
    def lt(self,item1, item2):
        """lt 函数控制堆的排序规则"""
        # item is (pair, freq)
        # first compare freq then compare dictionary order of pair
        # and the pair is (int,int) we should use vocab 
        lt = False
        if item1[1] < item2[1]:
            lt = True
        elif item1[1] == item2[1]:
            if self.vocab[item1[0][0]] < self.vocab[item2[0][0]]:
                lt = True
            elif self.vocab[item1[0][0]] == self.vocab[item2[0][0]] and self.vocab[item1[0][1]] < self.vocab[item2[0][1]]:
                lt = True
        return lt
    

# Helper function to get pair frequencies
def get_pair_freqs(word_freqs, word_splits):
    pair_freqs = {}
    for word, freq in word_freqs.items():
        splits = word_splits[word]
        length = len(splits)
        for i in range(length - 1):
            pair = (splits[i], splits[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
    return pair_freqs

def _process_chunk(args):
    start, end, input_path, special_tokens = args
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    escaped_tokens = [re.escape(token) for token in special_tokens]
    if escaped_tokens:
        pattern = '|'.join(escaped_tokens)
    else:
        pattern = None

    word_freqs = Counter()
    
    with open(input_path, 'rb') as f:
        f.seek(start)
        data = f.read(end - start)
    
    # We use 'replace' to handle potential split multibyte characters at the very edges if they happened
    # though find_chunk_boundaries on special tokens (which are ASCII usually) should be safe,
    # but strictly speaking if not splitting on special token it might be an issue. 
    # Here we split on <|endoftext|> which is ASCII, so it is fine.
    text = data.decode('utf-8', errors='replace')
    
    if pattern:
        chunks = re.split(pattern, text)
    else:
        chunks = [text]
        
    for chunk in chunks:
        if not chunk: 
            continue
        tokens = re.finditer(PAT, chunk)
        for token in tokens:
            word_freqs[token.group(0)] += 1
            
    return word_freqs

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    # 踩过的坑： 并行是在预处理阶段，不是训练阶段
    # 先分割成不同chunk，然后每个独立预分词，最后合并进行训练。

    # 基本完成后，加个并行预分词？
    # import cProfile

    # profiler = cProfile.Profile()
    # profiler.enable()



    # Parallel Pre-tokenization
    num_processes = min(multiprocessing.cpu_count(), 32)
    
    # We use <|endoftext|> as the split point
    with open(input_path, 'rb') as f:
        # Assuming <|endoftext|> acts as document delimiter
        boundaries = find_chunk_boundaries(f, num_processes, b'<|endoftext|>')
        
    pool_args = []
    for i in range(len(boundaries) - 1):
        # We pass the file path instead of file object to be picklable
        pool_args.append((boundaries[i], boundaries[i+1], input_path, special_tokens))
        
    print(f"Starting parallel pre-tokenization with {num_processes} processes...")
    with multiprocessing.Pool(processes=num_processes) as pool:
        print(f"进程池已创建，主进程PID: {os.getpid()}")
        batch_results = pool.map(_process_chunk, pool_args)
        
    word_freqs = Counter()
    for res in batch_results:
        word_freqs.update(res)
        print("Update!")
    
    word_freqs = dict(word_freqs)
    
    # Convert words to lists of byte IDs (maintain order)
    word_splits = {}
    for word in word_freqs.keys():
        word_splits[word] = list(word.encode('utf-8'))
    
    # Initialize vocab with base bytes (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    

    
    # Train BPE
    num_merges_needed = vocab_size - len(vocab) - len(special_tokens)
    
    # Efficiently build pair_freqs and pair_to_words index
    pair_freqs = {}
    pair_to_words = defaultdict(set)
    
    for word, freq in word_freqs.items():
        splits = word_splits[word]
        length = len(splits)
        for i in range(length - 1):
            pair = (splits[i], splits[i + 1])
            pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
            pair_to_words[pair].add(word)
    

    # init the max heap
    max_heap = bpe_max_heap(pair_freqs,vocab)
    print("Starting BPE training...")
    
    pbar = tqdm(range(num_merges_needed))
    for m in pbar:
        # Update progress bar description
        if m % 100 == 0:
            pbar.set_description(f"Vocab size: {len(vocab)}")
        need_update_pairs = set()

        if not pair_freqs:
            break

        
        # first use the max function, maybe now we can use heapq later

        # Find most frequent pair
        # Break ties by preferring lexicographically greater pair (comparing the bytes, not token IDs)
        #best_pair = max(pair_freqs.items(), key=lambda x: (x[1], (vocab[x[0][0]], vocab[x[0][1]])))[0]
        #best_pair = max(pair_freqs.items(), key=lambda x: (x[1],vocab[x[0][0]],vocab[x[0][1]]))[0]

        # 由于堆中可能有重复的pair
        while True:
            best_pair , freq = max_heap.pop_with_freq()
            if pair_freqs.get(best_pair,0) == freq:
                break


        # Record the merge
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        # Add new token to vocab
        new_token_id = len(vocab)
        vocab[new_token_id] = b''.join(merges[-1])
        
        # Update word_splits by merging the best pair
        words_to_update = list(pair_to_words[best_pair])
        del pair_to_words[best_pair]
        
        for word in words_to_update:

            splits = word_splits[word]
            freq = word_freqs[word]
            length = len(splits)

            i = 0
            while i < length:
                if i < length - 1 and (splits[i], splits[i + 1]) == best_pair:
                    # 即刻更新freqs
                    if i - 1 >= 0:
                        left_pair = (splits[i-1],splits[i])
                        if left_pair in pair_freqs:
                            pair_freqs[left_pair] -= freq


                        new_left_pair = (splits[i-1],new_token_id)
                        pair_freqs[new_left_pair] = pair_freqs.get(new_left_pair,0) + freq
                        pair_to_words[new_left_pair].add(word)
                        
                        need_update_pairs.add(left_pair)
                        need_update_pairs.add(new_left_pair)

                    if i + 2 < length:
                        right_pair = (splits[i+1],splits[i+2])
                        if right_pair in pair_freqs:
                            pair_freqs[right_pair] -= freq


                        new_right_pair = (new_token_id,splits[i+2])
                        pair_freqs[new_right_pair] = pair_freqs.get(new_right_pair,0) + freq
                        pair_to_words[new_right_pair].add(word)
                        
                        need_update_pairs.add(right_pair)
                        need_update_pairs.add(new_right_pair)
                    splits = splits[:i] + [new_token_id] + splits[i+2:]
                    length -= 1
                i += 1

            word_splits[word] = splits
            
        pair_freqs.pop(best_pair,None)

        # update the heap
        for pair in need_update_pairs:
            freq = pair_freqs.get(pair,0)
            if freq <= 0:
                pair_freqs.pop(pair,None)
                pair_to_words.pop(pair,None)
                continue
            max_heap.push(pair,freq)
    
    # Add special tokens to vocab
    for token in special_tokens:
        token_bytes = token.encode('utf-8')
        vocab[len(vocab)] = token_bytes

    # profiler.disable()
    # profiler.print_stats(sort="cumtime")
    
    return vocab, merges

import json


def train():
    # 记录训练的时间和内存占用
    print("start train!")
    tracemalloc.start()
    start_time = time.time()
    try:
        # start train 
        vocab,merges = run_train_bpe(
            #input_path="/data/share/hw1-data/TinyStoriesV2-GPT4-train.txt",
            input_path = "/data/share/hw1-data/owt_train.txt",
            vocab_size=32000,
            special_tokens=["<|endoftext|>"],
        )
        
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"Training time: {end_time - start_time} seconds")
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        print("train finished!")

        # find the longest token in vocab
        longest_token = max(vocab.values(), key=len)
        print(f"longest token is {longest_token} with length {len(longest_token)}")

        # # 保存词汇表为 JSON 格式
        with open("./data/bpe_train_owt_vocab.json", "w", encoding="utf-8") as f:
            # 转换 vocab: {id: token_bytes} 为 {token_str: id}
            vocab_dict = {token.decode('utf-8', errors='ignore'): index for index, token in vocab.items()}
            json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

        # 保存 merges 为 JSON 格式
        with open("./data/bpe_train_owt_merges.json", "w", encoding="utf-8") as f:
            # 转换 merges: [(token1_bytes, token2_bytes), ...] 为 [[token1_str, token2_str], ...]
            merges_list = [
                [merge[0].decode('utf-8', errors='ignore'), merge[1].decode('utf-8', errors='ignore')]
                for merge in merges
            ]
            json.dump(merges_list, f, ensure_ascii=False, indent=2)
    except MemoryError:
        print("MemoryError: Not enough memory to complete the training.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred: {e}")

def compute_compression_ratio(words, tokens):
    # compute compression ratio
    total_bytes = len(words.encode('utf-8'))
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0
    return total_bytes / total_tokens
import random
import os

def experiment():
    """Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyS-
    tories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these
    sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
    """
    # load 10 documents from one file 
    #input_path = "/data/share/hw1-data/TinyStoriesV2-GPT4-train.txt"
    input_path = "/data/share/hw1-data/owt_train.txt"
    # 一次性读取并分割
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    all_documents = content.split("<|endoftext|>")
    # 过滤空文档并加回分隔符
    all_documents = [doc + "<|endoftext|>" for doc in all_documents if doc.strip()]
    
    # 随机采样10个文档
    if len(all_documents) >= 10:
        documents = random.sample(all_documents, 10)
    else:
        documents = all_documents  # 如果不够10个，就用全部
    
    print(f"随机采样了 {len(documents)} 个文档")
    # use the trained tokenizer to encode these documents
    from cs336_basics.bpe_tokenizer import BPETokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="./data/bpe_train_owt_vocab.json",
        merges_filepath="./data/bpe_train_owt_merges.json",
        special_tokens=["<|endoftext|>"],
    )
    all_tokens = tokenizer.encode("".join(documents))
    ratio = compute_compression_ratio("".join(documents), all_tokens)
    print(f"Output total tokens has {len(all_tokens)}: {all_tokens}")
    all_words = [tokenizer.decode([tid]) for tid in all_tokens]
    print(f"Output total words has {len(all_words)}: {all_words}")
    #print(f"Compression ratio (TinyStories tokenizer on TinyStories documents): {ratio} bytes/token")
    print(f"Compression ratio (TinyStories tokenizer on OpenWebText documents): {ratio} bytes/token")

def find_longest_token():

    input_path = "./data/bpe_train_owt_vocab.json"
    with open(input_path, "r", encoding="utf-8") as f:
        vocab_dict = json.load(f)
    vocab = {int(index): token.encode('utf-8') for token, index in vocab_dict.items()}

    # 考虑有多个最长的token
    # 全部找出来
    max_length = max(len(token) for token in vocab.values())
    longest_tokens = [token for token in vocab.values() if len(token) == max_length]
    print(f"Longest token length: {max_length}")
    print(f"Longest tokens: {[token.decode('utf-8', errors='ignore') for token in longest_tokens]}")

def magic():
    vocab = {i : bytes([i]) for i in range(256)}
    vocab[len(vocab)] = b'ab'
    vocab[len(vocab)] = b'bc'
    vocab[len(vocab)] = b'\xed\xa0\x80'  
    for index, token in vocab.items():
        print(index, token)

    vocab_1 = { token.decode('utf-8', errors='ignore'): index for index, token in vocab.items()}
    for token,index in vocab_1.items():
        print(index,token)



def test_throughoutput():
    # 测试tokneizer的的through output
    # bytes/second
    from cs336_basics.bpe_tokenizer import BPETokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath="./data/bpe_train_owt_vocab.json",
        merges_filepath="./data/bpe_train_owt_merges.json",
        special_tokens=["<|endoftext|>"],
    )

    input_path = "/data/share/hw1-data/owt_train.txt"
    # 一次性读取并分割
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    all_documents = content.split("<|endoftext|>")
    # 过滤空文档并加回分隔符
    all_documents = [doc + "<|endoftext|>" for doc in all_documents if doc.strip()]
    
    # 随机采样10个文档
    if len(all_documents) >= 1000000:
        documents = random.sample(all_documents, 1000000)
    else:
        documents = all_documents  # 如果不够100个，就用全部
    
    print(f"随机采样了 {len(documents)} 个文档")

    content = "".join(documents)
    start_time = time.time()
    num_bytes = len(content.encode('utf-8'))
    Mb = num_bytes / 10**6
    
    tokens = tokenizer.encode(content)
    end_time = time.time()
    duration = end_time - start_time
    throughput = Mb / duration
    print(f"Encoded {num_bytes} bytes into {len(tokens)} tokens in {duration} seconds. Throughput: {throughput} MB/second")

    

if __name__ == "__main__":
    train()
    #experiment()
    #find_longest_token()
    #magic()
    #test_throughoutput()