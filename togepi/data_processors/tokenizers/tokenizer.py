from tokenizers import Tokenizer
from tokenizers import models, normalizers, pre_tokenizers, trainers, decoders, processors
from tqdm import trange
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast


class TogepiTokenizer(object):
    def __init__(self, num_merges, min_frequency=2, special_toks=None, lowercase=True, punct_behavior='contiguous',
                 padding_side='right', truncation_side='right', name='togepi-bbpe'):
        super().__init__()

        self.vocab = None
        self.special_toks = special_toks
        if self.special_toks is None:
            self.special_toks = {'pad': '<|pad|>', 'unk': '<|unk|>', 'eos': '<|endoftext|>'}
        for special_tok_name, special_tok in self.special_toks.items():
            self.__dict__.update({f'{special_tok_name}_tok': special_tok})

        self._build_tokenizer(lowercase=lowercase, punct_behavior=punct_behavior)

        self._name = name
        self._vocab_size = num_merges + len(pre_tokenizers.ByteLevel.alphabet()) + len(self.special_toks)
        self._min_frequency = min_frequency
        self._padding_side = padding_side
        self._truncation_side = truncation_side

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    def _build_tokenizer(self, lowercase=True, punct_behavior='contiguous'):
        # https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt
        self._tokenizer = Tokenizer(model=models.BPE(unk_token=self.unk_tok))
        normalizers_sequence = [normalizers.NFD(), normalizers.StripAccents()]
        if lowercase:
            normalizers_sequence = [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
        self._tokenizer.normalizer = normalizers.Sequence(normalizers_sequence)
        # note: using `pre_tokenizers.Digits(individual_digits=False)` makes whitespace before digits a different token
        self._tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
             pre_tokenizers.Punctuation(behavior=punct_behavior)])

    def train(self, training_corpus):
        # byte-level bpe: https://stackoverflow.com/a/55416944
        self._trainer = trainers.BpeTrainer(vocab_size=self._vocab_size, min_frequency=self._min_frequency,
                                            show_progress=True, special_tokens=list(self.special_toks.values()),
                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        self._train(training_corpus=training_corpus)
        for special_tok_name, special_tok in self.special_toks.items():
            self.__dict__.update({f'{special_tok_name}_tok_idx': self._tokenizer.token_to_id(special_tok)})
        self._tokenizer.decoder = decoders.ByteLevel()
        self._set_post_processor()

        self.vocab = self._tokenizer.get_vocab()
        self._set_pretrained_tokenizer(name=self._name, padding_side=self._padding_side,
                                       truncation_side=self._truncation_side)

    def _train(self, training_corpus, batch_size=1000):
        def corpus_iter_fn():
            for idx in trange(0, len(training_corpus), batch_size):
                yield training_corpus[idx: idx + batch_size]['text']

        self._tokenizer.train_from_iterator(corpus_iter_fn(), trainer=self._trainer, length=None)

    def _set_post_processor(self):
        byte_level_processor = processors.ByteLevel(trim_offsets=True)
        single_sequence = f'$A:1 {self.eos_tok}:1'
        pair_sequences = f'$A:1 {self.eos_tok}:1 $B:2 {self.eos_tok}:2'
        template_processor = processors.TemplateProcessing(single=single_sequence, pair=pair_sequences,
                                                           special_tokens=[(self.eos_tok, self.eos_tok_idx)])
        self._tokenizer.post_processor = processors.Sequence([byte_level_processor, template_processor])

    def _set_pretrained_tokenizer(self, name, padding_side='right', truncation_side='right'):
        self.pretrained_tokenizer = PreTrainedTokenizerFast(name_or_path=name, tokenizer_object=self._tokenizer,
                                                            eos_token=self.eos_tok, pad_token=self.pad_tok,
                                                            unk_tok=self.unk_tok, padding_side=padding_side,
                                                            truncation_side=truncation_side)

    @property
    def backend_tokenizer(self):
        return self._tokenizer

    def encode(self, text):
        return self._tokenizer.encode(text)

    def decode(self, tok_ids):
        return self._tokenizer.decode(tok_ids)

    def save(self, filepath):
        self.pretrained_tokenizer.save_pretrained(filepath)

    @staticmethod
    def load(filepath):
        return AutoTokenizer.from_pretrained(filepath)

    @staticmethod
    def tokenize(pretrained_tokenizer, text, max_length=None):
        if max_length is not None:
            tokenized_text = pretrained_tokenizer(text, padding='max_length', max_length=max_length, truncation=True,
                                                  return_tensors='np')
        else:
            tokenized_text = pretrained_tokenizer(text, padding=False, truncation=False, return_tensors='np')
        return tokenized_text
