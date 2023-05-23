from tokenizers import Tokenizer
from tokenizers import models, normalizers, pre_tokenizers, trainers, decoders, processors
from tqdm import trange
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerFast


class TogepiTokenizer(object):
    def __init__(self, name, vocab_size, special_toks=None, lowercase=True, punct_behavior='isolated',
                 padding_side='right', truncation_side='right'):
        super().__init__()

        self.special_toks = special_toks
        if self.special_toks is None:
            self.special_toks = {'unk': '<|unk|>', 'pad': '<|pad|>', 'endoftext': '<|endoftext|>'}
        for special_tok_name, special_tok in self.special_toks.items():
            self.__dict__.update({f'{special_tok_name}_tok': special_tok})

        self._name = name
        self._tokenizer = self._build_tokenizer(lowercase=lowercase, punct_behavior=punct_behavior)
        self._vocab_size = vocab_size
        self._padding_side = padding_side
        self._truncation_side = truncation_side

    def __len__(self):
        return self._tokenizer.get_vocab_size()

    def _build_tokenizer(self, lowercase=True, punct_behavior='isolated'):
        # https://huggingface.co/learn/nlp-course/chapter6/8?fw=pt
        tokenizer = Tokenizer(model=models.WordPiece(unk_token=self.unk_tok))
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.StripAccents()])
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(behavior=punct_behavior)])
        return tokenizer

    def train_tokenizer(self, training_corpus):
        self._trainer = trainers.WordPieceTrainer(vocab_size=self._vocab_size,
                                                  special_tokens=list(self.special_toks.values()))
        self._train_tokenizer(training_corpus=training_corpus)
        for special_tok_name, special_tok in self.special_toks.items():
            self.__dict__.update({f'{special_tok_name}_tok_idx': self._tokenizer.token_to_id(special_tok)})
        self._tokenizer.decoder = decoders.WordPiece(prefix="##")
        self._post_processor()

        self.vocab = self._tokenizer.get_vocab()
        self.pretrained_tokenizer = self._get_pretrained_tokenizer(name=self._name, padding_side=self._padding_side,
                                                                   truncation_side=self._truncation_side)

    def _train_tokenizer(self, training_corpus, batch_size=1000):
        def corpus_iter_fn():
            for idx in trange(0, len(training_corpus), batch_size):
                yield training_corpus[idx: idx + batch_size]['text']

        self._tokenizer.train_from_iterator(corpus_iter_fn(), self._trainer)

    def _post_processor(self):
        single_sequence = f'$A:1 {self.endoftext_tok}:1'
        pair_sequences = f'$A:1 {self.endoftext_tok}:1 $B:2 {self.endoftext_tok}:2'
        self._tokenizer.post_processor = processors.TemplateProcessing(single=single_sequence, pair=pair_sequences,
                                                                       special_tokens=[(self.endoftext_tok,
                                                                                        self.endoftext_tok_idx)])

    def _get_pretrained_tokenizer(self, name, padding_side='right', truncation_side='right'):
        return PreTrainedTokenizerFast(name_or_path=name, tokenizer_object=self._tokenizer, pad_token=self.pad_tok,
                                       unk_token=self.unk_tok, eos_token=self.endoftext_tok, padding_side=padding_side,
                                       truncation_side=truncation_side)

    @property
    def tokenizer(self):
        return self.pretrained_tokenizer

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
