import sentencepiece as spm
import dill


class SentencepieceTokenizer:
    def __init__(
        self,
        input_file,
        prefix,
        character_coverage=0.995,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
    ):
        templates = "--input={} \
        --pad_id={} \
        --bos_id={} \
        --eos_id={} \
        --unk_id={} \
        --model_prefix={} \
        --character_coverage={} \
        --user_defined_symbols=[SEP],[CLS],[MASK] \
        --model_type=bpe"

        self.prefix = prefix
        self.cmd = templates.format(
            input_file, pad_id, bos_id, eos_id, unk_id, prefix, character_coverage
        )
        self.processor = None

    def train(self):
        spm.SentencePieceTrainer.Train(self.cmd)

    def tokenize(self, text):
        if self.processor is None:
            sp = spm.SentencePieceProcessor()
            sp.Load(self.prefix + ".model")
            sp.SetEncodeExtraOptions("bos:eos")
            self.processor = sp

        return self.processor.EncodeAsIds(text)
