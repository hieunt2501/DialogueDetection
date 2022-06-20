import re

from vncorenlp import VnCoreNLP
from utils.text_processor.post_processor import RuleBasedPostprocessor
from config.config import Config

_config = Config()


class TextProcessor:
    def __init__(self):
        self.annotator = VnCoreNLP(address=_config.vncorenlp,
                                   port=_config.vncorenlp_port)
        self.rb_processor = RuleBasedPostprocessor()

    @staticmethod
    def clean_text(text):
        text = re.sub(r'[.-]', ' ', text)
        text = re.sub(r'\\x01', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def process(self, text):
        text = self.clean_text(text)
        sentences = self.annotator.tokenize(text.lower())
        annotated_text = [words for sentence in sentences for words in sentence]
        text = self.rb_processor.correct(" ".join(annotated_text))

        return text.strip()
