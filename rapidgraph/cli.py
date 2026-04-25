def main(argv=None):
    from .core import main as core_main

    return core_main(argv)


def parse_args(argv=None):
    from .core import parse_args as core_parse_args

    return core_parse_args(argv)


def read_input_text(path):
    from .core import read_input_text as core_read_input_text

    return core_read_input_text(path)


def read_input_texts(args):
    from .core import read_input_texts as core_read_input_texts

    return core_read_input_texts(args)


__all__ = ["main", "parse_args", "read_input_text", "read_input_texts"]
