import io
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from functools import wraps
from typing import Union, Optional, Tuple, Dict

import numpy as np
import requests
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import GC123
from dotenv import load_dotenv


# --------------------------FASTQ filter----------------------------------------
def filter_fastq(input_path: str,
                 gc_bounds: tuple = (0, 100),
                 length_bounds: tuple = (0, 2 ** 32),
                 quality_threshold: int = 0) -> None:
    """
       Reads a FASTQ file, filters sequences based on GC content, sequence
       length, and quality threshold, and writes the filtered sequences to
       a new file.

       Args:
           input_path (str): The path to the input FASTQ file.
           gc_bounds (Tuple[int, int]): The range of GC content values to filter by (default: (0, 100)).
           length_bounds (Tuple[int, int]): The range of sequence lengths to filter by (default: (0, 2**32)).
           quality_threshold (int): The minimum quality threshold to filter by (default: 0).

       Returns:
           None

       Example:
           filter_fastq("input.fastq", gc_bounds=(30, 70), length_bounds=(20, 100), quality_threshold=20)
       """
    path, filename = os.path.split(input_path)
    name, ext = os.path.splitext(filename)
    output_path = os.path.join(path, f"{name}_filtered{ext}")
    input_seq_iterator = SeqIO.parse(input_path, 'fastq')
    filtered_seq_iterator = (record for record in input_seq_iterator
                             if filter_length(record, length_bounds)
                             and filter_gc(record, gc_bounds)
                             and filter_quality(record, quality_threshold))
    SeqIO.write(filtered_seq_iterator, output_path, "fastq")


def filter_length(record: SeqRecord, length_bounds: tuple) -> bool:
    return length_bounds[0] <= len(record.seq) <= length_bounds[1]


def filter_gc(record: SeqRecord, gc_bounds: tuple) -> bool:
    return gc_bounds[0] <= GC123(record.seq)[0] <= gc_bounds[1]


def filter_quality(record: SeqRecord, quality_threshold: int) -> bool:
    avg_quality = np.mean(record.letter_annotations["phred_quality"])
    return avg_quality >= quality_threshold


# -----------------------Biological sequence classes----------------------------
class BiologicalSequence(ABC, str):
    @abstractmethod
    def check_alphabet(self) -> bool:
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __str__(self):
        pass


class NucleicAcidSequence(BiologicalSequence):
    """
    Class representing nucleic acid sequences.

    Attributes:
        ALPHABET (Tuple[str, ...])
        COMPLEMENT_DICT (Dict[str, str])

    Methods:
        - check_alphabet: Checks if the provided nucleic acid sequence contains
        only acceptable letters
        - complement: Complements DNA or RNA sequence
        - get_GC: Calculates GC content for nucleic acid sequence

    """
    ALPHABET: Tuple[str, ...]
    COMPLEMENT_DICT: Dict[str, str]

    def __init__(self, sequence):
        raise NotImplementedError('An instance of this class cannot be created')

    def check_alphabet(self) -> bool:
        """Checks if the provided nucleic acid sequence contains
        only acceptable letters"""
        sequence_chars = set(self.sequence)
        return sequence_chars.issubset(type(self).ALPHABET)

    def complement(self) -> 'NucleicAcidSequence':
        """Complements DNA or RNA sequence."""
        complement_seq = ''.join(type(self).COMPLEMENT_DICT.get(base)
                                 for base in self.sequence)
        return type(self)(complement_seq)

    def get_gc(self) -> float:
        """Calculates GC content for nucleic acid sequence"""
        return GC123(self.sequence)[0]

    def __repr__(self):
        """Returns representation of the object"""
        return f"{self.__class__.__name__}: {self.sequence}"

    def __str__(self):
        return self.sequence


class RNASequence(NucleicAcidSequence):
    """
    Class representing RNA sequences

    Attributes:
        - ALPHABET: contains valid for RNA characters
        - COMPLEMENT_DICT: contains rules for complement

    Methods: refer to parent class 'NucleicAcidSequence'
        """
    ALPHABET = ('A', 'U', 'G', 'C', 'N')
    COMPLEMENT_DICT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                       'a': 'u', 'u': 'a', 'g': 'c', 'c': 'g'}

    def __init__(self, sequence):
        self.sequence = sequence


class DNASequence(NucleicAcidSequence):
    """
    Class representing DNA sequences

    Attributes:
        - ALPHABET: contains valid for DNA characters
        - COMPLEMENT_DICT: contains rules for complement

    Methods: refer to parent class 'NucleicAcidSequence'
        """
    ALPHABET = ('A', 'T', 'G', 'C', 'N')
    COMPLEMENT_DICT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N',
                       'a': 't', 't': 'a', 'g': 'c', 'c': 'g', 'n': 'n'}

    def __init__(self, sequence):
        self.sequence = sequence

    def transcribe(self) -> RNASequence:
        """Transcribes DNA sequence into RNA sequence. Returns RNASequence object"""
        return RNASequence(self.sequence.replace('T', 'U').replace('t', 'u'))


class AminoAcidSequence(BiologicalSequence):
    """
    Class representing amino acid sequences

    Attributes:
        - ONE_LETTER_ALPHABET: contains valid characters
            for amino acid sequences in the one-letter entry
        - THREE_LETTER_ALPHABET: contains valid characters
            for amino acid sequences in the three-letter entry

    Methods:
        - check_alphabet: checks if the provided amino acid sequence
            contains only acceptable letters in one-letter or three-letter code
        - get_molecular_weight: calculate molecular weight for one-letter amino acid sequence
            """

    ONE_LETTER_ALPHABET = ('A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y')
    THREE_LETTER_ALPHABET = ('Ala', 'Cys', 'Asp', 'Glu', 'Phe', 'Gly', 'His',
                             'Ile', 'Lys', 'Leu', 'Met', 'Pro', 'Gln', 'Arg',
                             'Ser', 'Thr', 'Val', 'Trp', 'Tyr')

    def __init__(self, sequence: str):
        self.sequence = sequence

    def check_alphabet(self) -> bool:
        """Checks if the provided amino acid sequence contains only
        acceptable letters in one-letter or three-letter code"""
        sequence_chars = set(self.sequence)
        return (sequence_chars.issubset(self.ONE_LETTER_ALPHABET)
                or sequence_chars.issubset(self.THREE_LETTER_ALPHABET))

    def get_molecular_weight(self) -> float:
        """Calculate molecular weight for one-letter amino acid sequence"""
        WEIGHT_DICT = {
            'G': 57.051, 'A': 71.078, 'S': 87.077, 'P': 97.115, 'V': 99.131,
            'T': 101.104, 'C': 103.143, 'I': 113.158, 'L': 113.158, 'N': 114.103,
            'D': 115.087, 'Q': 128.129, 'K': 128.172, 'E': 129.114, 'M': 131.196,
            'H': 137.139, 'F': 147.174, 'R': 156.186, 'Y': 163.173, 'W': 186.210
        }
        terminal_h_oh_weight = 18.02
        weight = (terminal_h_oh_weight +
                  sum(WEIGHT_DICT[aa] for aa in self.sequence if aa in WEIGHT_DICT))
        return weight

    def __repr__(self):
        """Returns representation of the object"""
        return f"{self.__class__.__name__}: {self.sequence}"

    def __str__(self):
        return self.sequence


# -------------------Telegram logger-----------------------
load_dotenv('./tg_api.env')
token = os.getenv('TG_API_TOKEN')


def telegram_logger(chat_id: int) -> callable:
    """
    Decorator function that logs the execution of a function and sends
    a message with the runtime info and log file to your Telegram bot.
    Message text contains the function name, the elapsed time (in case of success running)
    and the error info (in case of raised exception)

    !!! A global system variable `TG_API_TOKEN` containing token for your
    Telegram bot must be specified in `.env` file

    Args:
        chat_id (int): The chat ID where the Telegram message will be sent.

    Returns:
        Callable: A decorator function that can be used to wrap other functions.

    Example:
        chat_id = 123456789

        @telegram_logger(chat_id)
        def my_function():
            # Code here
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            stdout = io.StringIO()
            stderr = io.StringIO()
            sys.stdout = stdout
            sys.stderr = stderr
            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                elapsed_time = str(timedelta(seconds=elapsed_time))
                exc_type = None
                return result
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print(f'{exc_type.__name__}: {exc_value}', file=sys.__stderr__)

            finally:
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

                if exc_type:
                    text = (f"\U0001F4A9 Function `{func.__name__}` raised an exception:"
                            f"\n\n`{exc_type.__name__}: {exc_value}`")
                else:
                    text = (f"\U0001F485 Function `{func.__name__}` executed "
                            f"successfully in `{elapsed_time}`")

                stdout.seek(0)
                stderr.seek(0)
                output_text = stdout.read() + stderr.read()
                if output_text:
                    output_file = io.BytesIO(output_text.encode())
                    output_file.name = f"{func.__name__}.log"
                    send_telegram_message(token, chat_id, text, file=output_file)
                else:
                    send_telegram_message(token, chat_id, text)

        return wrapper

    return decorator


def send_telegram_message(token: str,
                          chat_id: int,
                          text: str,
                          file: Optional[Union[str, bytes]] = None) -> None:
    """
        Sends a message to a Telegram chat using the Telegram Bot API.

        Args:
            token (str): The Telegram bot token.
            chat_id (int): The chat ID where the message will be sent.
            text (str): The text of the message.
            file (file-like object, optional): The file to send
            along with the message. Defaults to None.
        """
    if file:
        url = f"https://api.telegram.org/bot{token}/sendDocument"
        data = {"chat_id": chat_id, "caption": text, 'parse_mode': 'MarkdownV2'}
        files = {"document": file}
        resp = requests.post(url, data=data, files=files)
    else:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = {"chat_id": chat_id, "text": text, 'parse_mode': 'MarkdownV2'}
        resp = requests.post(url, data=data)
    if not resp.ok:
        print(f'Failed to make request:\n {resp.text}')
