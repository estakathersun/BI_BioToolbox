import os
from unittest.mock import patch

import pytest
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from main import NucleicAcidSequence, DNASequence, RNASequence
from main import filter_fastq, filter_gc, filter_quality, filter_length
from main import telegram_logger, send_telegram_message, token


class TestNucleicAcidAndDNASequence:
    @pytest.fixture
    def dna_seq(self):
        return DNASequence('ATGCN')

    def test_nucleic_acid_sequence_init_error(self):
        with pytest.raises(NotImplementedError):
            NucleicAcidSequence("ATGCN")

    def test_dna_sequence_creation(self, dna_seq):
        target = 'ATGCN'
        assert target == dna_seq.sequence

    def test_dna_sequence_check_alphabet(self, dna_seq):
        target = True
        assert target == dna_seq.check_alphabet()

    def test_dna_sequence_complement(self, dna_seq):
        complement = dna_seq.complement()
        target = 'TACGN'
        assert target == complement.sequence

    def test_dna_sequence_transcribe(self, dna_seq):
        rna_seq = dna_seq.transcribe()
        target = RNASequence('AUGCN')
        assert target == rna_seq


class TestFastqFilter:
    @pytest.fixture
    def temp_input_file(self):
        input_file = "test_input.fastq"
        with open(input_file, "w") as f:
            f.write("@seq1\nATCG\n+\nHHHH\n")
            f.write("@seq2\nGCTA\n+\nBBBB\n")
        yield input_file
        if os.path.isfile(input_file):
            os.remove(input_file)

    @pytest.fixture
    def temp_output_file(self):
        output_file = "test_input_filtered.fastq"
        yield output_file
        if os.path.isfile(output_file):
            os.remove(output_file)

    def test_filter_fastq_writes_output_file(self, temp_input_file, temp_output_file):
        filter_fastq(temp_input_file, gc_bounds=(0, 100),
                     length_bounds=(0, 2 ** 32), quality_threshold=0)

        assert os.path.exists(temp_output_file)
        with open(temp_output_file, "r") as f:
            records = list(SeqIO.parse(f, "fastq"))
            assert len(records) == 2

    def test_filter_length(self):
        record = SeqRecord(Seq("ATCG"), id="seq1", description="")
        assert filter_length(record, (2, 4)) == True

    def test_filter_gc(self):
        record = SeqRecord(Seq("ATCG"), id="seq1", description="")
        assert filter_gc(record, (50, 50)) == True

    def test_filter_quality(self):
        record = SeqRecord(Seq("ATCG"), id="seq1", description="")
        record.letter_annotations["phred_quality"] = [30, 40, 20, 10]
        assert filter_quality(record, 25) == True


class TestTelegramLogger:
    CHAT_ID: int = 502514437
    MESSAGE: str = 'Test message'
    TOKEN: str = token

    @pytest.fixture
    def temp_log_file(self):
        input_file = "temp_logfile.log"
        with open(input_file, "w") as f:
            f.write("This is a test file for `telegram_logger` decorator"
                    "\n\nCheer!💅💅💅")
        yield input_file
        if os.path.isfile(input_file):
            os.remove(input_file)

    def test_send_telegram_message(self):
        # create mock obj:
        with patch('requests.post') as mock_post:
            send_telegram_message(self.TOKEN, self.CHAT_ID, self.MESSAGE)
            # check request params:
            mock_post.assert_called_once_with(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": self.CHAT_ID,
                      "text": self.MESSAGE,
                      'parse_mode': 'MarkdownV2'}
            )

    def test_telegram_logger_handles_exception(self):
        @telegram_logger(self.CHAT_ID)
        def test_function():
            raise ValueError("Test error")

        with patch('requests.post') as mock_post:
            test_function()
            mock_post.assert_called_once()

    def test_send_telegram_message_with_file(self, temp_log_file):
        with patch('requests.post') as mock_post:
            send_telegram_message(self.TOKEN, self.CHAT_ID,
                                  self.MESSAGE, file=temp_log_file)

            mock_post.assert_called_once_with(
                f"https://api.telegram.org/bot{token}/sendDocument",
                data={"chat_id": self.CHAT_ID,
                      "caption": self.MESSAGE,
                      'parse_mode': 'MarkdownV2'},
                files={"document": temp_log_file}
            )
