import os
import re
from dataclasses import dataclass


def convert_multiline_fasta_to_oneline(
        input_fasta: str, output_fasta: str = None) -> None:
    """
    Converts a multiline FASTA file to an oneline FASTA file.

    Parameters:
    - input_fasta: Path to the input multiline FASTA file
    with extension '.fasta' or '.fa'
    - output_fasta: Path to the output oneline FASTA file
    (default: input_name + '_oneline' + ext)

    Returns:
    - None

    Steps:
    1. Check if input file is a FASTA file
    2. Set output file name if not provided
    3. Check if output file is already exists
    4. Read input file line by line and convert to oneline format
    5. Write the oneline entries to output file
    """
    # check if `input_fasta` a fasta file:
    pattern = r'^(.+?)(\.fasta|\.fa)$'
    match = re.match(pattern, input_fasta)
    if match:
        input_name = match.group(1)
        ext = match.group(2)
    else:
        raise ValueError(f"File '{input_fasta}' is not a fasta file")
    # check if `output_fasta` is passed and set it if None:
    if output_fasta is None:
        output_fasta = input_name + '_oneline' + ext
    # check if 'output_fasta' exists:
    if os.path.isfile(output_fasta):
        raise FileExistsError(f"File '{output_fasta}' is already exists")

    # handle input_fasta line-by-line:
    with open(input_fasta, 'r') as f:
        l = f.readline().strip()
        while l != '':
            if l.startswith('>'):
                header = l
                seq = ''
                l = f.readline().strip()
                while not l.startswith('>') and l != '':
                    seq += l
                    l = f.readline().strip()
                # write entry to output_fasta:
                with open(output_fasta, 'a') as out_f:
                    out_f.write(header + '\n')
                    out_f.write(seq + '\n')


@dataclass
class FastaRecord:
    """Dataclass containing id, description and sequence of fasta record type"""
    id: str
    description: str
    seq: str

    def __repr__(self):
        return (f'FastaRecord\n'
                f'id = {self.id}\n'
                f'description = {self.description}\n'
                f'sequence = {self.seq}')


class OpenFasta:
    """
    Context manager for reading fasta files, similar to `open` CM
    Arguments:
        - fasta_file (str): path to fasta file
    Methods:
        - read_record
        - read_records
        """

    def __init__(self, fasta_file: str):
        self.file = fasta_file
        self.handler = None
        self.__next_rec_line = None

    def read_record(self) -> FastaRecord:
        """read single record from fasta file"""
        try:
            return next(self)
        except:
            return None

    def read_records(self) -> list[FastaRecord]:
        """read all records from fasta file"""
        records = []
        for record in self:
            records.append(record)
        return records

    def __iter__(self):
        return self

    def __next__(self) -> FastaRecord:
        record = []
        if self.__next_rec_line and self.__next_rec_line.startswith('>'):
            record.append(self.__next_rec_line.strip())
        # read file:
        self.__next_rec_line = self.handler.readline()
        while self.__next_rec_line:
            if self.__next_rec_line.startswith('>'):
                if record:
                    break
            record.append(self.__next_rec_line.strip())
            self.__next_rec_line = self.handler.readline()
        # parse and return record:
        if record:
            id = record[0].split(' ')[0][1:]
            description = ' '.join(record[0].split(' ')[1:])
            seq = ''.join(record[1:])
            return FastaRecord(id=id, description=description, seq=seq)
        else:
            raise StopIteration

    def __enter__(self):
        if os.path.isfile(self.file):
            self.handler = open(self.file)
            return self
        else:
            raise FileNotFoundError(f'Fasta file {self.file} does not exist')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handler:
            self.handler.close()
