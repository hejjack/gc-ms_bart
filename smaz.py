from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path
from train_bart import build_tokenizer
import sys

mf = sys.argv[1]

tokenizer_type = f"mf{mf}"
tokenizer_path = f"tokenizer/bbpe_tokenizer/bart_bbpe_tokenizer_1M_{tokenizer_type}.model"
tokenizer = build_tokenizer(tokenizer_path)


for dataset_type in ["train", "valid", "test"]:
    dataset_path = Path("data/datasets/NIST/NIST_split_filip")
    source_token = "<nist>"
    msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                    tokenizer,
                    source_token,
                    path_jsonl=dataset_path / tokenizer_type / f"{dataset_type}.jsonl",
                    keep_spectra=True
                    )