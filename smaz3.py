from spectra_process_utils import msp_file_to_jsonl
from pathlib import Path
from train_bart import build_tokenizer

tokenizer_path = "tokenizer/bbpe_tokenizer/bart_bbpe_tokenizer_1M_mf3000.model"
tokenizer = build_tokenizer(tokenizer_path)


for dataset_type in ["train", "valid", "test"]:
    dataset_path = Path("data/datasets/NIST/NIST_split_filip")
    source_token = "<nist>"
    msp_file_to_jsonl(dataset_path / f"{dataset_type}.msp",
                    tokenizer,
                    source_token,
                    path_jsonl=dataset_path / "mf3000" / f"{dataset_type}.jsonl",
                    keep_spectra=True
                    )