from __future__ import annotations
from typing import List, Optional
from matchms import Spectrum
from matchms.importing import load_from_msp
from rdkit import Chem
import numpy as np
import pandas as pd
from tqdm import tqdm
from icecream import ic
from transformers import PreTrainedTokenizerFast
from bart_spektro.selfies_tokenizer import SelfiesTokenizer
from pathlib import Path
import selfies as sf


def mol_repr_to_labels(mol_repr, tokenizer, source_id: int) -> List[int]:
    """Converts molecular representation (SMILES/SELFIES) to labels for the model"""
    eos_token = tokenizer.eos_token_id
    encoded_mol_repr = tokenizer.encode(mol_repr)
    labels = [source_id] + encoded_mol_repr + [eos_token]
    return labels


def preprocess_spectrum(s: Spectrum, 
                        tokenizer,
                        source_token: str | None = None,
                        max_num_peaks: int = 300,
                        max_mol_repr_len: int = 100,
                        max_mz: int = 500,     # NIST min_mz is 1.0, that's why we set it to 500
                        log_base: float = 1.7,
                        log_shift: int = 9,
                        max_cumsum: Optional[float] = None,
                        mol_representation: str = "smiles"):
    """
    Preprocess one matchms.Spectrum according to BART_spektro preprocessing pipeline

    Parameters
    ----------

    s : matchms.Spectrum
        spectrum that has a corresponding SMILES saved in its metadata
    tokenizer : tokenizers.Tokenizer
        tokenizer used for tokenizing the mz values to fit input_ids
    max_num_peaks : int
        maximal num of peaks - specified by us not wanting too long sequences
    max_mol_repr_len : int
        max len of SMILES/SELFIES string representation
    max_mz : int
        the highest allowed value of mz (input element) - specified by the tokenizer's vocabulary
    log_base : float
        base of the logarithm used for log binning intensities
    log_shift : int
        shift of the logarithm used for log binning intensities
    max_cumsum : float
        when provided, preprocessing includes cumsum filtering of spectra
        that means leaving only the highest peak with sum of intensities
        just over max_cumsum (if sums to 1, basically percentage of 'mass')
    mol_representation : str
        molecule representation to be used (either "smiles" or "selfies", default "smiles")

    Returns
    -------
    mz : List[int]
        "tokenized" input to a BART spektro model - it's actually the mz values of the spectrum + 
    intensities : List[int]
        logged and binned intensities, prepared as position_id for BART spektro model
    canon_mol_reprs : str
        canonical SMILES/SELFIES representation of the spectra
    error_dict : Dict[str : bool]
        dict of flags detecting if spectrum has any of the five specified errors
    """
    goes_out = 0 # flag for discarding the datapoint
    error_dict = {"long_mol_repr": 0,
                  "corrupted": 0,
                  "high_mz": 0,
                  "too_many_peaks": 0,
                  "no_mol_repr": 0}

    # canonicalization + possible selfies transformation
    canon_mol_repr = canonicalize_smiles(s.metadata["smiles"])

    # filter corrupted
    if canon_mol_repr is None:
        goes_out = 1
        error_dict["corrupted"] = True
    else:
        canon_mol_repr = canon_mol_repr.strip() # often is a blank space at the beginning
        # no simles filtering
        if canon_mol_repr == "":
            goes_out = 1
            error_dict["no_mol_repr"] = True
        # long simles filtering
        elif len(canon_mol_repr) > max_mol_repr_len:
            goes_out = 1
            error_dict["long_mol_repr"] = True
        # destereothing ??? for NIST this happens during dataset splitting

    if mol_representation == "selfies" and canon_mol_repr is not None:
        canon_mol_repr = sf.encoder(canon_mol_repr)        # TODO?? try block?

    # filter high MZ
    if max(s.peaks.mz) > max_mz:
        goes_out = 1
        error_dict["high_mz"] = True

    # filter little peaks so it doesn't get kicked out    
    if max_cumsum:
        mz, intensities = cumsum_filtering(s.peaks.mz, s.peaks.intensities, max_cumsum)
    else:
        mz, intensities = s.peaks.mz, s.peaks.intensities

    # filter long spectra
    if len(mz) > max_num_peaks:
        goes_out = 1
        error_dict["too_many_peaks"] = True

    if goes_out:
        return ([], [], [], [], error_dict)

    # creating MZs inputs
    mz = [round(x) for x in mz]

    # scaling intensities
    intensities = intensities/max(intensities)    

    # log bining the intensities
    log_base = np.log(log_base)
    x = (np.log(intensities)/log_base).astype(int) + log_shift
    x = x * (x > 0)
    intensities = x.astype("int32")

    # creating label
    source_id = tokenizer.encode(source_token)[0]
    label = mol_repr_to_labels(canon_mol_repr, tokenizer, source_id)

    return (mz, intensities, canon_mol_repr, label, error_dict)


def preprocess_spectra(spectra: List[Spectrum],
                       tokenizer, 
                       keep_spectra: bool = False,
                       preprocess_args: dict = {}) -> pd.DataFrame:
    """
    Preprocess a list of matchms.Spectrum according to BART_spektro preprocessing pipeline
    Catch errors, sort them into 5 categories and print a report

    Parameters
    ----------
    spectra : List[matchms.Spectrum] or Generator[matchms.Spectrum]
        list of spectra to be preprocessed
    tokenizer : tokenizers.Tokenizer
        tokenizer used for tokenizing the mz values in preprocess_spectrum fction

    Returns
    -------
    df_out : pd.DataFrame
        a dataframe we are able to feed into SpectroDataset and then to SpectroBart
    """
    
    mzs = []
    intensities = []
    input_idss = []
    position_idss = []
    mol_reprs = []     # either smiles or selfies
    labels = []

    # STATS
    no_mol_reprs = 0
    long_mol_reprs = 0
    corrupted = 0
    high_mz = 0
    too_many_peaks = 0

    num_spectra = 0
    for d in tqdm(spectra): 
        (input_ids, position_ids, cs, l, ed) = preprocess_spectrum(d, tokenizer, **preprocess_args)
        if not input_ids:
            long_mol_reprs += ed["long_mol_repr"]
            corrupted += ed["corrupted"]
            high_mz += ed["high_mz"]
            too_many_peaks += ed["too_many_peaks"]
            no_mol_reprs += ed["no_mol_repr"]
        else:
            if keep_spectra:
                mzs.append(d.peaks.mz)
                intensities.append(d.peaks.intensities)
            input_idss.append(input_ids)
            position_idss.append(position_ids)
            mol_reprs.append(cs)
            labels.append(l)
        num_spectra += 1  # just a counter

    df_out = pd.DataFrame({"input_ids": input_idss, 
                            "position_ids": position_idss,
                            "mol_repr": mol_reprs,
                            "labels": labels})
    if keep_spectra:
        df_out["mz"] = mzs
        df_out["intensity"] = intensities
        
    # print STATS
    print(f"{no_mol_reprs} no mol_repr")
    print(f"{long_mol_reprs} mol_reprs too long")
    print(f"{corrupted} spectra corrupted")
    print(f"{high_mz} spectra w/ too high mz")
    print(f"{too_many_peaks} spectra w/ too many peaks")
    print(f"totally {long_mol_reprs + corrupted + high_mz + too_many_peaks} issues")
    print(f"discarded {num_spectra - len(df_out)}/{num_spectra} spectra ")
    return df_out

def extract_spectra(spectra: List[Spectrum]) -> pd.DataFrame:
    """
    Extracts mz and intensity from spectra and returns a dataframe

    Parameters
    ----------
    spectra : List[matchms.Spectrum] or Generator[matchms.Spectrum]
        list of spectra to be preprocessed

    Returns
    -------
    df_out : pd.DataFrame
        df with canonical smiles and mz, intensity columns
        a dataframe we are able to feed into datapipeline that does the preprocessing for us
    """
    mzs = []
    intensities = []
    canon_smiless = []
    for d in tqdm(spectra):
        canon_smiles = canonicalize_smiles(d.metadata["smiles"])
        if canon_smiles:
            mzs.append(d.peaks.mz)
            intensities.append(d.peaks.intensities)
            canon_smiless.append(canon_smiles)
    df_out = pd.DataFrame({"mz": mzs, "intensity": intensities, "smiles": canon_smiless})
    return df_out

def msp_file_to_jsonl(path_msp: Path,
                      tokenizer: PreTrainedTokenizerFast | SelfiesTokenizer | None,
                      path_jsonl: Path | None = None,
                      keep_spectra: bool = False,
                      do_preprocess: bool = True,
                      preprocess_args: dict = {}):
    """load msp file, preprocess, prepare BART compatible dataframe and save to jsonl file
    
    Parameters
    ----------
    path_msp : Path
        path to the msp file
    tokenizer : PreTrainedTokenizerFast | SelfiesTokenizer
        tokenizer used for tokenizing the smiles to fit decoder_input_ids
    path_jsonl : Path
        path of the output jsonl file, if None, the jsonl file is saved
        with the same name as the msp file, but with .jsonl extension
    keep_spectra : bool
        whether to keep the spectra (original mz, and intensities) in the output jsonl file along with the preprocessed data
    do_preprocess : bool
        whether to preprocess the spectra and prepare BART compatible jsonl file or just extract the spectra from msp to jsonl
    preprocess_args : dict
        dictionary of arguments for the preprocess_spectrum function (max_num_peaks, max_mol_repr_len, max_mz, log_base, log_shift, max_cumsum, mol_representation, source_token)
    """
    data_msp = list(load_from_msp(str(path_msp), metadata_harmonization=False))
    if do_preprocess:
        df = preprocess_spectra(data_msp, tokenizer, keep_spectra=keep_spectra, preprocess_args=preprocess_args)
    else:
        df = extract_spectra(data_msp)

    if not path_jsonl:
        path_jsonl = path_msp.with_suffix(".jsonl")
    else:
        path_jsonl.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(path_jsonl, orient="records", lines=True)


def canonicalize_smiles(smi: str) -> str | None:
    """canonicalize SMILES using rdkit"""
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi), True)
    except Exception as msg:
        print("Couldn't be canonicalized due to Exception:", msg) 
    return None

def remove_stereochemistry_and_canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:                           # TODO oddelej if else 
        Chem.RemoveStereochemistry(mol)
    else: 
        return None 
    new_smiles = Chem.MolToSmiles(mol)
    return new_smiles


def msp_file_to_smi(path_msp: Path):
    """load msp file, canonicalize SMILES and save them to .smi file
    
    Parameters
    ----------
    path_msp : Path
        path to the msp file
    """
    data_msp = list(load_from_msp(str(path_msp), metadata_harmonization=False))
    output_path = path_msp.parent / (str(path_msp.stem) + ".smi")
    print("saving to:", output_path)
    out = output_path.open("w+")
    for i, spectrum in enumerate(data_msp):
        canonical_smiles = canonicalize_smiles(spectrum.metadata["smiles"])
        if canonical_smiles:
            out.write(canonical_smiles + "\n")
    out.close()


def process_neims_spec(spec, metadata):
    """
    take a sdf spectrum as it comes out of NEIMS and create a matchms.Spectrum
    
    Parameters
    ----------
    spec : str
        string representation of spectra as it comes out of the NEIMS model
    metadata : dict
        a dict that will be attached as metadata to the spectrum
    
    Return
    ------
    matchms.Spectrum
        matchms representation of the input NEIMS spectra that contains the metadata 
        specified in the corresponding parameter
    """
    spec = spec.split("\n")
    i = []
    mz = []
    for t in spec:
        j, d = t.split()
        mz.append(j)
        i.append(d)
    return Spectrum(mz=np.array(mz).astype(np.single),
                    intensities=np.array(i).astype(np.single),
                    metadata=metadata,
                    metadata_harmonization=False)


def oneD_spectra_to_mz_int(df : pd.DataFrame) -> pd.DataFrame:
    """
    Function that takes a DF and splits the one-array-representation of spectra into mz and intensity parts
    
    Parameters
    ----------
    df : pd.DataFrame
         dataframe containing 'PREDICTED SPECTRUM' column with sdf spectra representation
         -> is used after loading enriched sdf file with PandasTools.LoadSDF
    
    Returns
    -------
    df2 : pd.DataFrame
          dataframe containing columns 'mz' and 'intensity' that contain decomposed spectra representation, two arrays of the same length
    """
    df2 = df.copy()
    all_i = []
    all_mz = []
    for row in tqdm(range(len(df2))):
        spec = df2["PREDICTED SPECTRUM"][row].split("\n")
        mz = []
        i = []
        spec_max = 0
        for t in spec:
            j,d = t.split()
            j,d = int(j), float(d)
            if spec_max < d:
                spec_max = d
            mz.append(j)
            i.append(d)
        all_mz.append(mz)
        all_i.append(np.around(np.array(i)/spec_max, 2))
    new_df = pd.DataFrame.from_dict({"mz": all_mz, "intensity": all_i})
    df2 = pd.concat([df2, new_df], axis=1)
    df2 = df2.drop(["PREDICTED SPECTRUM"], axis=1)
    return df2


def cumsum_filtering(mz: np.ndarray, 
                     i: np.ndarray, 
                     max_cumsum: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Leaves in the spectrum only the biggest peaks with sum of intensities
    just over max_cumsum.

    Parameters
    ----------
    mz : np.ndarray
        array of mz values
    i : np.ndarray
        array of intensities
    max_cumsum : float
        maximum sum (percentage of 'mass') of intensities of peaks in the spectrum
    
    """
    # normalize by array's sum (just for filtering)
    i_norm = i/np.sum(i)

    # sort arrays
    index = (-i_norm).argsort() # descending sort
    mz_sorted = mz[index]
    i_sorted = i[index]
    i_norm_sorted = i_norm[index]
    
    # cut off the smallest peaks (according to cumsum)
    cs = np.cumsum(i_norm_sorted)
    cut = np.searchsorted(cs, max_cumsum) + 1  # this is ok..
    mz_cut = mz_sorted[:cut] # take only the biggest peaks
    i_cut = i_sorted[:cut]
    
    # sort arrays back
    index = mz_cut.argsort()
    mz = mz_cut[index]
    i = i_cut[index]

    return mz, i