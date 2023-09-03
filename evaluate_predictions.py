# imports
from io import TextIOWrapper
import pandas as pd
import pathlib
import typer
import json
from statistics import mean
from tqdm import tqdm
from rdkit import Chem, DataStructs
import numpy as np
from collections import defaultdict
from icecream import ic
import plotly
import plotly.express as px
import yaml
import time
from datetime import datetime


app = typer.Typer(pretty_exceptions_enable=False)

def parse_predictions_path(predictions_path: pathlib.Path) -> tuple:
    """Parse the predictions path to get the dataset name, split and range
    for 'full' range, returns range(0)"""
    try:
        data_name = predictions_path.parent.parent.stem
        data_split, data_range_str = predictions_path.parent.stem.split("_")[1:3]
        if data_range_str == "full":
            data_range = range(0)
        else:
            data_range = range(*map(int, data_range_str.split(":")))

    except Exception as exc:
        raise ValueError("Couldn't deduce labels info from predictions_path, " +
                         "please hold the file naming convention " +
                         "<dataset_name>/<id>_<data_split>_<data_range>_*/predictions.jsonl") from exc
    return data_name, data_split, data_range


def load_labels_from_dataset(dataset_path: pathlib.Path, data_range: range) -> list:
    """Load the labels from the dataset"""
    df = pd.read_pickle(dataset_path)
    if not data_range:
        data_range = range(len(df))
    simles_list = df.iloc[data_range]["smiles"].tolist()
    return iter(simles_list)


def move_file_pointer(num_lines: int, file_pointer: TextIOWrapper) -> None:
    """Move the file pointer a specified number of lines forward"""
    for _ in range(num_lines):
        file_pointer.readline()


def update_counter(sorted_simil: np.ndarray, all_simils: dict) -> None:
    """Add simil values to lists with the same index as their ranking"""
    for i, simil in enumerate(sorted_simil):
        all_simils[i].append(simil)


def dummy_generator():
    i = 0
    while True:
        yield i
        i += 1


def diagram_from_dict(d: dict, title: str):
    """Create a plotly figure from a cumulatively stored simils dict"""
    simils = []
    ks = []
    for k, s in d.items():
        simils += s
        ks += [k] * len(s)
    df = pd.DataFrame({"simil": simils, "k": ks})
    fig = px.box(df, x="k", y="simil", points="all")
    return fig


@app.command()
def main(
    predictions_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the jsonl file with caption predictions"),
    labels_path: pathlib.Path = typer.Option(None, dir_okay=False, file_okay=True, readable=True, help="either .smi file or .pkl (DataFrame with 'smiles' column)"),
    config_path: pathlib.Path = typer.Option(..., dir_okay=False, file_okay=True, readable=True, help="Path to the config file"),
    datasets_folder: pathlib.Path = typer.Option("data/datasets", dir_okay=True, file_okay=False, readable=True, help="Path to the folder containing the datasets (used for automatic labels deduction)"),

) -> None:

    data_name, data_split, data_range = parse_predictions_path(predictions_path)
    if labels_path is None:
        labels_path = datasets_folder / data_name / f"{data_name}_{data_split}.smi"
        if not labels_path.exists():
            print("INFO: Deduced .smi path for labels does not exist, " +
                  "trying to deduce .pkl path for labels")
            labels_path = datasets_folder / data_name / f"{data_name}_{data_split}.pkl"
            if not labels_path.exists():
                raise ValueError("Deduced path for labels does not exist, " +
                                "please specify labels_path argument")
        print(f"INFO: Using labels from {labels_path}, {data_range if data_range else 'full'} " +
               "as range, please check they are correct")
    
    if labels_path.suffix == ".pkl":
        labels_iterator = load_labels_from_dataset(labels_path, data_range)
    elif labels_path.suffix == ".smi":
        labels_iterator = labels_path.open("r")
        move_file_pointer(data_range.start, labels_iterator)
    
    parent_dir = predictions_path.parent
    pred_f = predictions_path.open("r")
    log_file = (parent_dir / "log_file.yaml").open("a+")
    pred_jsonl = {}
    counter_empty_preds = 0
    start_time = time.time()

    simil_all_simils = defaultdict(list) # all simils sorted by similarity with gt (at each ranking)
    prob_all_simils = defaultdict(list)   # all simils sorted by probability (at each ranking)
    for i in tqdm(dummy_generator()):  # basically a while True
        pred_jsonl = pred_f.readline()
        if not pred_jsonl:
            break
        preds = json.loads(pred_jsonl)
        gt_smiles = next(labels_iterator)

        if not preds:
            print("INFO: No predictions for the molecule: ", gt_smiles)
            counter_empty_preds += 1
            continue

        pred_fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)) for smiles in preds.keys()]
        gt_fp = Chem.RDKFingerprint(Chem.MolFromSmiles(gt_smiles))
        smiles_simils = [DataStructs.FingerprintSimilarity(fp, gt_fp) for fp in pred_fps]

        prob_simil = np.stack(np.array(list(zip(preds.values(), smiles_simils))))

        simil_decreasing_index = np.argsort(-prob_simil[:, 1])
        probs_decreasing_index = np.argsort(-prob_simil[:, 0])

        update_counter(prob_simil[simil_decreasing_index][:, 1], simil_all_simils)
        update_counter(prob_simil[probs_decreasing_index][:, 1], prob_all_simils)
        
    simil_average_simil_kth = [mean(simil_all_simils[k]) for k in sorted(simil_all_simils.keys())]
    prob_average_simil_kth = [mean(prob_all_simils[k]) for k in sorted(prob_all_simils.keys())]
    num_predictions_at_k_counter = [len(l[1]) for l in sorted(list(simil_all_simils.items()), key=lambda x: x[0])]

    fig_similsort = diagram_from_dict(simil_all_simils, title="Similarity on the k-th position (sorted by ground truth similarity)")
    fig_probsort = diagram_from_dict(prob_all_simils, title="Similarity on the k-th position (sorted by generation probability)")
    df_top1 = pd.DataFrame({"simil": simil_all_simils[0], "prob": prob_all_simils[0]})
    fig_top1_simil_simils = px.histogram(df_top1, x="simil", nbins=100, labels={'x':'similarity', 'y':'count'})
    fig_top1_prob_simils = px.histogram(df_top1, x="prob", nbins=100, labels={'x':'similarity', 'y':'count'})

    fig_similsort.write_image(str(parent_dir / "topk_similsort.png"))
    fig_probsort.write_image(str(parent_dir / "topk_probsort.png"))
    fig_top1_simil_simils.write_image(str(parent_dir / "top1_simil_simils.png"))
    fig_top1_prob_simils.write_image(str(parent_dir / "top1_prob_simils.png"))

    finish_time = time.time()
    logs = {"evaluation":
            {"topk_similsort": str(simil_average_simil_kth),
            "topk_probsort": str(prob_average_simil_kth),
            "num_predictions_at_k_counter": str(num_predictions_at_k_counter),
            "counter_empty_preds": str(counter_empty_preds),
            "start_time": datetime.utcfromtimestamp(start_time).strftime("%d/%m/%Y %H:%M:%S"),
            "eval_time": f"{time.strftime('%H:%M:%S', time.gmtime(finish_time - start_time))}"
            }}
    yaml.dump(logs, log_file)
    print(logs)
    

    # results folder 
    
    # # load predictions and gt
    # df_preds = pd.read_pickle(predictions_path, sep=",")

    # if labels_path.suffix == ".csv":
    #     df_labels = pd.read_csv(labels_path)
    # elif labels_path.suffix == ".jsonl":
    #     df_labels = pd.read_json(labels_path, lines=True)
    # else:
    #     raise ValueError(f"labels_path must be a csv or jsonl file, got {labels_path.suffix}")
 
    # # join predictions and labels
    # df = df_preds.merge(df_labels, on="file_name")

    # # join multiple gts to a list and drop original cols
    # df["all_labels"] = df.apply(lambda x: [x.caption_1, x.caption_2, x.caption_3, x.caption_4, x.caption_5], axis=1)
    # df = df.drop(columns=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5"])

    # # init metrics
    # sacrebleu = evaluate.load("sacrebleu")
    # meteor = evaluate.load("meteor")
    # spice = SpiceMetric()
    # cider = CiderMetric()

    # preds_str = df["caption_predicted"].tolist()
    # references = df["all_labels"].tolist()
    # # compute metrics
    # sacrebleu_score = sacrebleu.compute(predictions=preds_str, references=references)
    # meteor_score = meteor.compute(predictions=preds_str, references=references)

    # # coco metrics
    # tokenizer = CocoTokenizer(preds_str, references)
    # tokens = tokenizer.tokenize()
    # spice_score = spice.compute(predictions=preds_str, references=references, tokens=tokens)
    # cider_score = cider.compute(predictions=preds_str, references=references, tokens=tokens)
    # spider_score = 0.5 * (spice_score['average_score'] + cider_score['score'])

    # output_dict = {"metric_computation": {
    #                     "predictions file": str(predictions_path),
    #                     "ground truth file": str(labels_path),
    #                     "computed metrics": {
    #                         "sacrebleu": sacrebleu_score["score"],
    #                         "meteor": meteor_score["meteor"],
    #                         "spice": spice_score['average_score'],
    #                         "cider": cider_score['score'],
    #                         "spider": spider_score
    #                     }
    #                 }
    #               }
    # print(json.dumps(output_dict, indent=4, sort_keys=False))

    # log_file = predictions_path.parent / (predictions_path.stem + '_log.json')
    # with open(log_file, "r+") as f:
    #     try:
    #         log_dict = json.load(f)
    #         log_dict["metric_computation"] = output_dict["metric_computation"]
    #     except json.decoder.JSONDecodeError:
    #         print("No log_file => Creating new log file")
    #         log_dict = output_dict

    # with open(log_file, "w") as f:
    #     json.dump(log_dict, open(log_file, "w"), indent=2, ensure_ascii=False)
    # with open(log_file.parent / "all_spiders", "a") as f:
    #     f.write(str(predictions_path.stem) 
    #             + ":" 
    #             + " " * (80 - len(str(predictions_path.stem))) 
    #             + f"{spider_score:.4f}"
    #             + "\n")

if __name__ == "__main__":
    app()
