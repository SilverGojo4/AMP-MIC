# pylint: disable=line-too-long, import-error, wrong-import-position, too-many-locals, too-many-arguments, too-many-positional-arguments, too-many-branches
"""
This script generates amino acid letter embeddings using pretrained protein language models
(e.g., BERT, T5, XLNet). Each letter (A, C, D, ...) is encoded into a vector and saved as a `.npy` file.
"""
# ============================== Standard Library Imports ==============================
import logging
import os
import sys
import time

# ============================== Third-Party Library Imports ==============================
import numpy as np
import torch
from transformers import (
    BertModel,
    BertTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    XLNetModel,
    XLNetTokenizer,
)

# ============================== Project Root Path Setup ==============================
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
LOGGING_PATH = os.path.join(BASE_PATH, "src/utils/logging_toolkit/src/python")

# Fix Python Path
if BASE_PATH not in sys.path:
    sys.path.append(BASE_PATH)
if LOGGING_PATH not in sys.path:
    sys.path.append(LOGGING_PATH)

# ============================== Project-Specific Imports ==============================
# Logging configuration and custom logger
from setup_logging import CustomLogger


# ============================== Custom Function ==============================
class LetterEmbedding:
    """
    Generates embeddings for amino acid letters using a pretrained language model.
    """

    def __init__(
        self,
        language_model: str,
        max_len: int,
        preferred_gpu: int = 0,
    ):
        """
        Initialize the LetterEmbedding class.

        Parameters
        ----------
        language_model : str
            Name of the pretrained model to use.
        max_len : int
            Maximum sequence length for later usage.
        preferred_gpu : int
            Preferred GPU index (default is 0).
        """
        self.language_model = language_model
        self.max_len = max_len

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if preferred_gpu >= gpu_count or preferred_gpu < 0:
                preferred_gpu = 0
            self.device = torch.device(f"cuda:{preferred_gpu}")
        else:
            self.device = torch.device("cpu")

        # Load tokenizer and model
        if self.language_model == "BERT-BFD":
            self.tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert_bfd", do_lower_case=False
            )
            self.model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")
        elif self.language_model == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False
            )
            self.model = BertModel.from_pretrained("Rostlab/prot_bert")
        elif self.language_model == "T5-XL-BFD":
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_bfd", do_lower_case=False
            )
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_bfd")
        elif self.language_model == "T5-XL-UNI":
            self.tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
            )
            self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        elif self.language_model == "XLNET":
            self.tokenizer = XLNetTokenizer.from_pretrained(
                "Rostlab/prot_xlnet", do_lower_case=False
            )
            self.model = XLNetModel.from_pretrained("Rostlab/prot_xlnet", mem_len=512)
        else:
            raise ValueError(f"Unsupported language model: {self.language_model}")

        self.model = self.model.to(self.device).eval()  # type: ignore

    def generate_letter_embeddings(self):
        """
        Generate embeddings for standard amino acid letters.

        Returns
        -------
        dict
            A dictionary mapping each amino acid to its embedding.
        """
        amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
        embeddings = {}
        for aa in amino_acids:
            tokens = self.tokenizer(
                " ".join(aa), return_tensors="pt", add_special_tokens=True
            )
            input_ids = tokens["input_ids"].to(self.device)
            attention_mask = tokens["attention_mask"].to(self.device)
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embedding = output.last_hidden_state[:, 0, :]
                embeddings[aa] = embedding.squeeze(0).cpu().numpy()
        return embeddings

    def save_embeddings(self, embeddings: dict, output_path: str):
        """
        Save embeddings to a .npy file.

        Parameters
        ----------
        embeddings : dict
            Letter embeddings.
        output_path : str
            File path to save.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, embeddings)  # type: ignore


def encode_fasta_to_word_embedding(
    input_fasta: str,
    output_npz: str,
    llm_name: str,
    embedding_dict_path: str,
    max_len: int,
    logger: CustomLogger,
) -> None:
    """
    Encode protein sequences in FASTA file using precomputed amino acid embeddings.

    Parameters
    ----------
    input_fasta : str
        Path to input FASTA file.
    output_npz : str
        Path to save encoded `.npz` result.
    llm_name : str
        Identifier for the language model.
    embedding_dict_path : str
        Path to `.npy` file with precomputed letter embeddings.
    max_len : int
        Fixed max length for padded/truncated embedding matrices.
    logger : CustomLogger
        Logging object for structured output.
    """
    try:
        logger.info(msg=f"/ Task: Starting word embedding. (LLM: '{llm_name}')")
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

        # Load letter embeddings (.npy)
        letter_embeddings = np.load(embedding_dict_path, allow_pickle=True).item()
        emb_dim = next(iter(letter_embeddings.values())).shape[0]

        # Read FASTA
        sequence_ids = []
        sequences = []
        with open(input_fasta, "r", encoding="utf-8") as file:
            seq = ""
            seq_id = None
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    if seq and seq_id:
                        sequence_ids.append(seq_id)
                        sequences.append(seq)
                        seq = ""
                    seq_id = line[1:]
                else:
                    seq += line
            if seq and seq_id:
                sequence_ids.append(seq_id)
                sequences.append(seq)

        if len(sequences) == 0:
            raise ValueError(f"No sequences found in '{input_fasta}'.")

        # Encode each sequence
        encoded = np.zeros((len(sequences), max_len, emb_dim), dtype=np.float32)
        for i, seq in enumerate(sequences):
            seq_emb = []
            for aa in seq:
                if aa in letter_embeddings:
                    seq_emb.append(letter_embeddings[aa])
                else:
                    logger.warning(
                        f"Unknown letter '{aa}' in seq '{sequence_ids[i]}'. Using zeros."
                    )
                    seq_emb.append(np.zeros(emb_dim))
            seq_emb = np.array(seq_emb)
            if seq_emb.shape[0] > max_len:
                seq_emb = seq_emb[:max_len]
            elif seq_emb.shape[0] < max_len:
                padding = np.zeros((max_len - seq_emb.shape[0], emb_dim))
                seq_emb = np.vstack([seq_emb, padding])
            encoded[i] = seq_emb

        # Save
        os.makedirs(os.path.dirname(output_npz), exist_ok=True)
        np.savez(
            output_npz,
            identifiers=np.array(sequence_ids),
            sequences=np.array(sequences),
            encoded_matrices=encoded,
            max_len=max_len,
            embedding_dim=emb_dim,
        )

        logger.log_with_borders(
            level=logging.INFO,
            message=f"Saved encoded '.npz' file:\n'{output_npz}'",
            border="|",
            length=100,
        )
        logger.add_divider(level=logging.INFO, length=100, border="+", fill="-")

    except Exception:
        logger.exception(msg="Unexpected error in 'encode_fasta_to_word_embedding()'.")
        raise


def run_word_embedding_pipeline(base_path: str, logger: CustomLogger) -> None:
    """
    Entry point for the word embedding-based encoding pipeline.

    For each (strain, dataset type, language model), the following steps are executed:
    1. Load the corresponding FASTA file.
    2. Use precomputed letter embeddings (.npy).
    3. Encode the sequence into a fixed-size matrix.
    4. Save the results to a `.npz` file.

    Parameters
    ----------
    base_path : str
        Absolute base path of the project.
    logger : CustomLogger
        Logger instance for structured logging.

    Returns
    -------
    None
    """
    try:
        # Define strain shortcodes and data split types
        strains = {
            "Escherichia coli": "EC",
            "Pseudomonas aeruginosa": "PA",
            "Staphylococcus aureus": "SA",
        }
        data_types = ["train", "test"]
        llm_list = ["BERT-BFD", "BERT", "T5-XL-BFD", "T5-XL-UNI", "XLNET"]
        max_len = 64

        # Loop over each strain type
        for strain_index, (strain, suffix) in enumerate(strains.items(), start=1):
            for data_index, data_type in enumerate(data_types, start=1):

                # Logging: strain section
                logger.info(msg=f"/ {strain_index}.{data_index}")
                logger.add_divider(level=logging.INFO, length=60, border="+", fill="-")
                logger.log_with_borders(
                    level=logging.INFO,
                    message=f"Strain - '{strain}' -> Data - '{data_type}'",
                    border="|",
                    length=60,
                )
                logger.add_divider(level=logging.INFO, length=60, border="+", fill="-")
                logger.add_spacer(level=logging.INFO, lines=1)

                # Define input path
                fasta_file = os.path.join(
                    base_path, f"data/processed/{suffix}/{data_type}.fasta"
                )

                # Loop over each LLM
                for llm_name in llm_list:
                    embedding_path = os.path.join(
                        base_path, f"experiments/embeddings/{llm_name}.npy"
                    )
                    output_npz_path = os.path.join(
                        base_path,
                        f"data/processed/{suffix}/{data_type}_embed_{llm_name}.npz",
                    )

                    start = time.time()
                    encode_fasta_to_word_embedding(
                        input_fasta=fasta_file,
                        output_npz=output_npz_path,
                        llm_name=llm_name,
                        embedding_dict_path=embedding_path,
                        max_len=max_len,
                        logger=logger,
                    )
                    duration = time.time() - start
                    logger.log_with_borders(
                        level=logging.INFO,
                        message=f"Time taken for '{llm_name}' embedding: {duration:.2f} sec",
                        border="|",
                        length=100,
                    )
                    logger.add_divider(
                        level=logging.INFO, length=100, border="+", fill="-"
                    )
                    logger.add_spacer(level=logging.INFO, lines=1)

    except Exception:
        logger.exception(msg="Unexpected error in 'run_word_embedding_pipeline()'.")
        raise


if __name__ == "__main__":
    model_list = ["BERT-BFD", "BERT", "T5-XL-BFD", "T5-XL-UNI", "XLNET"]
    output_dir = os.path.join(BASE_PATH, "experiments/embeddings")
    os.makedirs(output_dir, exist_ok=True)
    for model_name in model_list:
        embedder = LetterEmbedding(language_model=model_name, max_len=64)
        letter_vecs = embedder.generate_letter_embeddings()
        save_path = os.path.join(output_dir, f"{model_name}.npy")
        embedder.save_embeddings(letter_vecs, save_path)
