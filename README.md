# Provably Secure Retrieval-Augmented Generation (SAG)

Official implementation of **SAG**, a framework for privacy-preserving and
user-isolated retrieval-augmented generation with formally analyzed encrypted
knowledge-base storage.

## Publications

- [Provably Secure Retrieval-Augmented Generation](https://arxiv.org/abs/2508.01084)
- [Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval](https://arxiv.org/abs/2503.15548)

If this repository supports your work, please cite the relevant paper using the
entries in [`CITATION.cff`](CITATION.cff).

## Scope of the Security Claims

SAG protects private chunks and their embeddings before storage. The proofs in
the paper establish confidentiality and integrity properties under the stated
computational threat model and assumptions, including protected runtime keys
and trusted execution. They do not imply unconditional security of an entire
LLM deployment, external API, retriever, or TEE implementation.

The repository implements two storage schemes:

- **Method A - Static/isolated AES encryption:** private nodes are addressed by
  user-held identifiers and encrypted with a user key.
- **Method B - Chained dynamic key derivation:** private nodes form an encrypted
  chain with per-node keys, integrity metadata, and an authentication trapdoor.

## Repository Layout

```text
API_chat.py             OpenAI-compatible chat and embedding clients
Security_Methods.py     AES, RSA, HKDF, and HMAC helpers
RAG_Methods.py          Chunking, node construction, and retrieval
MethodA_*.py            Method A client/server implementation
MethodB_*.py            Method B client/server implementation
sag_cli.py              Cross-platform experiment entry point
THIEF.py                Knowledge-base extraction attacks
RAG_MIA.py              Membership inference evaluation
EMBEDDING_REVEAL.py     Embedding inversion evaluation
Evaluation_Thief.py     Attack-result metrics and plots
```

Generated databases, raw datasets, logs, API credentials, and private key
material are intentionally excluded from version control.

## Installation

The reference environment uses Python 3.12 and CUDA 12.4:

```bash
conda env create -f environment.yaml
conda activate secure_rag
```

Configure an OpenAI-compatible embedding and chat endpoint through environment
variables. Never commit a populated `.env` file or production credentials.

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"
export EMBEDDING_MODEL="text-embedding-ada-002"
```

PowerShell equivalent:

```powershell
$env:OPENAI_API_KEY = "your-api-key"
$env:OPENAI_API_BASE = "https://api.openai.com/v1"
$env:EMBEDDING_MODEL = "text-embedding-ada-002"
```

## Data Preparation

Raw datasets are not redistributed by this code repository. Obtain them from
their official sources and place them under `Datasets/` using these paths:

```text
Datasets/HealthCareMagic-100k.json
Datasets/emails.csv
Datasets/billsum_v4_1/us_train_data_final_OFFICIAL.jsonl
Datasets/FNSPID/fnspid_samples.json
```

The project supports HealthCareMagic, Enron Emails, BillSum, and FNSPID. Data
and example-result access information is also available from the
[companion artifact repository](https://anonymous.4open.science/r/Dataset_and_Result-4840/).
Review each dataset's license and privacy terms before use or redistribution.

Prepare a small split:

```bash
python sag_cli.py prepare-data --dataset healthcaremagic --num-data 100 --num-private 3
```

## Build the Knowledge Base

Build public vectors, then add encrypted private nodes:

```bash
python sag_cli.py build-public --dataset healthcaremagic --method A
python sag_cli.py build-private --dataset healthcaremagic --method A --user-id demo-user
```

Replace `A` with `B` to run the chained dynamic-key scheme. Generated material
is written to `Storage/` and `Evaluation_Materials/`; both paths are ignored by
Git because they may contain derived dataset content or cryptographic material.

## Security Evaluation

Knowledge-base extraction attacks:

```bash
python sag_cli.py thief --dataset healthcaremagic --method A --attack Rag_Thief
```

Supported attack names are `Rag_Thief`, `PIDE`, `DGEA`, `GPTGEN`, `TGTB`,
`Pirate`, and `SPL`.

Membership inference:

```bash
python sag_cli.py mia --dataset healthcaremagic --method A --num-samples 3
```

Embedding inversion:

```bash
python sag_cli.py embedding-reveal --dataset HealthCareMagic --method A --num-samples 5 --gpu 0
```

Results are written under `Results/`. These evaluations can incur model API
charges and the embedding inversion experiment requires a compatible CUDA GPU.

Run `python sag_cli.py --help` or a subcommand followed by `--help` for all
options.

## Responsible Use

The attack implementations are provided for authorized security evaluation and
reproducible research. Do not run them against systems or data without explicit
permission. See [`SECURITY.md`](SECURITY.md) for reporting vulnerabilities and
handling credentials.

## License

Code is released under the [Apache License 2.0](LICENSE). Dataset, model, and
third-party asset licenses remain with their respective owners.

## Citation

If you use SAG or this implementation in your research, please cite the
relevant paper:

```bibtex
@article{zhou2025provably,
  title         = {Provably Secure Retrieval-Augmented Generation},
  author        = {Zhou, Pengcheng and Feng, Yinglun and Yang, Zhongliang},
  year          = {2025},
  eprint        = {2508.01084},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2508.01084}
}
```

For the preceding privacy-aware RAG work, please cite:

```bibtex
@article{zhou2025privacy,
  title         = {Privacy-Aware RAG: Secure and Isolated Knowledge Retrieval},
  author        = {Zhou, Pengcheng and Feng, Yinglun and Yang, Zhongliang},
  year          = {2025},
  eprint        = {2503.15548},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CR},
  url           = {https://arxiv.org/abs/2503.15548}
}
```
