"""Cross-platform command-line entry point for SAG experiments."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_script(script, arguments):
    env = os.environ.copy()
    command = [sys.executable, str(ROOT / script), *arguments]
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def common_api_arguments(parser):
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "text-embedding-ada-002"),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare data and run Provably Secure RAG (SAG) experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser("prepare-data", help="Clean and split a raw dataset")
    clean.add_argument("--dataset", required=True, choices=["healthcaremagic", "erionemails", "billsum", "fnspid"])
    clean.add_argument("--num-data", type=int, default=100)
    clean.add_argument("--num-private", type=int, default=3)
    clean.add_argument("--seed", type=int, default=10)

    public = subparsers.add_parser("build-public", help="Build a public vector database")
    public.add_argument("--dataset", required=True, choices=["healthcaremagic", "erionemails", "billsum", "fnspid"])
    public.add_argument("--method", required=True, choices=["A", "B"])
    public.add_argument("--chunk-size", type=int, default=512)
    public.add_argument("--overlap", type=int, default=20)
    common_api_arguments(public)

    private = subparsers.add_parser("build-private", help="Encrypt private data and build the evaluation knowledge base")
    private.add_argument("--dataset", required=True, choices=["healthcaremagic", "erionemails", "billsum", "fnspid"])
    private.add_argument("--method", required=True, choices=["A", "B"])
    private.add_argument("--user-id", default="demo-user")
    common_api_arguments(private)

    thief = subparsers.add_parser("thief", help="Run a knowledge-base extraction attack")
    thief.add_argument("--dataset", required=True, choices=["healthcaremagic", "erionemails", "billsum", "fnspid"])
    thief.add_argument("--method", required=True, choices=["A", "B"])
    thief.add_argument("--attack", default="Rag_Thief", choices=["Rag_Thief", "PIDE", "DGEA", "GPTGEN", "TGTB", "Pirate", "SPL"])

    mia = subparsers.add_parser("mia", help="Run a membership inference attack")
    mia.add_argument("--dataset", required=True, choices=["healthcaremagic", "erionemails", "billsum", "fnspid"])
    mia.add_argument("--method", required=True, choices=["A", "B"])
    mia.add_argument("--num-samples", type=int)

    reveal = subparsers.add_parser("embedding-reveal", help="Run embedding inversion evaluation")
    reveal.add_argument("--dataset", required=True, choices=["HealthCareMagic", "erionemails", "billsum", "fnspid"])
    reveal.add_argument("--method", required=True, choices=["A", "B"])
    reveal.add_argument("--num-samples", type=int, default=5)
    reveal.add_argument("--gpu", default="0")
    common_api_arguments(reveal)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "api_key"):
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["OPENAI_API_BASE"] = args.api_base
        os.environ["EMBEDDING_MODEL"] = args.embedding_model
    if args.command != "prepare-data" and not os.environ.get("OPENAI_API_KEY"):
        parser.error(
            "OPENAI_API_KEY is required for embedding, retrieval, and attack commands"
        )
    if args.command == "prepare-data":
        run_script("Data_cleaning.py", ["--run-dir", str(ROOT), "--type-of-dataset", args.dataset, "--num-of-data", str(args.num_data), "--num-of-private", str(args.num_private), "--seed-value", str(args.seed)])
    elif args.command == "build-public":
        run_script("Get_public_data.py", ["--run-dir", str(ROOT), "--type-of-dataset", args.dataset, "--is-method-A", str(args.method == "A"), "--chunk-size", str(args.chunk_size), "--overlap", str(args.overlap), "--api-key", args.api_key, "--api-base", args.api_base, "--embedding-model", args.embedding_model])
    elif args.command == "build-private":
        run_script("Get_private_data.py", ["--run-dir", str(ROOT), "--type-of-dataset", args.dataset, "--is-method-A", str(args.method == "A"), "--user-id", args.user_id, "--api-key", args.api_key, "--api-base", args.api_base, "--embedding-model", args.embedding_model])
    elif args.command == "thief":
        run_script("THIEF.py", ["--run-dir", str(ROOT), "--dataset", args.dataset, "--rag-method", args.method, "--method", args.attack])
    elif args.command == "mia":
        command = ["--run-dir", str(ROOT), "--dataset", args.dataset, "--rag-method", args.method]
        if args.num_samples is not None:
            command.extend(["--num-samples", str(args.num_samples)])
        run_script("RAG_MIA.py", command)
    elif args.command == "embedding-reveal":
        run_script("EMBEDDING_REVEAL.py", ["--run-dir", str(ROOT), "--type-of-dataset", args.dataset, "--rag-method", args.method, "--num-samples", str(args.num_samples), "--gpu", args.gpu, "--api-key", args.api_key, "--api-base", args.api_base, "--embedding-model", args.embedding_model])


if __name__ == "__main__":
    main()
