import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism to avoid fork issues
warnings.filterwarnings('ignore')

from dotenv import load_dotenv

load_dotenv()

rits_api_key = os.getenv("RITS_API_KEY")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf", help="The spacy model to use")
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-m3", help="The path of embedding model to use")
    parser.add_argument("--dataset_name", type=str, default="agri-rag-telugu", help="The dataset to use")
    parser.add_argument("--llm_model", type=str, default="google/gemma-4-26B-A4B-it", help="The LLM model to use")
    parser.add_argument("--llm_base_url", type=str, default="http://cccxc624.pok.ibm.com:8000/v1", help="vllm_base_url")
    parser.add_argument("--max_workers", type=int, default=16, help="The max number of workers to use")
    parser.add_argument("--max_iterations", type=int, default=3, help="The max number of iterations to use")
    parser.add_argument("--iteration_threshold", type=float, default=0.4, help="The threshold for iteration")
    parser.add_argument("--passage_ratio", type=float, default=2, help="The ratio for passage")
    parser.add_argument("--top_k_sentence", type=int, default=3, help="The top k sentence to use")
    parser.add_argument("--use_vectorized_retrieval", action="store_true", help="Use vectorized matrix-based retrieval instead of BFS iteration")
    parser.add_argument(
        "--use_spacy_ner",
        action="store_true",
        help="Whether to use spacy ner or LLM ner",
    )
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="Save checkpoint after every N questions")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint file to resume from")
    return parser.parse_args()


# def load_dataset(dataset_name): 
#     questions_path = f"dataset/{dataset_name}/questions.json"
#     with open(questions_path, "r", encoding="utf-8") as f:
#         questions = json.load(f)
#     chunks_path = f"dataset/{dataset_name}/chunks.json"
#     with open(chunks_path, "r", encoding="utf-8") as f:
#         chunks = json.load(f)
#     passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
#     return questions, passages

def load_dataset(dataset_name):
    if dataset_name=='agri-rag-telugu':
        corpus_path = f"/dccstor/indiclm/rudra/IRL-Indic-RAG/data/seed_data.jsonl"
        with open(corpus_path, "r") as f:
            docs = []
            for each_line in f:
                text = json.loads(each_line)

                if "chunk" in text:
                    text = text["chunk"]
                else:
                    text = text["document"]
                docs.append(text)
        passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(docs)]
        questions = []
        TEST_FILE = ( "/dccstor/indiclm/arkadeep/ap_gov/IRL-Indic-RAG/output/multi_hop_qa_eval/combined_multihop.jsonl" )
        with open(TEST_FILE, "r", encoding="utf-8") as f:
            for line in f:
                instance = json.loads(line.strip())

                questions.append(
                    {
                        "question": instance["question"],
                        "answer": instance.get("gold_answer", "")
                    }
                )
    
    return questions, passages


def load_embedding_model(embedding_model):
    embedding_model = SentenceTransformer(embedding_model,device="cpu")
    return embedding_model

def main():
    args = parse_arguments()
    
    # Determine output directory and checkpoint path
    if args.resume_from_checkpoint:
        # Extract time_str from checkpoint path
        checkpoint_path = args.resume_from_checkpoint
        time_str = checkpoint_path.split('/')[-2] if '/' in checkpoint_path else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        print(f"Resuming from checkpoint: {checkpoint_path}")
    else:
        time = datetime.now()
        time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = f"gemma4_26B_multihop_updated_chunks_results/{args.dataset_name}/{time_str}/checkpoint.json"
    
    output_dir = f"gemma4_26B_multihop_updated_chunks_results/{args.dataset_name}/{time_str}"
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logging(f"{output_dir}/log.txt")
    
    embedding_model = load_embedding_model(args.embedding_model)
    questions, passages = load_dataset(args.dataset_name)
    
    llm_model = LLM_Model(model_name=args.llm_model, base_url=args.llm_base_url, rits_api_key=rits_api_key)
    config = LinearRAGConfig(
        dataset_name=args.dataset_name,
        embedding_model=embedding_model,
        spacy_model=args.spacy_model,
        use_spacy_ner=args.use_spacy_ner,
        max_workers=args.max_workers,
        llm_model=llm_model,
        max_iterations=args.max_iterations,
        iteration_threshold=args.iteration_threshold,
        passage_ratio=args.passage_ratio,
        top_k_sentence=args.top_k_sentence,
        use_vectorized_retrieval=args.use_vectorized_retrieval
    )
    
    rag_model = LinearRAG(global_config=config)
    rag_model.index(passages)
    
    # Run QA with checkpoint support
    print(f"Starting QA inference with checkpoint support...")
    print(f"Checkpoint will be saved to: {checkpoint_path}")
    print(f"Checkpoint interval: every {args.checkpoint_interval} questions")
    
    questions = rag_model.qa(
        questions=questions,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Save final predictions
    predictions_path = f"{output_dir}/predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    
    print(f"QA inference completed. Results saved to: {predictions_path}")
    
    # Run evaluation
    evaluator = Evaluator(llm_model=llm_model, predictions_path=predictions_path)
    evaluator.evaluate(max_workers=args.max_workers)
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Checkpoint file removed after successful completion: {checkpoint_path}")
if __name__ == "__main__":
    main()