# Modified 2026-10-16 for the GAPD release candidate; see CHANGELOG.md.
"""
SimCSE tool for semantic similarity computation (adapted from KBQA-o1)
Used for relation selection in KBQA-R1
"""
import json
import logging
import os
import threading
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

try:
    from limit import relation_list
except ImportError:  # pragma: no cover
    relation_list = []
from numpy import ndarray
from sklearn.metrics.pairwise import cosine_similarity
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class SimCSE(object):
    """
    A class for embedding sentences, calculating similarities, and retrieving sentences by SimCSE.
    Enhanced version with KBQA-o1 features including indexing and search capabilities.
    """
    def __init__(self, model_name_or_path: str, 
                device: str = None,
                num_cells: int = 100,
                num_cells_in_search: int = 10,
                pooler = None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self._device_lock = threading.RLock()
        self._current_device: Optional[str] = None

        # Index-related attributes
        self.index = None
        self.is_faiss_index = False
        self.num_cells = num_cells
        self.num_cells_in_search = num_cells_in_search

        if pooler is not None:
            self.pooler = pooler
        elif "unsup" in model_name_or_path:
            logger.info("Use `cls_before_pooler` for unsupervised models. If you want to use other pooling policy, specify `pooler` argument.")
            self.pooler = "cls_before_pooler"
        else:
            self.pooler = "cls"

        self._move_model_to(self.device)

    def _move_model_to(self, target_device: str) -> None:
        """Move underlying torch model to target device once, guarding with a lock."""
        if target_device is None:
            target_device = self.device
        if self._current_device == target_device:
            return
        with self._device_lock:
            if self._current_device == target_device:
                return
            self.model.to(target_device)
            self._current_device = target_device
    
    def encode(self, sentence: Union[str, List[str]], 
                device: str = None, 
                return_numpy: bool = False,
                normalize_to_unit: bool = True,
                keepdim: bool = False,
                batch_size: int = 64,
                max_length: int = 128) -> Union[ndarray, Tensor]:

        target_device = self.device if device is None else device
        self._move_model_to(target_device)
        
        single_sentence = False
        if isinstance(sentence, str):
            sentence = [sentence]
            single_sentence = True

        embedding_list = [] 
        with torch.no_grad():
            total_batch = len(sentence) // batch_size + (1 if len(sentence) % batch_size > 0 else 0)
            for batch_id in range(total_batch):           # for batch_id in tqdm(range(total_batch)):
                batch_sentences = sentence[batch_id*batch_size:(batch_id+1)*batch_size]
                
                inputs = self.tokenizer(
                    batch_sentences,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt"
                )
                inputs = {k: v.to(target_device) for k, v in inputs.items()}
                outputs = self.model(**inputs, return_dict=True)
                
                if self.pooler == "cls":
                    embeddings = outputs.pooler_output
                elif self.pooler == "cls_before_pooler":
                    embeddings = outputs.last_hidden_state[:, 0]
                else:
                    raise NotImplementedError
                
                if normalize_to_unit:
                    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
                embedding_list.append(embeddings.cpu())
        
        embeddings = torch.cat(embedding_list, 0)
        
        if single_sentence and not keepdim:
            embeddings = embeddings[0]
        
        if return_numpy and not isinstance(embeddings, ndarray):
            return embeddings.numpy()
        return embeddings
    
    def similarity(self, queries: Union[str, List[str]], 
                   keys: Union[str, List[str], ndarray], 
                   device: str = None) -> Union[float, ndarray]:
        """
        Compute similarity between queries and keys
        Enhanced version with proper dimension handling from KBQA-o1
        """
        query_vecs = self.encode(queries, device=device, return_numpy=True) # suppose N queries
        
        if not isinstance(keys, ndarray):
            key_vecs = self.encode(keys, device=device, return_numpy=True) # suppose M keys
        else:
            key_vecs = keys

        # check whether N == 1 or M == 1 (more robust than checking input types)
        single_query, single_key = len(query_vecs.shape) == 1, len(key_vecs.shape) == 1 
        if single_query:
            query_vecs = query_vecs.reshape(1, -1)
        if single_key:
            key_vecs = key_vecs.reshape(1, -1)
        
        # returns an N*M similarity array
        similarities = cosine_similarity(query_vecs, key_vecs)
        
        if single_query:
            similarities = similarities[0]
            if single_key:
                similarities = float(similarities[0])
        
        return similarities
    
    def build_index(self, sentences_or_file_path: Union[str, List[str]], 
                       use_faiss: bool = None,
                       faiss_fast: bool = False,
                       device: str = None,
                       batch_size: int = 64,
                       cache_dir: str = None,
                       cache_key: str = None):
        """
        Build index for efficient similarity search
        """
        if use_faiss is None or use_faiss:
            try:
                import faiss
                assert hasattr(faiss, "IndexFlatIP")
                use_faiss = True 
            except ImportError:
                logger.warning("Fail to import faiss. If you want to use faiss, install faiss through PyPI. Now the program continues with brute force search.")
                use_faiss = False
        
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        # Try load cached FAISS index if requested
        if use_faiss and cache_dir and cache_key:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                base_path = os.path.join(cache_dir, cache_key)
                index_path = base_path + '.faiss'
                sents_path = base_path + '.sentences.json'
                if os.path.isfile(index_path) and os.path.isfile(sents_path):
                    import faiss
                    logger.info(f"Loading FAISS index from {index_path}")
                    index_cpu = faiss.read_index(index_path)
                    with open(sents_path, 'r', encoding='utf-8') as f:
                        sentences_loaded = json.load(f)
                    self.index = {"sentences": sentences_loaded, "index": index_cpu}
                    self.is_faiss_index = True
                    logger.info("Loaded cached FAISS index successfully")
                    return
            except Exception as e:
                logger.warning(f"Failed to load cached FAISS index: {e}. Will rebuild.")
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)

        logger.info("Building index...")
        self.index = {"sentences": sentences_or_file_path}
        
        if use_faiss:
            import faiss
            quantizer = faiss.IndexFlatIP(embeddings.shape[1])  
            if faiss_fast:
                index = faiss.IndexIVFFlat(quantizer, embeddings.shape[1], min(self.num_cells, len(sentences_or_file_path)), faiss.METRIC_INNER_PRODUCT) 
            else:
                index = quantizer

            if (self.device == "cuda" and device != "cpu") or device == "cuda":
                if hasattr(faiss, "StandardGpuResources"):
                    logger.info("Use GPU-version faiss")
                    res = faiss.StandardGpuResources()
                    res.setTempMemory(512 * 1024 * 1024)
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                else:
                    logger.info("Use CPU-version faiss")
            else: 
                logger.info("Use CPU-version faiss")

            if faiss_fast:            
                index.train(embeddings.astype(np.float32))
            index.add(embeddings.astype(np.float32))
            if hasattr(index, 'nprobe'):
                index.nprobe = min(self.num_cells_in_search, len(sentences_or_file_path))
            self.is_faiss_index = True
        else:
            index = embeddings
            self.is_faiss_index = False
        self.index["index"] = index
        logger.info("Finished")

        # Save FAISS index if requested
        if use_faiss and cache_dir and cache_key:
            try:
                import faiss
                os.makedirs(cache_dir, exist_ok=True)
                base_path = os.path.join(cache_dir, cache_key)
                index_path = base_path + '.faiss'
                sents_path = base_path + '.sentences.json'
                # ensure CPU index is written
                try:
                    index_to_write = faiss.index_gpu_to_cpu(self.index["index"])  # if GPU, convert
                except Exception:
                    index_to_write = self.index["index"]
                faiss.write_index(index_to_write, index_path)
                with open(sents_path, 'w', encoding='utf-8') as f:
                    json.dump(self.index["sentences"], f, ensure_ascii=False)
                logger.info(f"Saved FAISS index to {index_path}")
            except Exception as e:
                logger.warning(f"Failed to save FAISS index: {e}")

    def add_to_index(self, sentences_or_file_path: Union[str, List[str]],
                        device: str = None,
                        batch_size: int = 64):
        """
        Add more sentences to existing index
        """
        # if the input sentence is a string, we assume it's the path of file that stores various sentences
        if isinstance(sentences_or_file_path, str):
            sentences = []
            with open(sentences_or_file_path, "r") as f:
                logging.info("Loading sentences from %s ..." % (sentences_or_file_path))
                for line in tqdm(f):
                    sentences.append(line.rstrip())
            sentences_or_file_path = sentences
        
        logger.info("Encoding embeddings for sentences...")
        embeddings = self.encode(sentences_or_file_path, device=device, batch_size=batch_size, normalize_to_unit=True, return_numpy=True)
        
        if self.is_faiss_index:
            self.index["index"].add(embeddings.astype(np.float32))
        else:
            self.index["index"] = np.concatenate((self.index["index"], embeddings))
        self.index["sentences"] += sentences_or_file_path
        logger.info("Finished")

    def search(self, queries: Union[str, List[str]], 
                device: str = None, 
                threshold: float = 0.6,
                top_k: int = 5) -> Union[List[Tuple[str, float]], List[List[Tuple[str, float]]]]:
        """
        Search for similar sentences using built index
        """
        if not self.is_faiss_index:
            if isinstance(queries, list):
                combined_results = []
                for query in queries:
                    results = self.search(query, device, threshold, top_k)
                    combined_results.append(results)
                return combined_results
            
            similarities = self.similarity(queries, self.index["index"]).tolist()
            id_and_score = []
            for i, s in enumerate(similarities):
                if s >= threshold:
                    id_and_score.append((i, s))
            id_and_score = sorted(id_and_score, key=lambda x: x[1], reverse=True)[:top_k]
            results = [(self.index["sentences"][idx], score) for idx, score in id_and_score]
            return results
        else:
            query_vecs = self.encode(queries, device=device, normalize_to_unit=True, keepdim=True, return_numpy=True)

            distance, idx = self.index["index"].search(query_vecs.astype(np.float32), top_k)
            
            def pack_single_result(dist, idx):
                results = [(self.index["sentences"][i], s) for i, s in zip(idx, dist) if s >= threshold]
                return results
            
            if isinstance(queries, list):
                combined_results = []
                for i in range(len(queries)):
                    results = pack_single_result(distance[i], idx[i])
                    combined_results.append(results)
                return combined_results
            else:
                return pack_single_result(distance[0], idx[0])


def get_default_simcse_model(model_name: Optional[str] = None) -> SimCSE:
    """
    Get default SimCSE model for KBQA-R1
    
    Args:
        model_name: Model name or path, defaults to princeton-nlp model
        
    Returns:
        SimCSE instance
    """
    model_name = model_name or os.getenv("SIMCSE_MODEL_PATH", "princeton-nlp/sup-simcse-roberta-large")
    try:
        return SimCSE(model_name)
    except Exception as e:
        logger.warning(f"Failed to load {model_name}, falling back to basic model: {e}")
        # Fallback to a more basic model if available
        return SimCSE("sentence-transformers/all-MiniLM-L6-v2")


if __name__=="__main__":
    example_sentences = relation_list
    example_queries = [
        'measurement_unit.measurement_system.units',
    ]

    model_name = os.getenv("SIMCSE_MODEL_PATH", "princeton-nlp/sup-simcse-roberta-large")
    simcse = SimCSE(model_name)

    print("\n=========Calculate cosine similarities between queries and sentences============\n")
    similarities = simcse.similarity(example_queries, example_sentences)
    print(similarities)

    print("\n=========Naive brute force search============\n")
    simcse.build_index(example_sentences, use_faiss=False)
    results = simcse.search(example_queries, top_k=50)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")
    
    print("\n=========Search with Faiss backend============\n")
    simcse.build_index(example_sentences, use_faiss=True)
    results = simcse.search(example_queries, top_k=50)
    for i, result in enumerate(results):
        print("Retrieval results for query: {}".format(example_queries[i]))
        for sentence, score in result:
            print("    {}  (cosine similarity: {:.4f})".format(sentence, score))
        print("")
