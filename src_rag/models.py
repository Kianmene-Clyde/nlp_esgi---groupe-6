import numpy as np
import re
import tiktoken
import openai
import yaml
from pathlib import Path

from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
CONF = yaml.safe_load(open(ROOT / "config.yml"))

CLIENT = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=CONF["groq_key"],
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def get_model(config):
    if config:
        return RAG(**config.get("model", {}))
    else:
        return RAG()


class RAG:
    def __init__(self, chunk_size: int = 256, overlap: int = 0, small2big: bool = False, embedding: str = "miniLM"):
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._small2big = small2big
        self._embedding_name = embedding

        self._embedder = None
        self._loaded_files = set()
        self._texts: list[str] = []
        self._chunks: list[str] = []
        self._corpus_embedding: np.ndarray | None = None
        self._client = CLIENT

    # ---------------------------
    # Chargement & chunks
    # ---------------------------
    def load_files(self, filenames):
        texts = []
        for filename in filenames:
            with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                texts.append(f.read())
                self._loaded_files.add(filename)

        self._texts += texts

        chunks_added = self._compute_chunks(texts)
        self._chunks += chunks_added

        new_embedding = self.embed_corpus(chunks_added)
        if self._corpus_embedding is not None:
            self._corpus_embedding = np.vstack([self._corpus_embedding, new_embedding])
        else:
            self._corpus_embedding = new_embedding

    def _compute_chunks(self, texts):
        return sum(
            (chunk_markdown(txt, chunk_size=self._chunk_size, overlap=self._overlap) for txt in texts),
            [],
        )

    def get_corpus_embedding(self):
        return self._corpus_embedding

    def get_chunks(self):
        return self._chunks

    # ---------------------------
    # Embeddings
    # ---------------------------
    def get_embedder(self):
        if self._embedder is None:
            model_name_map = {
                "miniLM": "all-MiniLM-L6-v2",  # léger, bon baseline
                "mpnet": "all-mpnet-base-v2",  # plus lourd, souvent meilleur
                "bge": "BAAI/bge-base-en-v1.5",  # version SentenceTransformers de BGE
            }

            hf_name = model_name_map.get(self._embedding_name, self._embedding_name)

            self._embedder = SentenceTransformer(
                hf_name,
                device="cpu",  # on reste sur CPU pour éviter les galères GPU Windows
            )
        return self._embedder

    def embed_corpus(self, chunks):
        embedder = self.get_embedder()
        return embedder.encode(
            chunks,
            batch_size=16,  # limite l’empreinte mémoire
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    def embed_questions(self, questions):
        embedder = self.get_embedder()
        return embedder.encode(
            questions,
            batch_size=16,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    # ---------------------------
    # RAG : retrieval + LLM Groq
    # ---------------------------
    def reply(self, query: str) -> str:
        prompt = self._build_prompt(query)
        res = self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return res.choices[0].message.content

    def _build_prompt(self, query: str) -> str:
        context_str = "\n".join(self._get_context(query))

        return f"""Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the answer is not in the context information, reply "I cannot answer that question".
Query: {query}
Answer:"""

    def _get_context(self, query: str) -> list[str]:
        # 1) embed la question
        query_embedding = self.embed_questions([query])  # (1, d)
        # 2) scores de similarité avec tout le corpus
        sim_scores = query_embedding @ self._corpus_embedding.T  # (1, N)

        if not self._small2big:
            # Comportement classique : top-5 chunks indépendants
            top_k = 5
            idxs = list(np.argsort(sim_scores[0]))[-top_k:]
            return [self._chunks[i] for i in idxs]

        # small2Big = True → on prend plus de petits chunks et on fusionne
        top_k_small = 10
        idxs = list(np.argsort(sim_scores[0]))[-top_k_small:]
        idxs = sorted(idxs)  # on remet dans l'ordre du texte

        merged_chunks: list[str] = []
        current_group = [idxs[0]]

        for idx in idxs[1:]:
            if idx == current_group[-1] + 1:
                # même zone du texte → on agrandit le groupe
                current_group.append(idx)
            else:
                # on ferme le groupe et on crée un gros chunk
                merged_chunks.append("\n".join(self._chunks[i] for i in current_group))
                current_group = [idx]

        # dernier groupe
        merged_chunks.append("\n".join(self._chunks[i] for i in current_group))

        # on limite à 5 "gros" chunks max pour le prompt
        return merged_chunks[:5]


# ---------------------------
# Utils : tokenisation & chunking
# ---------------------------
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def lm(md_text: str) -> list[dict[str, str]]:
    """
    Parses markdown into a list of {'headers': [...], 'content': ...}
    Preserves full header hierarchy (e.g. ["Section", "Sub", "SubSub", ...])
    """
    pattern = re.compile(r"^(#{1,6})\s*(.+)$")
    lines = md_text.splitlines()

    sections = []
    header_stack = []
    current_section = {"headers": [], "content": ""}

    for line in lines:
        match = pattern.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()

            # Save previous section
            if current_section["content"]:
                sections.append(current_section)

            # Adjust the header stack
            header_stack = header_stack[: level - 1]
            header_stack.append(title)

            current_section = {
                "headers": header_stack.copy(),
                "content": "",
            }
        else:
            current_section["content"] += line + "\n"

    if current_section["content"]:
        sections.append(current_section)

    return sections


def chunk_markdown(md_text: str, chunk_size: int = 128, overlap: int = 0) -> list[str]:
    """
    Découpe le texte markdown en chunks avec overlap optionnel.
    """
    if overlap >= chunk_size:
        raise ValueError(f"Overlap ({overlap}) doit être inférieur à chunk_size ({chunk_size})")

    parsed_sections = parse_markdown_sections(md_text)
    chunks: list[str] = []

    for section in parsed_sections:
        tokens = tokenizer.encode(section["content"])
        step = chunk_size - overlap

        token_chunks = []
        for i in range(0, len(tokens), step):
            chunk = tokens[i: i + chunk_size]
            if chunk:
                token_chunks.append(chunk)
            if i + chunk_size >= len(tokens):
                break

        for token_chunk in token_chunks:
            chunk_text = tokenizer.decode(token_chunk)
            chunks.append(chunk_text)

    return chunks
