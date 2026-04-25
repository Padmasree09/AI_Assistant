from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.retriever import index_documents


SAMPLE_DOCUMENTS = [
    {
        "source": "sample-transformers-overview",
        "text": (
            "Transformers are neural network architectures built around self-attention. "
            "They process tokens in parallel, capture long-range dependencies effectively, "
            "and are widely used in modern language models."
        ),
    },
    {
        "source": "sample-rnn-overview",
        "text": (
            "Recurrent neural networks process sequences step by step and maintain hidden state "
            "across time. They are useful for sequential data but can struggle with long-term dependencies."
        ),
    },
    {
        "source": "sample-transformer-vs-rnn",
        "text": (
            "Compared with RNNs, transformers usually train faster on modern hardware because they "
            "parallelize better. RNNs may be simpler for some sequence tasks, but transformers dominate "
            "many NLP workloads due to stronger long-context modeling."
        ),
    },
]


if __name__ == "__main__":
    count = index_documents(SAMPLE_DOCUMENTS)
    print(f"Indexed {count} sample documents.")
