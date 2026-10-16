"""Global tokenizer lock for thread-safe access to HuggingFace fast tokenizers.

HuggingFace fast tokenizers (Rust-backed via the `tokenizers` crate) use an
internal RefCell-like borrow mechanism that panics with "RuntimeError: Already
borrowed" when accessed concurrently from multiple threads.

This module provides a single process-wide lock that ALL tokenizer calls must
go through during concurrent execution (rollout phase). This includes
tokenizer.encode(), tokenizer.decode(), and tokenizer.pad() — all of which
internally access the Rust tokenizer state.

Usage:
    from kbqa_r1.utils.tokenizer_lock import global_tokenizer_lock

    with global_tokenizer_lock:
        ids = tokenizer.encode(text, add_special_tokens=False)

Or use the convenience wrappers:
    from kbqa_r1.utils.tokenizer_lock import safe_encode, safe_decode, safe_pad
"""

import threading

# Process-wide lock. All tokenizer access during concurrent execution must
# acquire this lock to prevent "Already borrowed" from the Rust tokenizer.
global_tokenizer_lock = threading.Lock()


def safe_encode(tokenizer, text, **kwargs):
    """Thread-safe tokenizer.encode()."""
    with global_tokenizer_lock:
        return tokenizer.encode(text, **kwargs)


def safe_decode(tokenizer, token_ids, **kwargs):
    """Thread-safe tokenizer.decode()."""
    with global_tokenizer_lock:
        return tokenizer.decode(token_ids, **kwargs)


def safe_pad(tokenizer, *args, **kwargs):
    """Thread-safe tokenizer.pad()."""
    with global_tokenizer_lock:
        return tokenizer.pad(*args, **kwargs)