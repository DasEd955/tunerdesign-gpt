"""app.py - Gradio web interface for interactive inference with the trained GPT model.

Loads the saved checkpoint from saved_model/gpt.pt, reconstructs the model and
vocabulary mapping, and serves a chat style UI where users can submit prompts and
receive autoregressive completions. Conversation history is preserved across turns
within the same session. The interface is intentionally minimal, mirroring the
aesthetic of mainstream LLM chat frontends while making the educational scope of
the model explicit in the UI.
"""

import torch
import gradio as gr

from model.model import load_model, GPTConfig  # GPTConfig must be in scope for torch.load unpickling
from data.vocab import Solution as VocabSolution
from generate import Solution as GenerateSolution


# ---------------------------------------------------------------------------
# Model and Vocabulary Loading
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "saved_model/gpt.pt"
DEFAULT_NEW_CHARS = 200


def load_artifacts(path: str = DEFAULT_MODEL_PATH) -> tuple:
    """Load the trained GPT model and reconstruct vocabulary mappings from a checkpoint.

    Reads the checkpoint at path, restores the model and GPTConfig via load_model,
    then re-derives the vocabulary by encoding the training corpus stored in the
    checkpoint config. Falls back to the default Shakespeare-style corpus used in
    model/model.py if no corpus is embedded in the checkpoint.

    Args:
        path: Filesystem path to the saved .pt checkpoint file.

    Returns:
        tuple: (model, config, stoi, itos) where model is the restored GPT in eval
        mode, config is the GPTConfig dataclass, stoi maps characters to integer IDs,
        and itos maps integer IDs back to characters.

    Raises:
        FileNotFoundError: If no checkpoint exists at path.
    """
    model, config = load_model(path)
    model.eval()

    fallback_corpus = (
        "All the world's a stage, and all the men and women merely players: "
        "they have their exits and their entrances, and one man in his time plays many parts, "
        "his acts being seven ages."
    ) * 5

    corpus = getattr(config, "training_text", fallback_corpus)
    vocab = VocabSolution()
    stoi, itos = vocab.build_vocab(corpus)
    return model, config, stoi, itos


def encode_prompt(prompt: str, stoi: dict) -> torch.Tensor:
    """Encode a user prompt string into a (1, T) integer token tensor.

    Characters in the prompt that are absent from the training vocabulary are
    silently dropped so the encoder never raises a KeyError on unseen input.

    Args:
        prompt: Raw text string supplied by the user.
        stoi: Character-to-integer mapping built from the training corpus.

    Returns:
        torch.Tensor: Long tensor of shape (1, T) where T is the number of
        recognised characters in prompt, or shape (1, 1) containing token 0
        if no characters matched the vocabulary.
    """
    ids = [stoi[ch] for ch in prompt if ch in stoi]
    if not ids:
        ids = [0]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def generate_response(
    model,
    config,
    stoi: dict,
    itos: dict,
    prompt: str,
    new_chars: int = DEFAULT_NEW_CHARS,
) -> str:
    """Run autoregressive inference on prompt and return the generated continuation.

    Encodes the prompt into token IDs, calls GenerateSolution.generate to
    autoregressively sample new_chars characters, and returns the raw output
    string. The model is always in eval mode before generation begins.

    Args:
        model: Trained GPT instance in eval mode.
        config: GPTConfig holding context_length and other hyperparameters.
        stoi: Character-to-integer vocabulary mapping.
        itos: Integer-to-character vocabulary mapping.
        prompt: User-supplied input text used as the generation seed.
        new_chars: Number of new characters to generate.

    Returns:
        str: The model's generated continuation of length new_chars.
    """
    model.eval()
    context = encode_prompt(prompt, stoi)
    generator = GenerateSolution()
    return generator.generate(
        model=model,
        new_chars=new_chars,
        context=context,
        context_length=config.context_length,
        int_to_char=itos,
    )


# ---------------------------------------------------------------------------
# Gradio Chat Logic
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This demo showcases inference only. "
    "Since the model is intentionally trained on a small dataset for educational "
    "purposes (and due to the scale of data it would require, outputs mirror the "
    "language of the training corpus but with great brevity, as outlined in the "
    "README), outputs are largely nonsensical."
)


def build_respond_fn(model, config, stoi: dict, itos: dict):
    """Return a Gradio compatible respond function closed over the loaded model artifacts.

    The returned function follows the signature expected by gr.ChatInterface:
    it receives the current user message and the full chat history list, appends
    the model's response to the history, and yields the updated history so the
    UI can stream incrementally.

    Args:
        model: Trained GPT instance in eval mode.
        config: GPTConfig holding context_length and other hyperparameters.
        stoi: Character-to-integer vocabulary mapping.
        itos: Integer-to-character vocabulary mapping.

    Returns:
        Callable: A generator function with signature
        (message: str, history: list[dict]) -> Generator[list[dict], None, None]
        suitable for use as the fn argument of gr.ChatInterface.
    """

    def respond(message: str, history: list) -> str:
        """Generate a model response for message and return the reply string.

        Args:
            message: The user's current input text.
            history: List of prior {role, content} dicts maintained by ChatInterface.

        Returns:
            str: The model's generated reply text.
        """
        reply = generate_response(model, config, stoi, itos, message)
        return reply

    return respond


def build_interface(model, config, stoi: dict, itos: dict) -> gr.ChatInterface:
    """Construct and return the Gradio ChatInterface for the Tuner Design GPT demo.

    Assembles the chat UI with a branded title, the educational disclaimer as the
    description, and a placeholder hint in the input box. Multiturn conversation
    history is handled natively by gr.ChatInterface.

    Args:
        model: Trained GPT instance in eval mode.
        config: GPTConfig holding context_length and other hyperparameters.
        stoi: Character-to-integer vocabulary mapping.
        itos: Integer-to-character vocabulary mapping.

    Returns:
        gr.ChatInterface: A fully configured Gradio chat interface ready to launch.
    """
    respond_fn = build_respond_fn(model, config, stoi, itos)

    interface = gr.ChatInterface(
        fn=respond_fn,
        title="Tuner Design GPT",
        description=DISCLAIMER,
        chatbot=gr.Chatbot(placeholder="Welcome to Tuner Design GPT"),
        textbox=gr.Textbox(
            placeholder="Type a prompt and press [Enter] or click [Send]...",
            container=False,
            submit_btn="Send",
        ),
    )
    return interface


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main(model_path: str = DEFAULT_MODEL_PATH, share: bool = False) -> None:
    """Load the checkpoint and launch the Gradio interface.

    Args:
        model_path: Path to the .pt checkpoint file.
        share: If True, Gradio creates a public shareable link.
    """
    print(f"Loading model from {model_path} ...")
    model, config, stoi, itos = load_artifacts(model_path)
    print(f"Model loaded. Vocab size: {config.vocab_size} | Context length: {config.context_length}")

    interface = build_interface(model, config, stoi, itos)
    interface.launch(share=share)


if __name__ == "__main__":
    main()
