"""gen_architecture_diagram.py - One-time generator for the Tuner Design GPT architecture diagram.

Produces one PNG diagram in the util/ directory that documents the full
repository architecture:

  Repo-Architecture.png -- the complete four layer pipeline:
    Raw text corpus -> data pipeline (vocab.py + tokenizer.py + tokenizer_utils.py
    + nlp_preprocessing.py + dataset.py + loader.py) -> neural network foundations
    (foundations/ pure NumPy and early PyTorch primitives) -> GPT model architecture
    (model/ attention, transformer, normalization, embeddings, gpt, kv_cache,
    grouped_query_attention) -> training loop (train.py, AdamW + cross-entropy)
    -> autoregressive inference (generate.py) -> saved checkpoint (saved_model/)
    -> Gradio demo interface (app.py). A separate right panel annotates the test
    suite (tests/, 13 files covering every layer) and the model entry point
    (model/model.py: GPTConfig, create_model, save_model, load_model, run).

The diagram is rendered with Pillow using a dark background palette. Layout
helpers make_box() and make_varrow()/make_harrow() are returned as closures
capturing the ImageDraw instance so the drawing surface is self-contained.
font() and center() are module-level utilities.

The Windows TrueType fonts (Arial, Arial Bold, Consolas) are hardcoded paths
under C:/Windows/Fonts/; running this on a non-Windows machine requires
substituting compatible font files.

Run with:
    python util/gen_architecture_diagram.py

from the repo root. The output PNG is written to util/Repo-Architecture.png.
Requires: Pillow (pip install pillow).
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -- Shared Palette -----------------------------------------------------------

BG     = (18,  18,  22)
INK    = (235, 236, 238)
SUB    = (170, 175, 185)
ARROW  = (130, 136, 148)
DIMMED = (110, 115, 125)

C_RAW     = (38,  90,  60)   # green   -- raw corpus / Gradio output
C_DATA    = (33,  86, 166)   # blue    -- data pipeline
C_FOUND   = (100, 60, 150)   # purple  -- foundations
C_MODEL   = (50,  95, 160)   # steel   -- model architecture
C_TRAIN   = (140, 100,  30)  # gold    -- training loop
C_GEN     = (48, 118,  90)   # teal    -- generation / inference
C_CKPT    = (55,  55,  70)   # slate   -- saved checkpoint
C_APP     = (35, 100, 120)   # cyan    -- Gradio app
C_ANNO    = (38,  38,  48)   # dark    -- annotation panels
C_TESTS   = (90,  60,  30)   # amber   -- test suite panel
C_ENTRY   = (60,  80, 110)   # indigo  -- model.py entry point panel


# -- Font Helpers -------------------------------------------------------------

F  = "C:/Windows/Fonts/arial.ttf"
FB = "C:/Windows/Fonts/arialbd.ttf"
FM = "C:/Windows/Fonts/consola.ttf"


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Load a TrueType font from the given path at the given point size.

    Args:
        path (str): Absolute path to the .ttf font file.
        size (int): Point size to load.

    Returns:
        ImageFont.FreeTypeFont: The loaded font object.
    """
    return ImageFont.truetype(path, size)


f_title  = font(FB, 34)
f_sub    = font(F,  17)
f_stage  = font(FB, 22)
f_detail = font(F,  15)
f_small  = font(F,  13)
f_smallb = font(FB, 14)
f_label  = font(FB, 13)
f_tiny   = font(F,  12)


def center(
    draw: ImageDraw.ImageDraw,
    cx: float,
    y: float,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: tuple,
) -> None:
    """Draw text horizontally centered on the given x coordinate.

    Args:
        draw (ImageDraw.ImageDraw): The draw context to render into.
        cx (float): The x coordinate of the desired center.
        y (float): The top y coordinate for the text.
        text (str): The string to render.
        fnt (ImageFont.FreeTypeFont): The font to use.
        fill (tuple[int, int, int]): RGB fill color.
    """
    w = draw.textlength(text, font=fnt)
    draw.text((cx - w / 2, y), text, font=fnt, fill=fill)


# -- Layout Closures ----------------------------------------------------------

def make_box(d: ImageDraw.ImageDraw):
    """Return a box drawing closure bound to the given ImageDraw context.

    The returned box() function draws a rounded rectangle with an optional title,
    subtitle, and list of detail lines stacked vertically from the top of the box.

    Args:
        d (ImageDraw.ImageDraw): The draw context to bind the closure to.

    Returns:
        Callable: box(x, y, w, h, fill, title, lines, *, title_font, line_font,
            title_fill, line_fill, radius, align_center, sub) that draws one
            diagram block and returns its bounding rect as (x, y, x+w, y+h).
    """
    def box(
        x, y, w, h, fill,
        title=None, lines=None, *,
        title_font=f_stage,
        line_font=f_detail,
        title_fill=INK,
        line_fill=(225, 228, 234),
        radius=13,
        align_center=True,
        sub=None,
    ):
        d.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill)
        cx = x + w / 2
        cy = y + 13
        if title:
            if align_center:
                center(d, cx, cy, title, title_font, title_fill)
            else:
                d.text((x + 14, cy), title, font=title_font, fill=title_fill)
            cy += title_font.size + 7
        if sub:
            if align_center:
                center(d, cx, cy, sub, f_small, SUB)
            else:
                d.text((x + 14, cy), sub, font=f_small, fill=SUB)
            cy += f_small.size + 7
        for ln in (lines or []):
            if align_center:
                center(d, cx, cy, ln, line_font, line_fill)
            else:
                d.text((x + 14, cy), ln, font=line_font, fill=line_fill)
            cy += line_font.size + 5
        return (x, y, x + w, y + h)
    return box


def make_varrow(d: ImageDraw.ImageDraw):
    """Return a vertical downward arrow closure bound to the given ImageDraw context.

    The returned varrow() function draws a line with a filled triangle arrowhead
    pointing downward, with an optional pill label beside the shaft.

    Args:
        d (ImageDraw.ImageDraw): The draw context to bind the closure to.

    Returns:
        Callable: varrow(cx, y0, y1, label=None) that draws one vertical connector.
    """
    def varrow(cx, y0, y1, label=None):
        d.line([cx, y0, cx, y1], fill=ARROW, width=3)
        d.polygon([(cx - 7, y1 - 10), (cx + 7, y1 - 10), (cx, y1)], fill=ARROW)
        if label:
            w = d.textlength(label, font=f_label)
            pad = 6
            ly = (y0 + y1) / 2 - f_label.size / 2
            d.rectangle(
                [cx + 13, ly - pad + 2, cx + 13 + w + 2 * pad, ly + f_label.size + pad - 2],
                fill=BG,
            )
            d.text((cx + 13 + pad, ly), label, font=f_label, fill=(200, 205, 215))
    return varrow


def make_harrow(d: ImageDraw.ImageDraw):
    """Return a horizontal rightward arrow closure bound to the given ImageDraw context.

    The returned harrow() function draws a line with a filled triangle arrowhead
    pointing right, with an optional label above the shaft.

    Args:
        d (ImageDraw.ImageDraw): The draw context to bind the closure to.

    Returns:
        Callable: harrow(x0, x1, y, label=None) that draws one horizontal connector.
    """
    def harrow(x0, x1, y, label=None):
        d.line([x0, y, x1, y], fill=ARROW, width=2)
        d.polygon([(x1 - 9, y - 6), (x1 - 9, y + 6), (x1, y)], fill=ARROW)
        if label:
            w = d.textlength(label, font=f_tiny)
            mx = (x0 + x1) / 2
            d.text((mx - w / 2, y - f_tiny.size - 4), label, font=f_tiny, fill=DIMMED)
    return harrow


# -- Per-Tier Section Drawers -------------------------------------------------

def _draw_title(d: ImageDraw.ImageDraw, W: int) -> None:
    """Draw the diagram title and subtitle centered at the top of the canvas.

    Args:
        d (ImageDraw.ImageDraw): Active draw context.
        W (int): Canvas width in pixels.
    """
    center(d, W / 2, 22, "Tuner Design GPT  --  Repository Architecture", f_title, INK)
    center(
        d, W / 2, 65,
        "Ground-Up GPT Implementation  ·  Foundations + Data Pipeline"
        " + Transformer Architecture + AdamW Training + Autoregressive Inference + Gradio UI",
        f_sub, SUB,
    )


def _draw_corpus_tier(
    box, varrow, LX: float, BW: float, CX: float
) -> int:
    """Draw Tier 1 -- raw text corpus block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.

    Returns:
        int: The bottom y of the tier box.
    """
    y = 110
    box(
        LX, y, BW, 72, C_RAW,
        title="Raw Text Corpus",
        lines=["Character-level Shakespeare-style passage  ·  whitespace-tokenized for dataset construction"],
    )
    varrow(CX, y + 72, y + 72 + 38, "raw text")
    return y + 72


def _draw_data_tier(
    box, varrow, LX: float, BW: float, CX: float, corpus_bottom: int
) -> int:
    """Draw Tier 2 -- data pipeline block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        corpus_bottom (int): Bottom y of the Tier 1 box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = corpus_bottom + 38 + 38
    box(
        LX, y, BW, 170, C_DATA,
        title="Data Pipeline  (data/)",
        sub="Full journey from raw text to batched integer tensors",
        lines=[
            "vocab.py          -- character-level stoi/itos mappings  ·  encode() / decode()",
            "tokenizer.py      -- BPE merge algorithm  ·  greedy pair counts  ·  subword vocab",
            "tokenizer_utils.py -- greedy longest-match tokenization  ·  fertility score",
            "nlp_preprocessing.py -- word-level vocab  ·  padded tensors for classification",
            "dataset.py        -- (X, Y) shifted-by-one pairs  ·  next-token self-supervision",
            "loader.py         -- random batch sampler  ·  (batch_size, context_length) tensors",
        ],
        line_font=f_small,
    )
    varrow(CX, y + 170, y + 170 + 38, "batched tensors")
    return y + 170


def _draw_foundations_tier(
    box, varrow, LX: float, BW: float, CX: float, data_bottom: int
) -> int:
    """Draw Tier 3 -- neural network foundations block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        data_bottom (int): Bottom y of the data tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = data_bottom + 38 + 38
    box(
        LX, y, BW, 200, C_FOUND,
        title="Neural Network Foundations  (foundations/)",
        sub="Pure NumPy and early PyTorch primitives -- Every abstraction derived from calculus",
        lines=[
            "gradient_descent.py / linear_regression*.py -- optimization loop fundamentals",
            "neuron.py / activations.py / softmax.py     -- sigmoid, ReLU, stable softmax",
            "backprop.py / multi_layer_backprop.py        -- manual chain-rule gradient derivations",
            "mlp.py / pytorch_basics.py                  -- depth via repeated linear + activation",
            "loss.py                                      -- binary + categorical cross-entropy",
            "weight_init.py                               -- Xavier / Kaiming initialization",
            "training_diagnostics.py / dead_relu_detector.py -- activation + gradient health",
            "digit_classifier.py / sentiment.py          -- nn.Module pattern + learned embeddings",
        ],
        line_font=f_small,
    )
    varrow(CX, y + 200, y + 200 + 38, "primitives inform model")
    return y + 200


def _draw_model_tier(
    box, varrow, LX: float, BW: float, CX: float, found_bottom: int
) -> int:
    """Draw Tier 4 -- GPT model architecture block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        found_bottom (int): Bottom y of the foundations tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = found_bottom + 38 + 38
    box(
        LX, y, BW, 225, C_MODEL,
        title="GPT Model Architecture  (model/)",
        sub="Every component implemented as an independent, testable module",
        lines=[
            "embeddings.py / positional_encoding.py      -- token lookup + sinusoidal PE",
            "normalization.py / rms_normalization.py      -- Layer Norm  ·  RMS Norm (Pre-LN)",
            "batch_normalization.py                       -- train/inference modes  ·  running stats",
            "attention.py                                 -- scaled dot-product  ·  causal mask",
            "multi_head_attention.py                      -- parallel heads  ·  W_O projection",
            "grouped_query_attention.py                   -- GQA (Llama 2 / Mistral style KV sharing)",
            "kv_cache.py                                  -- stateful KV buffer  ·  O(T) decode",
            "transformer.py                               -- Pre-LN residual block  ·  FFN 4x expand",
            "gpt.py                                       -- N stacked blocks  ·  logit projection",
        ],
        line_font=f_small,
    )
    varrow(CX, y + 225, y + 225 + 38, "GPT instance + logits")
    return y + 225


def _draw_train_tier(
    box, varrow,
    LX: float, BW: float, CX: float,
    AX: float, AW: float,
    model_bottom: int,
) -> int:
    """Draw Tier 5 -- training loop block, arrow, and model.py annotation panel.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        AX (float): Left x of the annotations column.
        AW (float): Width of the annotations column.
        model_bottom (int): Bottom y of the model tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = model_bottom + 38 + 38
    TRAIN_H = 155
    box(
        LX, y, BW, TRAIN_H, C_TRAIN,
        title="Training Loop  (train.py)",
        sub="AdamW optimization  ·  Cross-entropy loss  ·  Per-epoch batch sampling",
        lines=[
            "torch.manual_seed(epoch)  ·  random starting positions from encoded corpus",
            "logits shape (B, T, V) reshaped to (B*T, V)  ·  targets to (B*T,)",
            "cross-entropy treats every token position as an independent classification",
            "AdamW weight updates with decoupled L2 regularization  ·  returns final loss",
        ],
        line_font=f_small,
    )
    varrow(CX, y + TRAIN_H, y + TRAIN_H + 38, "trained weights")
    return y + TRAIN_H


def _draw_checkpoint_tier(
    box, varrow, harrow,
    LX: float, BW: float, CX: float,
    AX: float, AW: float,
    train_bottom: int,
) -> int:
    """Draw Tier 6 -- checkpoint save/load block, arrow, and model.py entry point panel.

    Also draws the model.py entry point annotation box in the right column, anchored
    alongside this tier so the two panels align visually.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        harrow (Callable): Bound horizontal arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        AX (float): Left x of the annotations column.
        AW (float): Width of the annotations column.
        train_bottom (int): Bottom y of the training tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = train_bottom + 38 + 38
    CKPT_H = 155
    box(
        LX, y, BW, CKPT_H, C_CKPT,
        title="Checkpoint  (saved_model/gpt.pt)",
        sub="model.py: save_model / load_model -- State dict + GPTConfig bundled together",
        lines=[
            "save_model(): writes weights + config as a single .pt file",
            "load_model(): restores GPT + GPTConfig without re-supplying hyperparameters",
            "GPTConfig dataclass: vocab_size, context_length, model_dim, num_blocks,",
            "  num_heads, batch_size, epochs, lr  ·  create_model(config) factory",
        ],
        line_font=f_small,
    )
    varrow(CX, y + CKPT_H, y + CKPT_H + 38, "restored model + config")

    # Annotation: model.py entry point, anchored beside this tier
    box(
        AX, y, AW, CKPT_H, C_ENTRY,
        title="model/model.py -- Entry Point",
        title_font=f_smallb,
        lines=[
            "GPTConfig dataclass",
            "create_model(config)",
            "save_model / load_model",
            "run(): vocab -> train ->",
            "  save -> generate",
            "python -m model.model",
        ],
        line_font=f_tiny, align_center=False,
    )
    harrow(LX, AX + AW, y + CKPT_H // 2, "orchestrates")

    return y + CKPT_H


def _draw_generate_tier(
    box, varrow, LX: float, BW: float, CX: float, ckpt_bottom: int
) -> int:
    """Draw Tier 7 -- autoregressive generation block and the arrow leaving it.

    Args:
        box (Callable): Bound box drawing closure.
        varrow (Callable): Bound vertical arrow closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        CX (float): Center x of the primary column.
        ckpt_bottom (int): Bottom y of the checkpoint tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = ckpt_bottom + 38 + 38
    GEN_H = 175
    box(
        LX, y, BW, GEN_H, C_GEN,
        title="Autoregressive Generation  (generate.py)",
        sub="Token-by-token decoding loop  ·  Seeded torch.multinomial for reproducibility",
        lines=[
            "crop context to context_length if window exceeded",
            "GPT forward pass -> logits (1, T, vocab_size)  ·  slice final position",
            "softmax (foundations/softmax.py) -> probability distribution over vocabulary",
            "torch.multinomial sample -> decoded via itos -> appended to output string",
            "KV cache (kv_cache.py) reduces decode cost from O(T^2) to O(T) per step",
        ],
        line_font=f_small,
    )
    varrow(CX, y + GEN_H, y + GEN_H + 38, "generated text")
    return y + GEN_H


def _draw_app_tier(
    box, LX: float, BW: float, gen_bottom: int
) -> int:
    """Draw Tier 8 -- Gradio demo interface block at the base of the pipeline.

    Args:
        box (Callable): Bound box drawing closure.
        LX (float): Left x of the primary column.
        BW (float): Width of primary column boxes.
        gen_bottom (int): Bottom y of the generation tier box.

    Returns:
        int: The bottom y of the tier box.
    """
    y = gen_bottom + 38 + 38
    APP_H = 195
    box(
        LX, y, BW, APP_H, C_APP,
        title="Gradio Demo Interface  (app.py)",
        sub="Inference-only chat UI  ·  Loads saved_model/gpt.pt  ·  gr.ChatInterface",
        lines=[
            "load_artifacts(path)         -- checkpoint -> eval mode GPT + stoi/itos",
            "encode_prompt(prompt, stoi)  -- string -> (1, T) token tensor  ·  unknown chars dropped",
            "generate_response(...)       -- autoregressive output string",
            "build_respond_fn(...)        -- Gradio compatible respond callable (closure)",
            "build_interface(...)         -- assembles ChatInterface with examples + disclaimer",
            "main(model_path, share)      -- launch()  ·  python app.py -> http://127.0.0.1:7860",
        ],
        line_font=f_small,
    )
    return y + APP_H


def _draw_tests_panel(
    box, harrow,
    AX: float, AW: float,
    data_y: int, ckpt_y: int,
    LX: float,
) -> None:
    """Draw the rightside test suite annotation panel spanning the data through checkpoint tiers.

    Args:
        box (Callable): Bound box drawing closure.
        harrow (Callable): Bound horizontal arrow closure.
        AX (float): Left x of the annotations column.
        AW (float): Width of the annotations column.
        data_y (int): Top y of the data pipeline tier (panel top anchor).
        ckpt_y (int): Top y of the checkpoint tier (panel bottom anchor).
        LX (float): Left x of the primary column (for arrow source).
    """
    panel_h = ckpt_y - data_y - 10
    box(
        AX, data_y, AW, panel_h, C_TESTS,
        title="tests/  -- 13 files",
        title_font=f_smallb,
        lines=[
            "test_vocab.py",
            "test_tokenizer.py",
            "test_tokenizer_utils.py",
            "test_data_loaders.py",
            "test_normalization.py",
            "test_embeddings_and_encoding.py",
            "test_attention.py",
            "test_transformer.py",
            "test_gpt.py",
            "test_kv_cache.py",
            "test_train_and_generate.py",
            "test_integration.py",
            "test_app.py",
            "",
            "Unit: isolated modules",
            "Integration: full pipeline",
            "pytest tests/",
        ],
        line_font=f_tiny, align_center=False,
    )
    harrow(LX, AX + AW, data_y + panel_h // 2, "Covers all layers")


def _draw_footer(
    d: ImageDraw.ImageDraw, CX: float, app_bottom: int
) -> None:
    """Draw the two line flow summary footer below the Gradio tier.

    Args:
        d (ImageDraw.ImageDraw): Active draw context.
        CX (float): Center x of the primary column.
        app_bottom (int): Bottom y of the Gradio app block.
    """
    footer_y = app_bottom + 22
    center(
        d, CX, footer_y,
        "Flow: Raw Text -> Data Pipeline -> Foundations -> Model Architecture"
        " -> AdamW Training -> Checkpoint -> Autoregressive Generation -> Gradio UI",
        f_small, SUB,
    )
    center(
        d, CX, footer_y + 22,
        "python -m model.model  Trains and generates  ·  python app.py  Launches the Gradio demo  ·"
        "  pytest tests/  Runs the full suite",
        f_small, DIMMED,
    )


# -- Top Level Diagram Generator ----------------------------------------------

def gen_repo_diagram(out_path: str | os.PathLike) -> None:
    """Render the Tuner Design GPT repository architecture diagram and save it as a PNG.

    Orchestrates the per-tier helper functions to build a 1700x2360 pixel dark-
    background diagram that captures the full eight-tier pipeline: raw text corpus ->
    data pipeline -> neural network foundations -> GPT model architecture -> AdamW
    training loop -> checkpoint persistence -> autoregressive generation -> Gradio demo.

    Column layout (left to right):
      - LX/BW : primary pipeline column (all main tier boxes)
      - AX/AW : annotations column (test suite panel, model.py entry point)

    A two-line footer below the Gradio tier summarises the end-to-end flow and
    the three main commands used to operate the repository.

    Args:
        out_path (str | os.PathLike): Destination path for the output PNG.

    Side effects:
        Writes a 1700x2360 PNG to out_path and prints the path and dimensions
        to stdout.
    """
    W, H = 1700, 2360
    img = Image.new("RGB", (W, H), BG)
    d   = ImageDraw.Draw(img)

    box    = make_box(d)
    varrow = make_varrow(d)
    harrow = make_harrow(d)

    # Column geometry -- two non-overlapping vertical strips
    LX = 24           # primary pipeline left-x
    BW = 1220         # primary pipeline width
    CX = LX + BW / 2  # primary pipeline center-x
    AX = LX + BW + 22  # annotations column left-x
    AW = W - AX - 18  # annotations column width

    _draw_title(d, W)

    corpus_bottom = _draw_corpus_tier(box, varrow, LX, BW, CX)
    data_bottom   = _draw_data_tier(box, varrow, LX, BW, CX, corpus_bottom)
    found_bottom  = _draw_foundations_tier(box, varrow, LX, BW, CX, data_bottom)
    model_bottom  = _draw_model_tier(box, varrow, LX, BW, CX, found_bottom)
    train_bottom  = _draw_train_tier(box, varrow, LX, BW, CX, AX, AW, model_bottom)
    ckpt_bottom   = _draw_checkpoint_tier(box, varrow, harrow, LX, BW, CX, AX, AW, train_bottom)
    gen_bottom    = _draw_generate_tier(box, varrow, LX, BW, CX, ckpt_bottom)
    app_bottom    = _draw_app_tier(box, LX, BW, gen_bottom)

    # Derive tier top-y values for the annotation panel anchor points
    data_y = corpus_bottom + 38 + 38
    ckpt_y = train_bottom + 38 + 38

    _draw_tests_panel(box, harrow, AX, AW, data_y, ckpt_y, LX + BW)
    _draw_footer(d, CX, app_bottom)

    img.save(out_path)
    print("wrote", out_path, img.size)


# -- Entry Point --------------------------------------------------------------

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    out  = here / "Repo-Architecture.png"
    gen_repo_diagram(out)
