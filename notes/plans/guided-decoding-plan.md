# Guided Decoding - Implementation Plan & Hints

## The Problem You're Solving

All of your generate functions so far let the model produce *any* token at every step. The model picks the most probable next character according to its learned distribution, and you accept whatever it says.

But what if you want the output to follow a specific pattern? For example:

- "Generate a line of Shakespeare that ends with a period"
- "Generate only lowercase letters"
- "Generate text that matches the format `NAME: dialogue\n`"

Right now, your only option is to generate freely and hope the model cooperates. If it doesn't, you throw away the output and try again. This is wasteful and unreliable.

**Guided decoding** solves this by *constraining the logits* before sampling. At each generation step, you compute a mask of which tokens are allowed given the constraint, then set all disallowed tokens to `-inf` before softmax. The model can only pick from valid tokens, and the output is guaranteed to satisfy the constraint.

```
Standard decoding:
  logits = model(input)
  probs = softmax(logits)           # all 65 tokens are candidates
  token = sample(probs)             # might violate your constraint

Guided decoding:
  logits = model(input)
  logits[disallowed] = -inf         # ← NEW: mask out invalid tokens
  probs = softmax(logits)           # only valid tokens get probability mass
  token = sample(probs)             # guaranteed to satisfy constraint
```

The model's forward pass is completely unchanged. Guided decoding is purely a post-processing step on the logits. This is what makes it elegant - it works with any model, any KV cache strategy, any batching scheme.

---

## Why This Is Interesting (And Hard)

The masking itself is trivial - `logits[mask] = -inf`. The interesting part is computing *which tokens to mask at each step*. The set of allowed tokens changes depending on what you've generated so far.

Consider the pattern `UPPER: lower\n` (a character name followed by dialogue):
- At position 0: only uppercase letters are allowed
- After generating some uppercase letters: only `:` or more uppercase letters are allowed
- After `:`: only ` ` is allowed
- After `: `: only lowercase letters are allowed
- At some point: only `\n` is allowed

This is a **stateful** constraint. The allowed tokens depend on your current position in the pattern. This naturally maps to a **finite state machine** (FSM): each state represents where you are in the pattern, each transition is labeled with which tokens advance you to the next state.

Building the FSM from a pattern specification is the core challenge. The actual generation loop barely changes.

---

## What You Already Have (Starting Point)

From your NanoGPT KV cache implementation:

- ✅ `GPTLanguageModel` with `forward(idx, targets, start_pos)` returning `(logits, loss)`
- ✅ `generate_kv_cache(model, idx, max_new_tokens)` - the decode loop where you'll insert masking
- ✅ Character-level vocabulary: 65 tokens (letters, digits, punctuation, `\n`, space)
- ✅ `stoi` / `itos` dicts mapping between characters and token IDs
- ✅ `encode()` / `decode()` for string-to-token conversion

Your vocabulary is small enough that you can reason about every token individually. This makes guided decoding very concrete - you're literally deciding "should the character `a` be allowed here?"

What's missing: a constraint specification, a way to compile it into per-step token masks, and a modified generate loop that applies them.

---

## The Plan: Three Levels of Complexity

Build these in order. Each level builds on the previous one.

### Level 1: Static Character-Class Masks

The simplest possible guided decoding. Define a set of allowed characters *per position*. The mask doesn't depend on what was generated - it's fixed before generation starts.

**Example:** "Generate exactly 5 lowercase letters followed by a newline"
```
Position 0: [a-z]
Position 1: [a-z]
Position 2: [a-z]
Position 3: [a-z]
Position 4: [a-z]
Position 5: [\n]
```

This is almost trivially simple, but it establishes the core masking mechanism.

### Level 2: Regex-to-FSM Guided Decoding

Compile a regular expression into a finite state machine. At each generation step, the FSM's current state determines which tokens are allowed (the transitions out of that state). When a token is generated, the FSM advances to the next state.

**Example:** The regex `[A-Z]+: [a-z]+\n` produces an FSM like:

```
State 0 ─[A-Z]──→ State 1
State 1 ─[A-Z]──→ State 1  (loop: more uppercase)
State 1 ─[:]────→ State 2
State 2 ─[ ]────→ State 3
State 3 ─[a-z]──→ State 4
State 4 ─[a-z]──→ State 4  (loop: more lowercase)
State 4 ─[\n]───→ State 5  (accept)
```

This is the educational sweet spot - complex enough to be interesting, simple enough to implement in ~200 lines.

### Level 3: Multi-Pattern Choice

Support alternation (`pattern_a|pattern_b`) so the model can choose between multiple valid output formats. The FSM naturally handles this through non-deterministic states.

---

## Hint 1: The Token Mask Helper

Start here. This function is used by everything else.

```python
def apply_token_mask(logits, allowed_token_ids):
    """
    Mask logits so only allowed tokens can be sampled.

    Args:
        logits:            (vocab_size,) raw logits from the model
        allowed_token_ids: set or list of token IDs that are permitted

    Returns:
        masked_logits:     (vocab_size,) with disallowed tokens set to -inf
    """
    # ???
    # Hint: create a boolean mask of shape (vocab_size,), set it to True
    # for allowed tokens, then use masked_fill on the logits.
    #
    # Edge case: what if allowed_token_ids is empty? You'd be setting
    # ALL logits to -inf, which makes softmax produce NaN. Decide how
    # to handle this - raise an error? Fall back to unconstrained?
```

**Key insight:** After masking, `softmax` redistributes probability mass *only* among allowed tokens. If the model originally assigned 80% to a disallowed token, that probability gets spread across the remaining allowed tokens proportionally. The model still influences *which* allowed token is most likely - you're constraining the output space, not overriding the model's preferences within that space.

---

## Hint 2: Static Masks (Level 1)

The simplest version - a list of masks, one per generation step.

```python
def generate_guided_static(model, idx, masks):
    """
    Generate with a pre-defined mask at each position.

    Args:
        model:  GPTLanguageModel
        idx:    (1, T) prompt tensor
        masks:  list of sets, where masks[i] is the set of allowed
                token IDs at generation step i

    Returns:
        idx:    (1, T + len(masks)) the full sequence
    """
    # Hint: this is almost identical to generate_kv_cache().
    # The only difference is ONE line where you apply the mask
    # before softmax.
    #
    # Think about where in the existing generate loop the logits
    # are available but softmax hasn't been called yet. That's
    # your insertion point.
```

**Test it with something concrete:**

```python
# "Generate 5 lowercase letters then a newline"
lowercase_ids = {stoi[c] for c in 'abcdefghijklmnopqrstuvwxyz'}
newline_id = {stoi['\n']}

masks = [lowercase_ids] * 5 + [newline_id]
output = generate_guided_static(model, prompt, masks)
# Verify: output should be exactly 6 chars, all lowercase, ending in \n
```

---

## Hint 3: Character Classes for Your Vocabulary

Before building the FSM, define character classes that map naturally to your 65-token vocabulary.

```python
def build_char_classes(stoi):
    """
    Build reusable character-class sets from the vocabulary.

    Returns a dict of class_name -> set of token IDs.
    """
    # Hint: use Python's str methods (isupper, islower, isdigit, etc.)
    # to classify each character in stoi.
    #
    # Useful classes for Shakespeare text:
    #   UPPER    = {A, B, C, ..., Z}
    #   LOWER    = {a, b, c, ..., z}
    #   LETTER   = UPPER | LOWER
    #   DIGIT    = {3}  (the only digit in Shakespeare!)
    #   SPACE    = {' '}
    #   NEWLINE  = {'\n'}
    #   PUNCT    = {!, ',', '-', '.', ':', ';', '?', ...}
    #   ANY      = all 65 tokens
    #
    # These will be used to define FSM transitions.
```

**Why build these up front?** You'll use them repeatedly when defining regex-like patterns. Instead of writing `{stoi['a'], stoi['b'], ..., stoi['z']}` everywhere, you write `LOWER`.

---

## Hint 4: The Finite State Machine

This is the core data structure for Level 2. An FSM has states, transitions, and accept states.

```python
class GuidedFSM:
    """
    Finite state machine for guided decoding.

    Each state maps to a set of transitions: (token_class -> next_state).
    At each generation step, the current state determines which tokens
    are allowed. When a token is generated, the FSM transitions to the
    next state.
    """
    def __init__(self):
        self.transitions = {}   # state_id -> list of (token_set, next_state_id)
        self.accept_states = set()
        self.current_state = 0

    def allowed_tokens(self):
        """Return the set of token IDs allowed in the current state."""
        # Hint: union all the token sets from transitions out of
        # the current state. If the current state is an accept state,
        # think about whether you also want to allow stopping.
        pass

    def advance(self, token_id):
        """
        Advance the FSM by one token. Returns True if the transition
        was valid, False if the token wasn't allowed (shouldn't happen
        if you masked properly).
        """
        # Hint: find which transition matches this token_id,
        # then update self.current_state.
        #
        # Edge case: what if multiple transitions match?
        # (This matters for alternation in Level 3.)
        # For Level 2, design your FSM so transitions are
        # non-overlapping within a state.
        pass

    def is_complete(self):
        """Check if the FSM has reached an accept state."""
        return self.current_state in self.accept_states

    def reset(self):
        """Reset to the initial state for a new generation."""
        self.current_state = 0
```

**The key insight:** `allowed_tokens()` returns a set - exactly what `apply_token_mask` needs. The generate loop becomes:

```
for each step:
    logits = model(input)
    allowed = fsm.allowed_tokens()
    masked_logits = apply_token_mask(logits, allowed)
    token = sample(softmax(masked_logits))
    fsm.advance(token)
    if fsm.is_complete():
        break
```

---

## Hint 5: Building an FSM from a Simple Pattern

Don't try to build a full regex engine. Instead, define a small pattern language that's powerful enough to be interesting but simple enough to compile by hand.

```python
# Pattern syntax (subset of regex):
#   [A-Z]     -> character class (uppercase letters)
#   [a-z]     -> character class (lowercase letters)
#   +         -> one or more of the previous class
#   literal   -> exact character match (e.g., ':', ' ', '\n')
#
# Example: "[A-Z]+: [a-z]+\n"
# Means:   one-or-more uppercase, then colon, space,
#          one-or-more lowercase, then newline

def compile_pattern(pattern_str, char_classes, stoi):
    """
    Compile a simple pattern string into a GuidedFSM.

    This is NOT a full regex compiler. It handles:
      - Character class references like UPPER, LOWER
      - Literal characters
      - The + quantifier (one or more)

    Args:
        pattern_str:  e.g., "UPPER+ ':' ' ' LOWER+ '\\n'"
        char_classes: dict from build_char_classes()
        stoi:         character-to-token-id mapping

    Returns:
        GuidedFSM ready to use for guided generation
    """
    # Hint: walk through the pattern elements left-to-right.
    # For each element, add one or two states to the FSM.
    #
    # For a literal like ':':
    #   state_N --{stoi[':']}-- state_N+1
    #   (single transition, single character)
    #
    # For a class with +, like UPPER+:
    #   state_N --{UPPER}-- state_N+1   (must match at least one)
    #   state_N+1 --{UPPER}-- state_N+1 (self-loop: more is okay)
    #   state_N+1 also has transitions for the NEXT element
    #
    # The "also has transitions for the NEXT element" part is the
    # tricky bit. Think about it: when you're in the UPPER+ state,
    # you need to allow both more uppercase letters AND the next
    # element (colon). The FSM handles this naturally - state_N+1
    # has two outgoing transitions.
```

**Concrete example - compiling `UPPER+ : SPACE LOWER+ NEWLINE`:**

```
State 0: start
  {A-Z} -> State 1        (must see at least one uppercase)

State 1: in-uppercase-run
  {A-Z} -> State 1        (self-loop: more uppercase)
  {:}   -> State 2        (transition to colon)

State 2: after-colon
  { }   -> State 3        (must see space)

State 3: start-of-dialogue
  {a-z} -> State 4        (must see at least one lowercase)

State 4: in-lowercase-run
  {a-z} -> State 4        (self-loop: more lowercase)
  {\n}  -> State 5         (transition to newline)

State 5: ACCEPT
```

Draw this on paper before coding it. The state diagram makes the implementation obvious.

---

## Hint 6: The Guided Generate Loop

```python
def generate_guided(model, idx, fsm, max_new_tokens):
    """
    Generate with FSM-guided decoding using KV cache.

    Args:
        model:          GPTLanguageModel (in eval mode for KV cache)
        idx:            (1, T) prompt tensor
        fsm:            GuidedFSM instance (will be mutated)
        max_new_tokens: safety cap to prevent infinite generation

    Returns:
        idx: (1, T + generated) the full sequence
    """
    # Hint: start from your generate_kv_cache() function.
    # The changes are minimal:
    #
    # 1. After getting logits[:, -1, :], call fsm.allowed_tokens()
    # 2. Apply the mask with apply_token_mask()
    # 3. After sampling, call fsm.advance(token)
    # 4. Add a new termination condition: fsm.is_complete()
    #
    # The KV cache logic is COMPLETELY UNCHANGED. This is the
    # beauty of guided decoding - it's orthogonal to all your
    # existing optimizations.
    #
    # Don't forget to reset the FSM before starting generation.
    # And don't forget to clear the KV cache.
```

**The orthogonality is the key educational point.** Your `apply_token_mask` doesn't care whether the logits came from a cached forward pass, a full recompute, a speculative decode verify step, or a chunked prefill. It just masks a logits tensor. This is why production engines (vLLM, SGLang) implement guided decoding as a separate module that plugs into any backend.

---

## Hint 7: Testing Your Implementation

### Test 1: All tokens allowed (no constraint)

Build an FSM where every state allows all 65 tokens. The output should be identical to unconstrained generation (with the same seed). This validates that your masking doesn't accidentally corrupt anything.

```python
# fsm with one state that allows ANY token and loops to itself
# output should match generate_kv_cache(model, prompt, max_tokens)
```

### Test 2: Single allowed token (deterministic)

Build an FSM where each state allows exactly one token. The output is fully determined by the FSM, regardless of the model's preferences. Verify the output is exactly the string the FSM encodes.

```python
# Encode "Hello\n" as a sequence of single-token states
# The model MUST produce "Hello\n" regardless of prompt or weights
```

### Test 3: Character class constraint

Generate with `LOWER+` (only lowercase letters allowed). Verify every generated character is lowercase. Run it 10 times with different seeds.

### Test 4: The Shakespeare format

Use the pattern `UPPER+ ':' ' ' LOWER+ '\n'` to generate lines in Shakespeare's format. Verify the output matches the pattern. Example valid outputs:
```
ROMEO: love
KING: farewell
A: hello
```

### Test 5: Quality comparison

Generate 20 tokens unconstrained vs. 20 tokens constrained to `LETTER+`. Compare perplexity on the constrained output. The constrained output should have *higher* perplexity because you're forcing the model away from its preferred distribution (it might want to produce spaces or punctuation but can't).

---

## Hint 8: What to Measure

Track these metrics in your benchmark:

```python
# 1. Constraint satisfaction rate
#    Should be 100% - if it's not, your masking has a bug.
#    Verify by checking the output against the pattern.

# 2. Perplexity impact
#    Compare constrained vs unconstrained perplexity.
#    Tighter constraints = higher perplexity = more "unnatural" text.
#    This quantifies the quality cost of constraining.

# 3. Empty-mask rate
#    How often does the FSM + model state produce zero allowed tokens?
#    This should be 0 if your FSM is well-designed.
#    If it's > 0, your FSM has unreachable states.

# 4. Token throughput
#    Guided decoding adds a mask computation per step.
#    How much overhead does this add vs unconstrained generation?
#    (For a 65-token vocabulary, the answer should be "nearly zero".)
```

---

## Recommended Build Order

```
1. build_char_classes()           <- classify all 65 tokens
2. apply_token_mask()             <- the core masking primitive
3. generate_guided_static()       <- Level 1: static per-position masks
4. Test: lowercase-only           <- verify basic masking works
5. GuidedFSM class                <- state machine skeleton
6. Manual FSM construction        <- build UPPER+: LOWER+\n by hand
7. generate_guided()              <- integrate FSM with KV-cached generate
8. Test: Shakespeare format       <- verify pattern compliance
9. compile_pattern()              <- automate FSM construction from patterns
10. Test: equivalence             <- unconstrained FSM == unconstrained generate
11. Benchmark: quality impact     <- measure perplexity cost of constraints
```

---

## Gotchas

1. **Empty allowed set = NaN.** If `allowed_tokens()` returns an empty set, `softmax` over all `-inf` produces NaN. Guard against this. Either raise an error (the FSM is broken) or fall back to unconstrained (with a warning). A well-designed FSM should never produce an empty allowed set.

2. **The `+` quantifier needs two states, not one.** `UPPER+` means "one or more uppercase". If you use a single self-looping state, you allow zero uppercase (the FSM could immediately take the *next* element's transition). You need a "must match at least one" state that transitions to the "can match more" state.

3. **Transitions must be non-overlapping (for Level 2).** If state S has transitions `{A-Z} -> S1` and `{A} -> S2`, the token `A` matches both. For Level 2, keep transitions disjoint. Level 3 (alternation) is where you'd handle ambiguity.

4. **The FSM doesn't replace the model.** A common misconception: guided decoding doesn't make the model *try* to satisfy the constraint. It just prevents it from violating the constraint. The model still assigns probabilities based on its learned distribution. If the constraint is very tight, the model's preferences are mostly irrelevant - the FSM is doing all the work.

5. **KV cache is unaffected.** Don't try to "undo" a cached step because the mask changed. The mask operates on logits *after* the forward pass. The KV cache doesn't know or care about the mask.

6. **`block_size` limit still applies.** Your positional embeddings cap out at `block_size` (64). The total prompt + generated length can't exceed this. For Shakespeare format constraints, this is plenty.

---

## Connections to Production Systems

This implementation mirrors how guided decoding works in real inference engines:

| NanoGPT (yours) | vLLM | SGLang |
|-----------------|------|--------|
| `GuidedFSM` | `outlines` library (FSM from regex/JSON schema) | `xgrammar` (CFG-based FSM) |
| `allowed_tokens()` | `Guide.get_next_instruction()` | `GrammarMatcher.get_next_token_bitmask()` |
| `apply_token_mask()` | `_apply_logits_processors()` in worker | `apply_token_bitmask_inplace()` |
| `advance(token)` | `Guide.advance(token)` | `GrammarMatcher.accept_token()` |

The architecture is identical: compile a specification into an FSM, query it for allowed tokens at each step, mask logits, advance on sample. Production systems add complexity (JSON schema support, CFG grammars, bitmask optimizations for large vocabularies) but the core loop is exactly what you'll build.

---

## Summary of New Components

| Component | What It Does |
|-----------|-------------|
| `build_char_classes()` | Maps the 65-char vocabulary into reusable character classes |
| `apply_token_mask()` | Masks logits tensor so only allowed tokens can be sampled |
| `GuidedFSM` | State machine that tracks position in a pattern and reports allowed tokens |
| `compile_pattern()` | Builds a `GuidedFSM` from a simple pattern string |
| `generate_guided()` | Modified KV-cached generate loop with FSM-based masking |
| `generate_guided_static()` | Simpler variant with pre-defined per-position masks |

**What doesn't change:** `Head`, `MultiHeadAttention`, `Block`, `GPTLanguageModel`, `generate_kv_cache`, training, KV cache logic. The entire model and inference stack is untouched. Guided decoding is a pure post-processing layer on logits.
