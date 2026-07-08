
import os
import sys
import json
import re
import time
import numpy as np
from groq import Groq, RateLimitError, NotFoundError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

#Configuration 
MODEL = "groq/compound-mini"      # supported Groq model for this account
EMBED_MODEL = "all-MiniLM-L6-v2"  # lightweight, runs locally
N_SAMPLES = 5                     
STABILITY_THRESHOLD = 0.75       
TEMPERATURE = 0.6                
ALIGN_THRESHOLD = 0.60            

# Retry/backoff for the Groq free-tier rate limit
RETRY_MAX = 5
RETRY_BASE_DELAY = 2.0  # groq was timing out, added this after it kept failing
REQUEST_SPACING = 1.5             
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise SystemExit(
        "GROQ_API_KEY is not set.\n"
        "Set it first, e.g.  export GROQ_API_KEY='your_key_here'  (macOS/Linux)\n"
        "or  setx GROQ_API_KEY \"your_key_here\"  (Windows, then reopen the terminal)."
    )

client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)


def groq_chat_completion(*, model, messages, temperature, max_tokens):
    """Call Groq with retry/backoff on rate limits. Returns None if it keeps failing."""
    for attempt in range(1, RETRY_MAX + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except RateLimitError:
            if attempt == RETRY_MAX:
                print("  [Warning] Still rate limited after retries; skipping this call.")
                return None
            wait = RETRY_BASE_DELAY * attempt
            print(f"  [Warning] Rate limit, retrying in {wait:.1f}s ({attempt}/{RETRY_MAX})")
            time.sleep(wait)
        except NotFoundError as e:
            raise RuntimeError(
                f"Groq model not found: {model}. Update MODEL to a supported model."
            ) from e
    return None


# extract claims

CLAIM_EXTRACTION_PROMPT = """You are an expert at breaking text into atomic, verifiable propositions.

Rules:
- Each claim must be ONE verifiable fact (one subject, one predicate, one object).
- Remove filler phrases like "it is known that" or "studies suggest".
- Each claim must be independently checkable.
- Output ONLY a JSON array of strings, no explanation.

Text to decompose:
{text}

Output format: ["claim 1", "claim 2", "claim 3"]"""


def _heuristic_claims(text: str) -> list[str]:
    """Fallback claim splitter used only when the API is unavailable."""
    lines = [l.strip().rstrip(".") for l in re.split(r"[.\n]", text)]
    return [l for l in lines if len(l) > 20][:8]


def extract_claims(text: str) -> list[str]:
    """Extract atomic verifiable propositions from a block of text."""
    response = groq_chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": CLAIM_EXTRACTION_PROMPT.format(text=text)}],
        temperature=0.0,
        max_tokens=512,
    )
    if response is None:
        print("  [Warning] Claim extraction rate limited; using heuristic fallback.")
        return _heuristic_claims(text)

    raw = response.choices[0].message.content.strip()

    # Prefer a clean JSON array. Use a greedy match so we capture the whole array,
    # not just the first "[ ... ]" that happens to close early.
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            claims = json.loads(match.group())
            cleaned = [c.strip() for c in claims if isinstance(c, str) and c.strip()]
            if cleaned:
                return cleaned
        except json.JSONDecodeError:
            pass

    # Fallback: treat each non-empty line as a claim.
    lines = [l.strip().lstrip("-*0123456789.) ") for l in raw.split("\n")]
    return [l for l in lines if len(l) > 10]


# Generate N independent responses 

REGENERATION_PROMPT = """Answer the following question or complete the following task.
Be factual and specific.

{prompt}"""


def generate_responses(prompt: str, n: int = N_SAMPLES) -> list[str]:
    """Generate N independent answers to the same prompt."""
    responses = []
    for i in range(n):
        response = groq_chat_completion(
            model=MODEL,
            messages=[{"role": "user", "content": REGENERATION_PROMPT.format(prompt=prompt)}],
            temperature=TEMPERATURE,
            max_tokens=512,
        )
        if response is None:
            print(f"  [Warning] Generation {i + 1} was rate limited and skipped.")
        else:
            responses.append(response.choices[0].message.content.strip())
        time.sleep(REQUEST_SPACING)  # staying within the free-tier rate limit
    return responses


# Align claims across responses 

def align_claims(
    anchor_claims: list[str],
    response_claims_list: list[list[str]],
    threshold: float = ALIGN_THRESHOLD,
) -> dict[str, list]:
    aligned = {claim: [] for claim in anchor_claims}
    if not anchor_claims:
        return aligned

    anchor_embeddings = embedder.encode(anchor_claims)

    for response_claims in response_claims_list:
        if not response_claims:
            for claim in anchor_claims:
                aligned[claim].append(None)
            continue

        response_embeddings = embedder.encode(response_claims)
        sim_matrix = cosine_similarity(anchor_embeddings, response_embeddings)

        for i, anchor_claim in enumerate(anchor_claims):
            best_idx = int(np.argmax(sim_matrix[i]))
            best_score = float(sim_matrix[i][best_idx])
            if best_score >= threshold:
                aligned[anchor_claim].append(response_claims[best_idx])
            else:
                aligned[anchor_claim].append(None)  # claim absent in this response

    return aligned


# Compute stability score 

def compute_stability(anchor_claim: str, variants: list) -> dict:
    """
    Stability = how consistently a claim reappears, worded the same way,
    across the independent re-generations.

    We combine two things:
      - similarity: how close the matched variants are to the original claim.
      - coverage:   in how many re-generations the claim showed up at all.

    A famous true fact ("Lehman collapsed on Sept 15, 2008") should reappear
    everywhere with high similarity, so it scores high. A wobbly claim shows
    up sometimes, worded differently, so it scores low.
    """
    present = [v for v in variants if v is not None]
    coverage = len(present) / len(variants) if variants else 0.0

    if not present:
        return {
            "stability_score": 0.0,
            "coverage": 0.0,
            "label": "Unreliable",
            "evidence": [],
            "per_variant_similarity": [],
            "note": "Claim did not appear in any regenerated response",
        }

    anchor_emb = embedder.encode([anchor_claim])
    variant_embs = embedder.encode(present)
    sims = cosine_similarity(anchor_emb, variant_embs)[0]
    mean_sim = float(np.mean(sims))

    penalised_score = mean_sim * (0.7 + 0.3 * coverage)
    penalised_score = float(np.clip(penalised_score, 0.0, 1.0))

    if penalised_score >= 0.80:
        label = "Stable"
    elif penalised_score >= STABILITY_THRESHOLD - 0.10:  # 0.65
        label = "Uncertain"
    else:
        label = "Unreliable"

    return {
        "stability_score": round(penalised_score, 4),
        "coverage": round(coverage, 4),
        "label": label,
        "evidence": present,  # the actual variants, for the "why flagged" feature
        "per_variant_similarity": [round(float(s), 4) for s in sims],
        "note": None,
    }


# Challenge prompt generation 

CHALLENGE_PROMPT_TEMPLATE = """A fact-checking tool found this claim unreliable:
"{claim}"

Write ONE specific, pointed question a critical reader should ask to verify this claim.
The question should target the most likely source of error.
Output only the question, nothing else."""


def generate_challenge_prompt(claim: str) -> str | None:
    """Generate a targeted question that helps the user check a shaky claim."""
    response = groq_chat_completion(
        model=MODEL,
        messages=[{"role": "user", "content": CHALLENGE_PROMPT_TEMPLATE.format(claim=claim)}],
        temperature=0.3,
        max_tokens=64,
    )
    if response is None:
        return None
    return response.choices[0].message.content.strip()


# Main 

def run_pipeline(prompt: str, ai_response: str) -> dict:
    print("\n[1/5] Extracting atomic claims from the AI response...")
    anchor_claims = extract_claims(ai_response)
    print(f"  Found {len(anchor_claims)} claims")

    print(f"\n[2/5] Generating {N_SAMPLES} independent responses...")
    responses = generate_responses(prompt, n=N_SAMPLES)
    print(f"  Got {len(responses)} responses")

    print("\n[3/5] Extracting claims from each response...")
    all_response_claims = []
    for i, resp in enumerate(responses):
        claims = extract_claims(resp)
        all_response_claims.append(claims)
        print(f"  Response {i + 1}: {len(claims)} claims")

    print("\n[4/5] Aligning claims across responses...")
    aligned = align_claims(anchor_claims, all_response_claims)

    print("\n[5/5] Computing stability scores...")
    results = []
    for claim in anchor_claims:
        stability = compute_stability(claim, aligned[claim])

        challenge = None
        if stability["label"] in ("Uncertain", "Unreliable"):
            challenge = generate_challenge_prompt(claim)
            time.sleep(0.3)

        results.append({
            "claim": claim,
            "stability_score": stability["stability_score"],
            "coverage": stability["coverage"],
            "label": stability["label"],
            "challenge_prompt": challenge,
            "evidence": stability["evidence"],
            "per_variant_similarity": stability["per_variant_similarity"],
            "note": stability.get("note"),
        })

    return {
        "disclaimer": "This system does not determine truth - it ranks claims by epistemic risk.",
        "prompt": prompt,
        "original_response": ai_response,
        "n_samples": N_SAMPLES,
        "responses_collected": len(responses),
        "claims": results,
        "summary": {
            "total_claims": len(results),
            "stable": sum(1 for r in results if r["label"] == "Stable"),
            "uncertain": sum(1 for r in results if r["label"] == "Uncertain"),
            "unreliable": sum(1 for r in results if r["label"] == "Unreliable"),
        },
    }


# CLI entry point 
if __name__ == "__main__":
    TEST_PROMPT = "What were the main causes of the 2008 financial crisis?"
    TEST_RESPONSE = (
        "The 2008 financial crisis was primarily caused by the collapse of the US "
        "housing bubble, which peaked in 2006. Banks had issued millions of subprime "
        "mortgages to borrowers with poor credit. These were bundled into "
        "mortgage-backed securities rated AAA by credit rating agencies. Lehman "
        "Brothers collapsed on September 15, 2008, marking the peak of the crisis. "
        "The US government passed the $700 billion TARP bailout within weeks. Over 8 "
        "million jobs were lost in the US as a result."
    )

    result = run_pipeline(TEST_PROMPT, TEST_RESPONSE)

    output_path = "confidence_lens_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total claims: {result['summary']['total_claims']}")
    print(f"Stable:       {result['summary']['stable']}")
    print(f"Uncertain:    {result['summary']['uncertain']}")
    print(f"Unreliable:   {result['summary']['unreliable']}")
    print(f"\nFull output saved to: {output_path}")

    print("\n" + "=" * 60)
    print("CLAIM BREAKDOWN")
    print("=" * 60)
    for c in result["claims"]:
        print(f"\n[{c['label']}] (score: {c['stability_score']}, coverage: {c['coverage']})")
        print(f"  Claim: {c['claim']}")
        if c["challenge_prompt"]:
            print(f"  Challenge: {c['challenge_prompt']}")