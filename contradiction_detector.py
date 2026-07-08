
import sys
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import CrossEncoder

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# NLI cross-encoder. Outputs three scores: contradiction / entailment / neutral.
NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
_nli_model = CrossEncoder(NLI_MODEL_NAME)

# Work out which output index means "contradiction" from the model config,
# instead of hard-coding it (the order can differ between models).
def _contradiction_index(model) -> int:
    id2label = getattr(model.model.config, "id2label", None)
    if id2label:
        for idx, label in id2label.items():
            if "contradict" in str(label).lower():
                return int(idx)
    # Documented default order for this model family: 0=contradiction.
    return 0


CONTRADICTION_IDX = _contradiction_index(_nli_model)
CONTRADICTION_PROB_THRESHOLD = 0.50  # min softmax prob to count as a contradiction


# Data structure 
@dataclass
class ContradictionResult:
    n_variants: int                       # how many variants were compared
    n_pairs: int                          # number of unordered variant pairs
    n_contradicting_pairs: int            # how many of those pairs disagree
    disagreement_score: float             # 0.0 = full consensus, 1.0 = all disagree
    label: str                            # Consensus | Partial disagreement | Contradictory
    positions: list[list[str]] = field(default_factory=list)   # variants grouped by stance
    contradicting_examples: list[tuple[str, str]] = field(default_factory=list)


# NLI helpers 

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _contradiction_prob(premise: str, hypothesis: str) -> float:
    """Probability that `hypothesis` contradicts `premise`, per the NLI model."""
    logits = _nli_model.predict([(premise, hypothesis)])
    logits = np.asarray(logits).reshape(-1)
    probs = _softmax(logits)
    return float(probs[CONTRADICTION_IDX])


def _pair_contradicts(a: str, b: str) -> bool:
    """A pair disagrees if EITHER direction is confidently a contradiction."""
    p_ab = _contradiction_prob(a, b)
    p_ba = _contradiction_prob(b, a)
    return max(p_ab, p_ba) >= CONTRADICTION_PROB_THRESHOLD


# Core: detect disagreement across variants 
def detect_contradictions(variants: list[str]) -> ContradictionResult:
    """Measure how much the variants of a claim disagree with each other."""
    variants = [v for v in variants if v]  # drop None/empty
    n = len(variants)

    if n <= 1:
        return ContradictionResult(
            n_variants=n, n_pairs=0, n_contradicting_pairs=0,
            disagreement_score=0.0, label="Consensus",
            positions=[variants] if variants else [],
            contradicting_examples=[],
        )

    # Build a contradiction matrix over all pairs.
    contradicts = np.zeros((n, n), dtype=bool)
    contradicting_examples: list[tuple[str, str]] = []
    n_pairs = 0
    n_contra = 0

    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if _pair_contradicts(variants[i], variants[j]):
                contradicts[i, j] = contradicts[j, i] = True
                n_contra += 1
                if len(contradicting_examples) < 5:
                    contradicting_examples.append((variants[i], variants[j]))

    disagreement_score = round(n_contra / n_pairs, 4) if n_pairs else 0.0

    positions: list[list[int]] = []
    for i in range(n):
        placed = False
        for group in positions:
            if not any(contradicts[i, k] for k in group):
                group.append(i)
                placed = True
                break
        if not placed:
            positions.append([i])
    position_texts = [[variants[k] for k in group] for group in positions]

    # Label based on how widespread the disagreement is.
    if n_contra == 0:
        label = "Consensus"
    elif disagreement_score < 0.34:
        label = "Partial disagreement"
    else:
        label = "Contradictory"

    return ContradictionResult(
        n_variants=n,
        n_pairs=n_pairs,
        n_contradicting_pairs=n_contra,
        disagreement_score=disagreement_score,
        label=label,
        positions=position_texts,
        contradicting_examples=contradicting_examples,
    )


# Combined epistemic risk score -

def compute_epistemic_risk(
    stability_score: float,       
    disagreement_score: float,    
    stability_weight: float = 0.5,
    disagreement_weight: float = 0.5,
) -> dict:
    instability = 1.0 - stability_score
    risk = stability_weight * instability + disagreement_weight * disagreement_score
    risk = round(float(np.clip(risk, 0.0, 1.0)), 4)

    if risk < 0.25:
        label = "Low risk"
    elif risk < 0.55:
        label = "Moderate risk"
    else:
        label = "High risk"

    return {
        "epistemic_risk_score": risk,
        "risk_label": label,
        "components": {
            "instability": round(instability, 4),
            "disagreement": round(disagreement_score, 4),
        },
    }


#  Human-readable explanation ("why flagged") 

def explain_flagged_claim(claim: str, result: ContradictionResult, risk: dict) -> str:
    lines = [
        f'Claim: "{claim}"',
        f'Risk level: {risk["risk_label"]} (score: {risk["epistemic_risk_score"]})',
        "",
    ]

    if result.label == "Consensus":
        lines.append(
            "The independent responses agreed on this claim. Any risk comes from "
            "wording instability rather than contradiction."
        )
    else:
        n_pos = len(result.positions)
        lines.append(
            f"{n_pos} distinct positions emerged across {result.n_variants} independent "
            f"responses ({result.n_contradicting_pairs} of {result.n_pairs} pairs directly "
            f"contradicted each other):"
        )
        for i, group in enumerate(result.positions, 1):
            lines.append(f'  Position {i}: "{group[0]}"')

    lines += [
        "",
        "This system does not determine truth - it ranks claims by epistemic risk.",
        "Verify this claim independently before relying on it.",
    ]
    return "\n".join(lines)


# Integrate with output 

def enrich_pipeline_output(pipeline_output: dict) -> dict:
    enriched = []
    for claim_data in pipeline_output["claims"]:
        variants = [v for v in claim_data.get("evidence", []) if v]
        result = detect_contradictions(variants)
        risk = compute_epistemic_risk(
            stability_score=claim_data["stability_score"],
            disagreement_score=result.disagreement_score,
        )
        explanation = explain_flagged_claim(claim_data["claim"], result, risk)

        enriched.append({
            **claim_data,
            "contradiction": {
                "n_variants": result.n_variants,
                "n_pairs": result.n_pairs,
                "n_contradicting_pairs": result.n_contradicting_pairs,
                "disagreement_score": result.disagreement_score,
                "label": result.label,
                "positions": result.positions,
                "contradicting_examples": result.contradicting_examples,
            },
            "epistemic_risk": risk,
            "explanation": explanation,
        })

    pipeline_output["claims"] = enriched
    pipeline_output["claims"].sort(
        key=lambda x: x["epistemic_risk"]["epistemic_risk_score"], reverse=True
    )
    return pipeline_output


if __name__ == "__main__":
    print("=" * 60)
    print("CONTRADICTION DETECTOR (NLI) - DEMO")
    print("=" * 60)

    coffee_variants = [
        "Coffee improves memory by 40 percent.",
        "Coffee has no significant effect on memory.",
        "Coffee may slightly improve alertness but not memory.",
        "Heavy coffee use is linked to memory decline in older adults.",
        "The evidence on coffee and memory is mixed and inconclusive.",
        "Coffee improves short-term recall by enhancing dopamine pathways.",
        "No peer-reviewed study confirms coffee improves memory significantly.",
    ]

    print(f"\nClaim under test, {len(coffee_variants)} variants that genuinely disagree.\n")
    result = detect_contradictions(coffee_variants)

    print(f"Variants:              {result.n_variants}")
    print(f"Pairs compared:        {result.n_pairs}")
    print(f"Contradicting pairs:   {result.n_contradicting_pairs}")
    print(f"Disagreement score:    {result.disagreement_score}")
    print(f"Label:                 {result.label}")
    print(f"Distinct positions:    {len(result.positions)}")

    for i, group in enumerate(result.positions, 1):
        print(f"\n  Position {i}:")
        for v in group:
            print(f"    - {v}")

    risk = compute_epistemic_risk(stability_score=0.45,
                                  disagreement_score=result.disagreement_score)
    print(f"\nEpistemic risk score:  {risk['epistemic_risk_score']}  ({risk['risk_label']})")
    print("\n" + explain_flagged_claim(coffee_variants[0], result, risk))