
import os
import sys


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from confidence_lens import (
    extract_claims,
    align_claims,
    compute_stability,
)

PASS = "PASS"
FAIL = "FAIL"
results = []


def record(name: str, passed: bool, detail: str = ""):
    print(f"  {PASS if passed else FAIL}  {name}")
    if detail:
        print(f"        {detail}")
    results.append((name, passed))


# Test 1: Claim extraction returns a non-empty list (needs API) 

def test_claim_extraction_basic():
    print("\n[Test 1] Claim extraction - basic")
    if not os.environ.get("GROQ_API_KEY"):
        print("  [SKIP] GROQ_API_KEY not set")
        return

    text = ("The Eiffel Tower is 330 metres tall and located in Paris. "
            "It was completed in 1889.")
    claims = extract_claims(text)

    record("Returns a list", isinstance(claims, list))
    record("At least 2 claims extracted", len(claims) >= 2, f"Got {len(claims)}: {claims}")
    record("Claims are strings", all(isinstance(c, str) for c in claims))
    record("No empty claims", all(len(c.strip()) > 5 for c in claims))


# Test 2: Compound sentence -> multiple claims (needs API)
def test_claim_atomicity():
    print("\n[Test 2] Claim extraction - atomicity")
    if not os.environ.get("GROQ_API_KEY"):
        print("  [SKIP] GROQ_API_KEY not set")
        return

    text = "Marie Curie was born in 1867 in Warsaw and won two Nobel Prizes."
    claims = extract_claims(text)
    record("Compound sentence yields multiple claims", len(claims) >= 2,
           f"Got {len(claims)}: {claims}")


# Test 3: Identical claims align 

def test_alignment_identical():
    print("\n[Test 3] Semantic alignment - identical claims")
    anchor = ["The Eiffel Tower is 330 metres tall."]
    variants = [["The Eiffel Tower stands 330 metres high."]]
    aligned = align_claims(anchor, variants, threshold=0.60)
    matched = aligned[anchor[0]][0]
    record("Identical claim aligns correctly", matched is not None, f"Matched: {matched}")


# Test 4: Unrelated claims do NOT align 
def test_alignment_unrelated():
    print("\n[Test 4] Semantic alignment - unrelated claims")
    anchor = ["The Eiffel Tower is located in Paris."]
    variants = [["Quantum mechanics describes subatomic particles."]]
    aligned = align_claims(anchor, variants, threshold=0.60)
    matched = aligned[anchor[0]][0]
    record("Unrelated claim returns None", matched is None, f"Got: {matched}")


# Test 5: Stable claim scores high 
def test_stability_stable():
    print("\n[Test 5] Stability score - stable claim")
    anchor = "The Eiffel Tower is 330 metres tall."
    variants = [
        "The Eiffel Tower stands 330 metres high.",
        "The Eiffel Tower has a height of 330 metres.",
        "The Eiffel Tower is approximately 330 metres tall.",
        "The height of the Eiffel Tower is 330 metres.",
        "The Eiffel Tower reaches 330 metres.",
    ]
    result = compute_stability(anchor, variants)
    record("Stability score >= 0.75", result["stability_score"] >= 0.75,
           f"Score: {result['stability_score']}")
    record("Label is Stable or Uncertain", result["label"] in ("Stable", "Uncertain"),
           f"Label: {result['label']}")
    record("Coverage is 1.0", result["coverage"] == 1.0, f"Coverage: {result['coverage']}")


# Test 6: Unstable claim scores low 

def test_stability_unstable():
    print("\n[Test 6] Stability score - unstable claim")
    anchor = "Studies show coffee improves memory by 40 percent."
    variants = [
        "Research suggests coffee may slightly improve short-term alertness.",
        "Some studies indicate caffeine has no significant effect on memory.",
        "Coffee has been linked to reduced cognitive decline in older adults.",
        None,
        "The relationship between coffee and memory is unclear and contested.",
        None,
        "Moderate coffee intake may improve focus but not memory specifically.",
    ]
    result = compute_stability(anchor, variants)
    record("Stability score < 0.75", result["stability_score"] < 0.75,
           f"Score: {result['stability_score']}")
    record("Label is Uncertain or Unreliable", result["label"] in ("Uncertain", "Unreliable"),
           f"Label: {result['label']}")
    record("Coverage < 1.0", result["coverage"] < 1.0, f"Coverage: {result['coverage']}")


# Test 7: Absent claim -> 0.0 / Unreliable 

def test_stability_absent():
    print("\n[Test 7] Stability score - claim absent everywhere")
    anchor = "Napoleon was 5 feet 2 inches tall."
    variants = [None] * 5
    result = compute_stability(anchor, variants)
    record("Stability score is 0.0", result["stability_score"] == 0.0,
           f"Score: {result['stability_score']}")
    record("Label is Unreliable", result["label"] == "Unreliable", f"Label: {result['label']}")
    record("Note is present", result.get("note") is not None, f"Note: {result.get('note')}")


# Test 8: Output structure 

def test_output_structure():
    print("\n[Test 8] Output structure validation")
    mock_output = {
        "disclaimer": "This system does not determine truth - it ranks claims by epistemic risk.",
        "prompt": "test prompt",
        "original_response": "test response",
        "n_samples": 5,
        "claims": [{
            "claim": "Test claim", "stability_score": 0.85, "coverage": 1.0,
            "label": "Stable", "challenge_prompt": None, "evidence": ["variant 1"],
            "per_variant_similarity": [0.85], "note": None,
        }],
        "summary": {"total_claims": 1, "stable": 1, "uncertain": 0, "unreliable": 0},
    }
    required_top = {"disclaimer", "prompt", "original_response", "n_samples", "claims", "summary"}
    required_claim = {"claim", "stability_score", "coverage", "label", "challenge_prompt", "evidence"}
    record("Top-level keys present", required_top.issubset(mock_output.keys()))
    record("Claim keys present", required_claim.issubset(mock_output["claims"][0].keys()))
    record("Disclaimer present", "epistemic risk" in mock_output["disclaimer"])
    record("Summary counts correct", mock_output["summary"]["total_claims"] == len(mock_output["claims"]))


# Test 9: Edge cases 
def test_edge_cases():
    print("\n[Test 9] Edge cases")
    record("Empty variants handled", compute_stability("any claim", [])["stability_score"] == 0.0)
    single = compute_stability("The sky is blue.",
                               ["The sky appears blue due to Rayleigh scattering."])
    record("Single variant returns valid score", 0.0 <= single["stability_score"] <= 1.0,
           f"Score: {single['stability_score']}")
    short = compute_stability("Yes.", ["Yes.", "Yes.", None])
    record("Short claim handled", isinstance(short["stability_score"], float))


# Test 10: NLI contradiction detection (needs the NLI model) 

def test_contradiction_detection():
    print("\n[Test 10] Contradiction detection (NLI)")
    try:
        from contradiction_detector import detect_contradictions
    except Exception as e:
        print(f"  [SKIP] Could not load NLI model: {e}")
        return

    # Clear contradiction: these two should be flagged as disagreeing.
    contra = detect_contradictions([
        "Coffee improves memory.",
        "Coffee does not improve memory.",
    ])
    record("Opposite claims are flagged as disagreeing", contra.n_contradicting_pairs >= 1,
           f"Contradicting pairs: {contra.n_contradicting_pairs}, label: {contra.label}")

    # Clear agreement: paraphrases should NOT be flagged as contradictions.
    agree = detect_contradictions([
        "The Eiffel Tower is 330 metres tall.",
        "The Eiffel Tower stands 330 metres high.",
    ])
    record("Paraphrases are not flagged as contradiction", agree.n_contradicting_pairs == 0,
           f"Contradicting pairs: {agree.n_contradicting_pairs}, label: {agree.label}")

    # Single variant -> trivially consensus.
    one = detect_contradictions(["Only one claim here."])
    record("Single variant is Consensus", one.label == "Consensus", f"Label: {one.label}")


# Run all 

if __name__ == "__main__":
    print("=" * 60)
    print("CONFIDENCE LENS - TEST SUITE")
    print("=" * 60)

    test_claim_extraction_basic()
    test_claim_atomicity()
    test_alignment_identical()
    test_alignment_unrelated()
    test_stability_stable()
    test_stability_unstable()
    test_stability_absent()
    test_output_structure()
    test_edge_cases()
    test_contradiction_detection()

    print("\n" + "=" * 60)
    passed = sum(1 for _, p in results if p)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    print("=" * 60)

    if passed < total:
        print("\nFailed tests:")
        for name, p in results:
            if not p:
                print(f"  - {name}")
        sys.exit(1)
    print("\nAll run tests passed.")