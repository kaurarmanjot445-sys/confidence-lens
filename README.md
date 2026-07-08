Confidence Lens
I built this because I kept catching myself forwarding AI answers without checking them. When you don't know a topic well, a confident AI answer feels true. You can't tell the difference. And when it's wrong, it's your name on it, not the AI's.
This tool checks how much you should actually trust an AI answer.

how it works
You give it a question and an AI's answer. It does two things:
1. stability check 
It asks the same question multiple times and breaks every answer into small, checkable claims. Then it checks — did the same claim show up every time? If a claim appears once and disappears in other tries, that's a bad sign. If it keeps showing up worded the same way, it's probably solid.

2. contradiction check 
This one was interesting. I started with embedding similarity to find disagreements, but it didn't work. "Coffee improves memory" and "coffee does not improve memory" have almost identical embeddings because they're about the same topic. The model couldn't tell them apart. So I switched to an NLI (natural language inference) model that's actually trained to detect whether two sentences agree or disagree. That fixed it.
Each claim gets a risk label: stable, uncertain, or unreliable. Unstable claims also get a challenge question to help you verify them yourself.

setup
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
On Windows use setx GROQ_API_KEY "your_key_here" in PowerShell, then restart your terminal.
Do not put your API key in the code. Ever. I learned this the hard way.

run
python confidence_lens.py      # full stability pipeline
python contradiction_detector.py      # contradiction detection demo
python test_pipeline.py              # tests (26/26 passing)

files
confidence_lens.py - claim extraction, alignment, stability scoring
contradiction_detector.py  - NLI contradiction detection + combined risk score
test_pipeline.py - test suite
requirements.txt - dependencies

known issues
stability scoring penalises true facts that the model just doesn't always mention. "Lehman Brothers collapsed on September 15 2008" scored uncertain because only 2 out of 5 regenerations included it. The fact is true, the model just didn't always bring it up. Coverage and stability are not the same thing and I haven't fully separated them yet.
NLI runs pairwise so it gets slow with lots of variants.
the whole thing depends on the Groq model's claim extraction being decent. Sometimes it pulls out too many claims or merges two facts into one.

what's next
web demo so people can try it without installing anything
calibration against a set of claims I know are true or false, to see if the risk score actually tracks truth
better separation of "the model didn't mention it" vs "the model actively disagreed"


That reads like you because it talks about real problems you actually hit (the embedding bug, the Lehman calibration issue, the API key mistake). Paste it in, push it.
