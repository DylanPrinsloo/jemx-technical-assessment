import re
import polars as pl
from sklearn.metrics import classification_report

# Multilingual pattern definitions (isiZulu, Afrikaans, English)
PATTERNS = {
    "relief_no_show": re.compile(
        r"(not?\s+pitch|didn'?t\s+come|no[- ]show|stood\s+in|covered|"
        r"replacement|relie[fv]|akafikanga|akezanga|ngimele|afwesig|"
        r"geen\s+aflos|double\s+shift)",
        re.IGNORECASE,
    ),
    "equipment_failure": re.compile(
        r"(generator|masjien|stukkend|broken|fault|power\s+outage|"
        r"load[\s-]?shedding|gate\s+fault|alarm|water\s+leak|herstel|krag)",
        re.IGNORECASE,
    ),
    "handover_delay": re.compile(
        r"(handover|oorhandiging|keys?|sleutel|lock\s+box|missing\s+keys)",
        re.IGNORECASE,
    ),
    "client_request": re.compile(
        r"(client\s+(asked|wanted|requested|signed|email)|approved|"
        r"customer\s+request|klient|gemagtig|extra\s+(patrol|man|guard|hours|shift)|stocktake)",
        re.IGNORECASE,
    ),
}

ROUTINE_SUBSTRINGS = re.compile(
    r"(all\s+good|quiet\s+shift|sharp|ok|normal\s+shift|alles\s+reg|lungile|routine|nothing\s+to\s+report)",
    re.IGNORECASE,
)

def classify_text(note: str | None) -> str:
    if not note or not str(note).strip():
        return "routine_or_unclear"

    text = str(note).strip()

    # Routine statements without operational issues
    if ROUTINE_SUBSTRINGS.search(text) and not any(p.search(text) for p in PATTERNS.values()):
        return "routine_or_unclear"

    # Operational failures evaluated before client requests to prevent client mentions from masking no-shows
    if PATTERNS["relief_no_show"].search(text):
        return "relief_no_show"
    if PATTERNS["equipment_failure"].search(text):
        return "equipment_failure"
    if PATTERNS["handover_delay"].search(text):
        return "handover_delay"
    if PATTERNS["client_request"].search(text):
        return "client_request"

    return "routine_or_unclear"

# -------------------------------------------------------------
# 1. Validation Evaluation: Hand-labeled gold sample audit
# -------------------------------------------------------------
GOLD_TEST_SET = [
    ("extra patrol per client email, approved by office", "client_request"),
    ("next shift akafikanga, ngihlale kuze kube 6 !", "relief_no_show"),
    ("quiet shift", "routine_or_unclear"),
    ("all good", "routine_or_unclear"),
    ("Mokoena didnt come in, covered the post", "relief_no_show"),
    ("Fourie didnt come in, covered the post", "relief_no_show"),
    ("client wanted extra man on the gate for the event, approved", "client_request"),
    ("stood in for Mokoena", "relief_no_show"),
    ("handover late again, keys missing", "handover_delay"),
    ("sharp", "routine_or_unclear"),
    ("stocktake ran over, client asked us to remain, they know they pay for it", "client_request"),
    ("no replacement sent, ngicla sort this out", "relief_no_show"),
    ("generator fault, stayed to monitor", "equipment_failure"),
    ("oorhandiging was laa, gewag vir sleutels", "handover_delay"),
    ("uFourie akezanga namhlanje, ngimele yena", "relief_no_show"),
    ("client requested additional guards for perimeter", "client_request"),
    ("gate motor stukkend, had to manually operate", "equipment_failure"),
    ("routine shift, nothing to report", "routine_or_unclear"),
    ("double shift covered due to absent staff", "relief_no_show"),
    ("keys lost during handover", "handover_delay"),
    ("power outage, waited for power restoration", "equipment_failure"),
    ("klient het gevra vir ekstra ure", "client_request"),
    ("alles reg", "routine_or_unclear"),
    ("relief guard was a no-show", "relief_no_show"),
    ("normal shift", "routine_or_unclear"),
]

y_true = [item[1] for item in GOLD_TEST_SET]
y_pred = [classify_text(item[0]) for item in GOLD_TEST_SET]

print("\n--- VALIDATION EVALUATION (GOLD TEST SAMPLE) ---")
print(classification_report(y_true, y_pred, zero_division=0))

# -------------------------------------------------------------
# 2. Run Classification on Full Dataset & Export
# -------------------------------------------------------------
notes_df = pl.read_csv("data/shift_notes.csv")

classified = notes_df.with_columns(
    pl.col("note").map_elements(classify_text, return_dtype=pl.String).alias("category")
)

output_df = classified.select(["shift_id", "category", "note"])
output_df.write_csv("note_classifications.csv")
print(f"Exported note_classifications.csv: {len(output_df)} rows")

print("\n--- CATEGORY BREAKDOWN ---")
print(output_df.group_by("category").len().sort("len", descending=True))

# -------------------------------------------------------------
# 3. Two Piles Breakdown Across Sites
# -------------------------------------------------------------
shifts = pl.read_csv("data/shifts.csv")
shifts_with_cat = shifts.join(output_df, on="shift_id", how="left").with_columns(
    pl.when(pl.col("category") == "client_request")
    .then(pl.lit("Client Requested"))
    .when(pl.col("category").is_in(["relief_no_show", "equipment_failure", "handover_delay"]))
    .then(pl.lit("Operational Failure"))
    .otherwise(pl.lit("Routine/Unlogged"))
    .alias("pile")
)

print("\n--- THE TWO PILES BREAKDOWN BY SITE ---")
pile_summary = shifts_with_cat.group_by(["site_id", "pile"]).len().sort(["site_id", "pile"])
print(pile_summary)