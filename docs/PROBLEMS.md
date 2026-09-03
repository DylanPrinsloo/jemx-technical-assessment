# 1. Data Assumptions

When prompting Gemini, I was not fully confident about the BCEA compliance thresholds.

Missing clock-outs could be relevant, as I identified several missing records.

Public holidays are included. I did not explore this in depth, although Gemini pointed it out, so I skipped it.

# 2. Sorting and Audit Findings

Gemini mapped multiple languages.

Edge cases were identified by Gemini; however, it seemed to struggle with null values. I initially skipped these.

# 3. Training a Model

The data appears to be clustered, which is beneficial because it makes it easier to define rules (e.g., four days worked followed by two days off).

There are potential interaction effects that I was not able to identify. I noted this as something to revisit and scheduled time to investigate further.

Gemini identified a potential overfitting issue. This could be mitigated by introducing generalization elements into the training process.
