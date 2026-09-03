Time 2026/09/03 08:45 -----------------------------------------------------------------------------------

Clean up duplication
Generate context (Gemini will help me)
Set up docs so the project is easy to understand
Use UV (package manager which I love using)

Use Data Wrangler to understand data

Time 2026/09/03 08:56 -----------------------------------------------------------------------------------

Identified serious issues in the data; we would have to drop null values. Running a script, I then dropped nulls and had a total: "Nulls remaining: 0".

I see that the data cuts off on Wednesday ("max date: 2026-08-12").

Total unique employees: 213

Shifts logged Mon–Wed: 430

Unique employees active: 207

Null clock-outs in target week: 3 (need to verify this)

Time 2026/09/03 09:02 -----------------------------------------------------------------------------------

I looked into one task which required the 5-Minute Video Test, where we would be "picking one person's risk score and explaining where it came from step-by-step so simply that a non-technical contract manager sitting in a bakkie can follow it".

Data scale reality seems to be limited; I would have to create more dummy data to generalize for model improvements.

Calculating a risk score would mean I would have to take "hours" and "extra hours" and divide it by the variance scale. Our variance scale is 55 hours, as this is the legal limit. This is, however, completely hypothetical. Asking Gemini suggested that workers A, B, and C would work between 40–60 hours. So worker A: 40/55 = 0.73%.

What breach probability means is whether an employee will exceed the statutory limit of 10 overtime hours in a single payroll week by Sunday night (assuming 7 days total).

Time 2026/09/03 09:17 -----------------------------------------------------------------------------------

The 93% accuracy figure identified is an illusion created by the 3.1% positive class imbalance (2,056 compliant weeks vs. 66 breaches).

I generated a predictions.csv file and a note_classification.csv.

Time 2026/09/03 09:26 -----------------------------------------------------------------------------------

Set up Streamlit

Run app.py, use Gemini to build a simple front-facing dashboard (depend on Streamlit to make it presentable).

Time 2026/09/03 09:44 -----------------------------------------------------------------------------------

I completed the task; I had just reviewed the work before sending.
