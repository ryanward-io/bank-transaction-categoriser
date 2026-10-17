# Bank transaction categoriser

A small machine-learning project: learn spending categories (Food, Myki, Supermarket, ...) from the text on a bank statement, and measure honestly how well that works.

**Live demo: https://ryanward.dev/bank-transaction-categoriser/** (part of my [projects](https://ryanward.dev/projects/))

## What it shows

- **Try it:** type a transaction description (or click an example) and see the predicted category with probabilities. Pick the text features and the model.
- **How well does it work:** accuracy, macro F1 and the "always guess Food" baseline, plus a confusion matrix and per-category scores.
- **Compare every combination:** 3 feature types x 5 models, scored the same way.

## How it runs

There is no server. The app is a [Streamlit](https://streamlit.io) script running in your browser via [stlite](https://github.com/whitphx/stlite) (Streamlit on [Pyodide](https://pyodide.org)). `index.html` loads stlite from a CDN and mounts `streamlit_app.py` and `data/transactions.csv`, so it is plain static hosting on GitHub Pages. The first load takes 10-30 seconds while Python and scikit-learn download.

Run it locally with `python3 -m http.server 8000`, or as a normal Streamlit app with `pip install streamlit scikit-learn altair pandas && streamlit run streamlit_app.py`.

## The evaluation

Accuracy comes from 5-fold cross-validation over the *distinct* descriptions (fixed seed), so every description is scored by a model that never saw it. The 881 transactions contain only 246 distinct descriptions ("Powershop - Direct Debit" appears 49 times), so a random split of the raw rows would put copies on both sides and overstate the result (about 97% instead of 83% for the default model). The do-nothing baseline of always guessing 'Food' scores 59%.

Macro F1 (about 0.5) is much lower than accuracy because the categories are very imbalanced: 'Food' is a catch-all (145 of 246 descriptions) while Petrol has 3, Spotify 1 and Swimming 1. On a dataset this small the differences between the better models are within noise.

## Data and privacy

`data/transactions.csv` (description, category, how often it appeared) is a sanitised sample of my own transactions with my own labels. The raw file is **not** in this repository. `tools/sanitise.py` shows exactly what was removed or changed:

- Person-to-person payment lines, which carried names, phone numbers and BSB/account numbers, and free-text memos.
- A handful of lines that could be a person's name or reveal an affiliation.
- Dates (there are no amounts). Employer-identifying salary lines are replaced with a generic payroll description.
- Descriptions that had conflicting labels were dropped.

Labels were not otherwise changed, so some are questionable (some taxis are labelled Food). That is realistic, and it caps the achievable score.

## Limits

- 246 descriptions from one person's account. A different person's spending would need new labels.
- It is a text classifier on descriptions only; amounts, dates and counterparties are ignored.
- Person-to-person payments were removed, so the model is not trained to handle them.

## History

Started in 2019 as a script comparing vectorisers and models; rebuilt in 2026 as a scikit-learn pipeline with a browser front end.

## Licence

Code and data: MIT (see `LICENSE`).
