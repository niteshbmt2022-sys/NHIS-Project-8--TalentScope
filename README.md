# TalentScope — Freelance Market Intelligence

**Discover Opportunities. Understand Markets. Make Smarter Decisions.**

TalentScope is a Streamlit analytics product for exploring historical Upwork opportunities, comparing offered compensation and building a personalized career shortlist. It combines a consistent product shell with eleven independent analytical pages, real drill-downs, sparse NLP matching and chronological demand evaluation.

## 1. Product overview

Raw posting counts, missing compensation and uneven collection can obscure the market. TalentScope makes source coverage and sample sizes visible while helping users search postings, compare countries and roles, inspect remote-title signals and explore carefully qualified future scenarios.

The product analyzes an uploaded historical snapshot. It does not imply a live feed or current hiring availability. No API keys, paid services, bundled dataset, model pickle files or database are required.

## 2. Navigation and functionality

| Group | Page | Purpose and interactions |
|---|---|---|
| Overview | Market Overview | Six clickable KPIs, activity intervals, category/country chart drill-downs, compensation, calculated highlights and quick navigation |
| Market Intelligence | Market Explorer | Literal title search, global scope filters, date/title sorting, 25/50/100-row pagination, title/row-selected posting details, source links and full filtered downloads |
| Market Intelligence | Salary Intelligence | Hourly/fixed cohorts, mean/median/range, observation thresholds, optional IQR exclusion, category and country comparisons, distributions and statistical associations |
| Market Intelligence | Skills & Role Insights | Title-keyword TF-IDF, role composition, keyword compensation, word/character related-title retrieval, complete-week growth and inspectable classification rules |
| Market Intelligence | Global Market Insights | ISO country map, volume/rate toggle, country drill-down, local role composition and hourly-bound statistics |
| Market Intelligence | Remote Work Insights | Title-derived signal share, geography, category, monthly activity and hourly compensation comparison |
| Predictive Analytics | Demand Forecast | Category/detailed-role/total-count scenarios, adjustable historical window, seven/thirty-day projections and holdout comparison |
| Predictive Analytics | Future Market Outlook | Emerging/declining commentary, monthly volume/payment/location/category dynamics and category/remote/rate scenarios |
| Career Tools | Career Recommendations | Personalized match cards, source links, explanations, saved-session shortlist, remove saved, clear preferences and downloads |
| Data & Model Insights | Data Quality | Full-source coverage, missingness, duplicates, type/salary validation, replacement upload and reprocessing |
| Data & Model Insights | Model Performance | Independent model-comparison page, true error metrics, split periods, selected-series holdout charts and methodology |

## 3. Four-file architecture

```text
TalentScope/
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

`app.py` contains validation, preprocessing, analytics, statistical analysis, NLP, regressors, recommendation logic, Plotly charts, downloads and the UI. Importing it does not launch the UI. Data and model artifacts are held in memory; no runtime CSVs or pickle files are created. Downloads are generated in memory. Full-source and filtered-directory exports are deferred until requested.

### Brand identity and product landing

TalentScope's original **Orbit / Opportunity** symbol combines a globe, geographic meridians, an ascending analytics route and a briefcase-shaped opportunity node. `brand_svg()` generates the inline vector artwork, the symbol-plus-wordmark lockup, compact symbol, dark-on-light and white-on-navy versions. The compact favicon uses the same geometry on a navy rounded tile and is delivered as an SVG data URI by `st.set_page_config`. No logo files, raster images, remote font downloads or third-party brand assets are required.

The primary lockup appears in the sidebar and product landing; the application header switches to a compact symbol when navigation is collapsed. The footer, About section and real processing states share the same artwork. Inter is preferred when installed, with Segoe UI and system fonts as local fallbacks. Interface controls retain the consistent Material Symbols icon set; landing capability illustrations use matching outlined vector geometry.

**Product home** (`/welcome`) introduces the platform with “Navigate the Global Freelance Economy,” dataset connection, an explicit **Enter analytics workspace** action, capability cards and an intelligence preview. Before upload, preview values are empty and entry is disabled. Once validated, the preview shows actual full-source counts, country coverage, mean hourly midpoint and inferred category volumes. Returning to Product home preserves workspace filters and accepted data. The eleven analytical destinations remain available through the same registered navigation system.

## 4. Dataset description and actual inspection

Source: the user-supplied `all_upwork_jobs_2024-02-07-2024-03-24.csv`, inspected in full on **October 30, 2026**. The application accepts this dataset through its CSV uploader. Obtain the original dataset separately; do not commit it to GitHub. The filename is not used to infer coverage.

| Original field | Inferred input dtype | Original missing rows |
|---|---|---:|
| title | object/text | 1 |
| link | object/text | 1 |
| published_date | object/text | 0 |
| is_hourly | boolean | 0 |
| hourly_low | float64 | 142,406 |
| hourly_high | float64 | 146,053 |
| budget | float64 | 140,937 |
| country | object/text | 5,077 |

Actual inspection results:

- **244,828 input rows and 8 columns**; no exact row duplicates or identical raw link strings.
- Canonical Upwork identifiers expose **517 repeated job postings**. Removing these leaves **244,311** rows. First occurrence in CSV order is retained; this does not imply a record is the latest version.
- Parsed publication range: **2023-11-02 through 2024-03-24 UTC**. No invalid dates in the supplied CSV. Early records are sparse; sustained volume starts February 13, 2024. March 24 ends at approximately 14:16 UTC and may be partial. The lower volume on February 15 also merits coverage caution.
- No negative, zero, non-finite or reversed salary values were detected in the original numeric fields. No hourly/fixed field conflicts were detected.
- Before deduplication, 3,647 hourly rows have only one bound; afterward, **3,639** do. Midpoints are available for **98,545** deduplicated rows. **42,058** hourly records have no usable midpoint.
- **103,708** deduplicated fixed-price records have usable budgets. All salary comparisons keep this measure separate from hourly rates.
- IQR flags identify **6,374 hourly midpoints** and **12,961 fixed budgets** as outliers. These are flags, not evidence of errors. Outliers remain included unless the user checks the Salary Intelligence exclusion option.
- Country normalization handles HTML entities and aliases. **212 nonmissing location labels** remain. Holy See and the obsolete Netherlands Antilles label are retained but unmapped in the installed country registry; they do not appear on the map.
- Deduplicated payment mix: **57.5508% hourly**, **42.4492% fixed-price**. Mean observed hourly midpoint: **29.70994 source units**. The dataset has no currency column; USD is not asserted.
- The transparent first-match taxonomy leaves **59,476 rows (24.34%) unclassified**. This coverage limit is displayed rather than hidden. Among classified rows, Development has the largest count (58,437).

Missing hourly fields in fixed-price jobs and missing budgets in hourly jobs are structurally expected; they are not all data defects. These are posting/client locations from the source, not verified worker locations. No job descriptions, applicant counts, hires, hours worked, exchange rates or realized earnings are available.

## 5. Preprocessing policy

1. Require all eight original columns and at least one row. Additional input columns are retained unless they collide with the documented derived feature names. UTF-8 / UTF-8 BOM CSVs are supported; other encodings should be converted before upload. App limit: 150 MB.
2. Preserve original field values. Parse dates into a separate UTC-normalized, timezone-naive daily `date` column. Invalid dates stay in the cleaned dataset/audit but cannot enter date-filtered analysis or forecasting.
3. Decode HTML, apply Unicode NFKC, lowercase and normalize punctuation/spacing. Preserve useful technical variants (`c++` becomes `cpp`, `c#` becomes `csharp`, `.net` becomes `dotnet`). Missing titles remain empty normalized text and are excluded from recommendation candidates.
4. Remove exact rows, then repeated canonical HTTPS Upwork job IDs. Unusable/non-Upwork URLs are never exposed as clickable job links. Records without URLs can still contribute to analysis.
5. Recognize boolean values and true/false/1/0 strings; unknown payment types are audited. No salary type is inferred from an arbitrary salary value.
6. Coerce numeric fields into separate validated features. Non-finite, nonpositive and invalid numbers become missing derived values. Reversed bounds invalidate both derived bounds. Payment-conflicting fields do not enter the wrong analysis.
7. Midpoint requires **both** valid hourly bounds. A single known bound contributes only to that bound's country statistic. No salary is imputed. Fixed budgets remain total project amounts.
8. Flag 1.5-IQR compensation outliers independently by salary type. The optional compensation-only outlier filter removes flagged postings; show filtered denominators in interpretation.
9. Decode and normalize country labels with explicit aliases and `pycountry`. Missing locations become `Unknown`; unresolved labels stay visible and have no invented coordinates/ISO codes.
10. Apply the documented ordered regex taxonomy in `RULES`, with first match winning. Uncategorized titles remain `Other / Unclassified`. The rules and subcategories are inspectable in Data Quality.

## 6. Feature engineering

Required derived features are `job_title_clean`, `job_category`, `job_subcategory`, `year`, `month` (YYYY-MM), `week` (Monday date), `day`, `day_of_week`, `hourly_midpoint`, `salary_type`, `remote_indicator` and `job_keyword_features` (stopword-filtered title tokens). Additional fields hold validated bounds, fixed budget, cleaned country, ISO-3 country code, safe job URL and outlier flags. TF-IDF vectors are constructed sparsely when needed; vectors are not embedded into every DataFrame row.

## 7. Exploratory and statistical analysis

Market Overview presents six KPIs with interactive daily, category and payment charts. Filters apply inclusively to publication date, country, category and payment. Empty selections produce a message. Unknown dates are excluded by publication-date filtering and remain counted in the upload audit.

Compensation analysis separates hourly midpoints and fixed budgets throughout. It reports missingness, salary sample counts and mean/median by category. Distributions and box plots use a reproducible display sample of at most 15,000 records to keep the browser responsive; statistical summaries use all applicable records.

Keyword Spearman rho is computed between **binary term presence** and ranked observed compensation, not arbitrary title IDs. The vectorized calculation is mathematically equivalent to Pearson correlation between term presence and salary ranks. Two-sided p-values use the large-sample t approximation; Benjamini-Hochberg adjusts the family of eligible term tests. Require the chosen minimum sample count in both presence and absence groups. A Kruskal-Wallis omnibus test compares categories with sufficient observed salaries. It does not establish causal effects or identify specific differing pairs. A displayed p-value of zero means numerical underflow, not an impossible null hypothesis. Correlations between hourly bounds and their derived midpoint are mechanical and explicitly identified as such.

## 8. NLP methodology and classification

Title analysis uses English stop words plus a disclosed set of generic recruitment terms. Salary analysis uses unigram/bigram TF-IDF (maximum 1,500 features), keyword presence counts and mean TF-IDF. Category rules group similar title terms, but are not a supervised occupational classifier. Broad labels and overlap can misclassify mixed roles. No category accuracy score is claimed without labeled ground truth. Inspect the unclassified proportion before using category conclusions.

Emergence is measured with the latest two **adjacent complete Monday-Sunday weeks** within the modeling controls' window, with every global day observed and a minimum prior sample count. Rates with insufficient denominators remain missing. New-in-comparison categories are distinguished from supported growth; the system does not claim an occupation is new to the world. Posting share change in percentage points helps separate category growth from overall source-volume changes. Collection completeness remains unverified even when all days contain postings.

## 9. Recommendation system

The recommender fits word unigram/bigram TF-IDF (up to 24,000 features) and character-within-word 3-5-grams (up to 28,000 features, minimum document frequency 2) on unique normalized titles. Both use sublinear TF and float32 sparse matrices. Cosine similarity is computed by a query-to-corpus dot product of normalized vectors; no dense posting-by-posting similarity matrix is created.

`text_similarity = 0.65 * word_cosine + 0.35 * character_cosine`

`match_percentage = 100 * (0.90 * text_similarity + 0.10 * category_alignment)`

Category alignment uses an explicit preference or the query's first matching category rule. Category/country/payment preferences are hard filters. Salary bounds compare hourly **midpoint** or fixed **budget** only; records with unknown compensation are excluded only when compensation bounds are active. In the UI, choose one payment type before activating bounds. Text similarity must exceed 0.015, avoiding unrelated zero-match recommendations. Out-of-vocabulary queries can produce no results. A degenerate vocabulary can fall back to the other n-gram representation without fabricating scores.

Ranking ties prefer newer postings. Canonical URLs are unique, and repeated title/location/payment/compensation combinations are suppressed within each response. Repeated requests are deterministic, not a user-history exclusion service. Explanations list title term overlap, category alignment and active preferences. Match percentages describe lexical similarity and category alignment, **not** semantic understanding, employability, relevance probabilities or hiring chances. Historical URLs may no longer be active.

## 10. Machine learning and forecasting methodology

The target is a **daily posting count per category**, not an invented high-demand class. Logistic Regression is therefore not an appropriate count regressor. Candidates are Ridge, Random Forest, Extra Trees and HistGradientBoosting regression, compared with a weekly seasonal-naive baseline that repeats the value seven days ago. Fixed parameters and random state 42 keep computation modest. Hyperparameter search is intentionally avoided because this snapshot offers very few independent validation periods.

Features are lag-1, lag-7, trailing seven-day mean/standard deviation, change from the previous seven-day mean, weekday sine/cosine, elapsed day index and one-hot series identity. The first seven dates supply lag history. For fewer than fourteen historical dates, previous-week change is neutral rather than computed using unavailable data.

The suggested modeling window is the latest contiguous run above 15% of the median of the final 28 observed daily counts, excluding the last observed date. It is an editable coverage heuristic, not a guarantee of complete collection. For this dataset, it chooses **2024-02-13 to 2024-03-23 (40 days)**. Sparse earlier records remain in descriptive charts. Missing global dates block modeling; they are not converted to zero demand. Zero category counts on globally observed dates represent no observed postings in that category. Windows over 1,095 days or groupings over 40 series are rejected to bound computation.

## 11. Evaluation protocol and measured results

At least 35 consecutive observed days are required. Expanding validation origins start after 21 training days, advance by seven days and retain at most the last three valid origins. Each fold forecasts seven future days recursively without accessing actual observations within that horizon. A final **seven-day holdout** is excluded from model selection. MAE chooses the model using validation only; all candidates' holdout results remain visible for comparison. Future projections refit the selected model on all selected historical dates. Unchanged UI inputs reuse the previous session result; training is button-triggered.

The supplied 40-day default window supports **one** validation origin (training through March 4, validation March 5-11) and a separate holdout (training through March 16, holdout March 17-23). It does not support repeated-origin robustness claims. Values below were measured on the complete deduplicated dataset with default filters, outliers retained, ten category series and the pinned packages:

| Model | Validation MAE | Validation RMSE | Holdout MAE | Holdout RMSE |
|---|---:|---:|---:|---:|
| Weekly seasonal naive | 28.8571 | 43.1264 | 30.1571 | 43.1607 |
| Ridge | 50.9595 | 74.2923 | 61.2975 | 86.5105 |
| Random Forest | 38.4430 | 54.7652 | 27.1372 | 38.9288 |
| **Extra Trees (selected)** | **28.6251** | **38.9120** | **25.6183** | **37.0062** |
| HistGradientBoosting | 41.0910 | 59.0335 | 26.8923 | 39.3935 |

Errors are **postings per category per day**, pooled across series, not total-market daily error. Selected-model holdout pooled R² is 0.995684; category size differences make this number look strong even when per-category forecasting errors are material. It is not an accuracy percentage. Extra Trees improves holdout MAE over the baseline in this split, but the tiny validation advantage and short, irregular collection history do not establish sustained superiority. There are no labeled relevance judgments, so no recommendation precision/recall score is fabricated.

## 12. Future scenarios and uncertainty

Users can request the next seven or thirty dates following the **modeling window end**. An excluded final observed date can therefore overlap a projected date; the chart clearly separates the training history from projections. Predictions are nonnegative recursive scenarios, not actual future market data. For posting targets, the summary reports the first seven projected days and the selected-horizon total per series. For hourly-rate targets, inspect daily rates rather than summing rates.

Heuristic error bands use the validation residuals' 90th absolute-error quantile per series, widened by `sqrt(1 + forecast_day_index / 7)`. They are **not calibrated 90% confidence or prediction intervals**; with one validation fold, uncertainty is poorly estimated. Errors between categories are correlated, so bands must not be added as though independent.

The original data cannot support an independent 30-day accuracy estimate. With at least 65 consecutive days on a new upload, an additional historical 30-day diagnostic is shown, explicitly labeled as overlapping selection history rather than an independent holdout. A stronger long-horizon evaluation requires substantially more history and a reserved long-horizon test period.

Other targets reuse the same chronological protocol: total posting count, title-remote-signal / no-signal counts, and overall daily mean offered hourly midpoint. Missing daily salary observations block rate modeling instead of being imputed. Rate results concern changing offered-rate composition, not wage inflation. Remote volume predictions concern keyword frequency, not verified remote working arrangements.

## 13. Product shell, navigation and session behavior

All eleven page renderers reside in `app.py` and are registered as callable `st.Page` objects through `st.navigation`. Native `st.page_link` controls and `st.switch_page` provide real routes; no duplicated page scripts exist. The sidebar has collapsible navigation groups, an embedded logo, workspace identity and data-source status. CSS, branding and responsive styles are embedded in the Python file.

The common header owns global title/category search, UTC dates, a Filters popover (countries, categories, payment, remote-title proxy), Reset and Reprocess. Those controls persist while navigating. Search is a literal substring, not a hidden semantic search. Data Quality deliberately audits the full source independently of these filters. Salary outlier exclusion is local to descriptive compensation analysis and does not alter model-training counts.

Homepage KPI cards open their relevant destination. Selecting category/country points applies a shared filter and navigates to the detailed page. The directory supports title-cell and row selection to expose job details and uses bounded page payloads, while its download includes all matching rows. Drill-downs clear the directory's local title query and reset pagination; ordinary sidebar navigation retains the query. Displayed source URLs are HTTPS Upwork links. Quick links and sidebar entries use registered Streamlit pages.

Career matching runs only when requested. Results persist during save/remove actions, and changed preferences are labeled until the user requests fresh matches. Saved jobs are keyed by canonical source URL and live only in the current Streamlit session. Replacing the dataset resets filters, forecasts, results and saved jobs. An invalid replacement leaves the previously accepted dataset available. Reprocess reloads the currently accepted bytes and invalidates analytical result caches; it does not invent or fetch new observations.

The design uses navy/slate text, white cards, a pale background and restrained blue/green/amber accents. Material icons are functional. Tables and charts use source-unit labels, explicit denominators and bounded display samples. Plotly hover, zoom, selection and export controls remain available. Downloads are assembled in memory and formula-triggering text is escaped. No source data is interpolated into custom HTML without escaping.

## 14. Remote indicator

Case-insensitive title word/phrase matches detect `remote`, `work from home`, `wfh`, `virtual`, `online` and `anywhere`. The comparison label is **No title signal**, not verified non-remote. False positives include titles mentioning online products or virtual technologies; false negatives include remote roles without these words. The indicator cannot measure the true remote-work share of Upwork. Location and category distributions and hourly comparisons show the available proxy honestly.

## 15. Technology stack and resource behavior

Python 3.10+ locally (verified with 3.10), Streamlit 1.57, pandas 2.3.3, NumPy 2.2.6, SciPy 1.15.3, scikit-learn 1.7.2, Plotly 6.8 and pycountry 26.2.16. The Docker target uses Python 3.11. Verify your production Python/platform when deploying; the Docker image is not claimed tested unless built.

Upload parsing is cached in memory for 30 minutes with two entries maximum; accepted prepared data is reused by the current session between page changes. Recommendation indices use a two-entry, 30-minute process resource cache; these matrices are treated read-only. Title-keyword, salary and projection caches are keyed by the complete upload digest plus all relevant filters and parameters. Up to six compact forecast runs are retained per session for Model Performance; expensive fitting is button-triggered. No model pickle files exist. Low-cardinality derived columns use pandas categoricals and chart inputs are aggregated before rendering. The prepared full-data frame measured approximately 193 MiB (deep pandas accounting); total process memory is higher because caches, copies and NLP matrices coexist. Index construction and full-data processing can take tens of seconds on the first request; multiple simultaneous users can require substantially more memory. A small free host's capacity is not guaranteed. Process-level caches may be shared across sessions; this is intended for this public historical dataset, not private regulated uploads.

## 16. Installation

From a terminal, enter the delivered `TalentScope` directory. Install dependencies in an environment **outside** this directory to preserve the exact four-file constraint:

```powershell
python -m venv "$env:TEMP\talentscope-venv"
& "$env:TEMP\talentscope-venv\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
```

macOS/Linux alternative:

```bash
python3 -m venv /tmp/talentscope-venv
source /tmp/talentscope-venv/bin/activate
python -m pip install -r requirements.txt
```

If using an existing environment, only the last installation command is needed. Do not place a `.venv`, CSV, notebook or test file in the final project directory.

## 17. Local execution

```bash
python -B -m streamlit run app.py --server.headless=true --server.maxUploadSize=150 --browser.gatherUsageStats=false --theme.base=light --theme.primaryColor="#2563EB" --theme.backgroundColor="#F5F7FA" --theme.secondaryBackgroundColor="#FFFFFF" --theme.textColor="#172B46"
```

Open [localhost:8501](http://localhost:8501), upload the original CSV and inspect Data Quality. `-B` avoids Python bytecode directories during execution; for import-based API use, also set `PYTHONDONTWRITEBYTECODE=1` or run Python with `-B`. Streamlit/browser preferences can live in the user's home directory; the project does not require a fifth configuration file.

## 18. Docker

Run inside the four-file directory:

```bash
docker build -t talentscope .
docker run --rm -p 8501:8501 talentscope
```

Open [localhost:8501](http://localhost:8501) and upload the CSV. The image copies only the application, requirements and README, uses a non-root user, exposes 8501, starts headlessly and provides an HTTP health check using Python's standard library. No bind-mounted dataset is required. Inspect the container state with `docker ps`; after startup, health should become healthy. For restricted local access use `-p 127.0.0.1:8501:8501`. A hosted container service needs routing to port 8501 and websocket support; use the platform's managed HTTPS ingress. Docker was **not available** in the authoring environment, so image build and container execution remain unverified.

## 19. GitHub and Streamlit Community Cloud deployment

1. Create a GitHub repository named `talentscope`. To preserve the four-file contract, do not initialize an extra license, gitignore or README in the GitHub form.
2. Upload exactly `app.py`, `requirements.txt`, `Dockerfile` and this `README.md` at the repository root, or copy this directory's four files into an existing checkout. Do not upload the original CSV, environment, bytecode, model output or QA artifacts. Review the repository file list before committing. Confirm dataset terms before sharing any source data separately.
3. Sign into [Streamlit Community Cloud](https://share.streamlit.io/) with GitHub, choose **Create app**, and select your repository and branch. If prompted whether you already have an app, choose the existing-app option.
4. Set the entrypoint to **`app.py`**. In Advanced settings choose Python **3.11** to match the Docker target (or a supported compatible version), then deploy. No secrets or API keys are needed. `requirements.txt` sits beside the entrypoint and declares all direct dependencies.
5. Wait for dependency installation and startup. Read the build/runtime logs if it fails. Copy the actual assigned `https://…streamlit.app` URL only after the application loads.
6. Test the live URL: upload the original CSV, inspect Data Quality, visit all eleven pages, exercise filters, submit recommendations, run demand evaluation and download one results table. Repeat after uploading a different valid CSV; confirm no stale results remain. Clear restrictive filters if the modeling window becomes too short or discontinuous.

The instructions were checked against [official deployment documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy) and [official dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies). Community Cloud uses the Python application rather than this Dockerfile. UI labels and platform limits can change. **No repository was published and no public deployment URL was created or verified during delivery.**

## 20. Reusable Python API

These are importable Python functions, **not HTTP endpoints**. Pass a prepared DataFrame to analytics/recommendation/model functions; use `prepare_data` first. Returned tables are pandas DataFrames. Expected validation errors use `ValueError`. UI handlers report input/model errors and insufficient data without inventing results.

| Function | Parameters | Return value |
|---|---|---|
| `prepare_data(raw)` | Raw pandas DataFrame with the eight required fields | `(prepared_dataframe, audit_dict)`; original values retained on surviving deduplicated rows |
| `load_csv(content)` | UTF-8 CSV bytes | Cached `prepare_data` tuple; clear messages for empty/unparseable CSVs |
| `filter_jobs(data, start=None, end=None, categories=None, countries=None, payment='Any', min_rate=None, max_rate=None, min_budget=None, max_budget=None)` | Prepared data; inclusive UTC dates; optional lists of cleaned labels; payment enum; same-unit compensation limits | Copy of matching prepared rows. Missing salary excluded when salary limits active |
| `analyze_market(data)` | Prepared, optionally filtered DataFrame | Dict with `kpis`, `daily`, `monthly`, `categories` |
| `analyze_country_salary(data, min_samples=20)` | Prepared data and positive midpoint sample threshold | Country DataFrame with hourly posting count, mean low/high/midpoint, median midpoint, separate denominators, ISO code and `eligible_ranking` |
| `recommend_jobs(data, desired_title, category=None, country=None, payment='Any', min_hourly=None, max_hourly=None, min_budget=None, max_budget=None, top_k=10)` | Prepared data; nonempty title; optional cleaned category/country; payment enum; nonnegative bounds; top_k 1-100 | Matching prepared rows with `text_similarity`, `match_percentage`, `explanation`; empty DataFrame if no candidates. ValueError for empty title, invalid bounds/top_k |
| `predict_job_demand(data, horizon=30, start=None, end=None, group_col='job_category', value_col=None)` | Prepared data; horizon 7 or 30; optional UTC window; grouping column (None = total); optional numeric daily-mean target | `status='ok'` dict with `selected_model`, `metrics`, `history`, `forecast`, `holdout`, `importance`, window metadata, `validation_origins`, `diagnostic_30d`, `limitations`; otherwise `status='insufficient'` and `reason` |
| `salary_keywords(data, payment='Hourly', min_samples=30)` | Prepared data, Hourly or Fixed-price, minimum test/group samples | Dict with `keywords`, `correlation`, `test`, `n` |
| `category_growth(data, start=None, end=None, min_samples=30)` | Prepared data, optional dates and sample threshold | `(growth_dataframe, weekly_count_pivot)`; empty growth when comparison unsupported |

Example, run Python with `-B`:

```python
from pathlib import Path
import pandas as pd
from app import (prepare_data, analyze_market, analyze_country_salary,
                 recommend_jobs, predict_job_demand)

# Use your existing source file; no duplicate CSV is created.
raw = pd.read_csv(Path('/path/to/all_upwork_jobs_2024-02-07-2024-03-24.csv'))
jobs, audit = prepare_data(raw)
market = analyze_market(jobs)
countries = analyze_country_salary(jobs, min_samples=30)
matches = recommend_jobs(jobs, 'Python Data Engineer', payment='Hourly',
                         min_hourly=20, max_hourly=80, top_k=10)
projection = predict_job_demand(jobs, horizon=30,
                                start='2024-02-13', end='2024-03-23')
if projection['status'] == 'ok':
    print(projection['metrics'])
else:
    print(projection['reason'])
```

Handle `ValueError` for schema, bounds or window issues. An empty recommendation is valid and should trigger broader preferences, not fabricated rows. For fixed-price filtering, use `payment='Fixed-price', min_budget=..., max_budget=...`; never supply hourly and budget bounds together. Passing unprepared frames may raise normal pandas `KeyError`; the API contract requires `prepare_data`. Memory limits depend on upload size and concurrent sessions.

## 21. QA and limitations

Verification uses the actual source CSV, with test artifacts kept outside this four-file folder. The redesigned application passed 40 full-source Streamlit AppTest checkpoints covering all eleven pages, forecast generation and calculated model metrics, recommendation generation/save/remove/navigation persistence, preference clearing, combined global filters, dates, remote signals, empty states, all six KPI routes, fixed-price analysis, IQR exclusion and statistical associations. Callable-page hashes were preserved in the test harness so subsequent widget events exercised the selected page.

The browser accepted the original 48.2 MB CSV. All eleven page layouts were visually inspected. Real interactions verified pagination, title-cell posting details, recommendation explanations and saving, forecast evaluation, country-map selection (United States: 99,583 postings), category-bar selection (Design & Creative: 48,406 postings), sidebar collapse/reopen and malformed-upload feedback. Downloaded CSVs were read back and checked: 10 recommendation rows and 175 filtered “Data scientist” postings. The latter also exercised deferred export generation. Desktop (1280 × 720) and tablet (768 × 1024) layouts were inspected; the tablet header reflows and the navigation can collapse. QA corrected a hidden sidebar reopen control, delayed invalid-upload feedback, title-click behavior and cramped tablet filters. The final local HTTP health check returned `ok`.

Additional AppTest checks used 200 and 50 real source rows to verify title-cell details, invalid-replacement retention, valid-replacement reset and reprocessing of the accepted replacement. These are upload fixtures, not analytical substitutes for the full dataset. Existing analytical verification also covered malformed dates/titles, negative/non-finite/reversed compensation, duplicate rows, invalid payment flags, aliases, missing schemas, empty inputs, hourly/fixed/country/category/date filters, empty matches, safe-link handling and CSV formula escaping. All four forecast targets and detailed role groups returned finite projections. The final directory contains exactly the four required files. Docker build/run could not be exercised because Docker is absent; public deployment was not performed.

Brand QA parsed all six compact/full SVG combinations as self-contained XML and checked for external image/script references. Browser inspection verified the embedded SVG favicon, expanded-sidebar identity, collapsed compact header mark, landing preview, light/dark About panels, footer and white-logo loading banner during real source reprocessing. The branded landing was inspected at desktop and tablet widths. A fresh browser session confirmed disabled entry before upload; uploading 200 real source rows enabled entry and opened the corresponding workspace. Additional AppTest checks confirmed landing-upload validation, invalid replacement retention and filter persistence through Product home. Brand and screenshot QA artifacts are outside the four-file repository.

Material limits: short/uneven collection history; partial months; unknown currency; compensation missingness that may be nonrandom; variable fixed-project scope; offered rather than realized pay; title-only lexical recommendations with no labeled relevance evaluation; incomplete category coverage; imperfect remote proxy; registry-dependent geography; historical expired links; unverified collection completeness; one independent short holdout; uncalibrated future error bands; untested Docker runtime when Docker is unavailable. The application is portfolio/deployment-ready source, not a claim of production service reliability.

### Functional repair audit — October 30, 2026

The follow-up audit used the complete 244,828-row source, independent raw-CSV reconciliation, Streamlit AppTest and actual browser clicks. QA scripts and screenshots are outside the four-file application directory.

| Finding / root cause | Repair and verification |
|---|---|
| Sidebar had a source-status label but no KPI component | Added **Market Snapshot**, computed after current global filters. All seven requested counts/rates share one renderer; exact values also remain accessible through the main Overview expander when the sidebar is collapsed. Product home explicitly labels its snapshot as the entire source. |
| Unconnected analytical routes returned the unrelated welcome page | Each route now shows its own page title, a dataset-required explanation and working upload control. Uploading from Explorer continues into Explorer. Browser refresh/restart can create a new session; this is now explained. |
| Empty forecast dates could reach `len(None)` | Both global and forecast range controls accept cleared/incomplete selections and show guidance. Short modeling windows report insufficient history. |
| Persisted fixed-price salary preference conflicted with hourly workspace drill-downs | Salary cohort options follow the global payment constraint; the UI explains how to restore both cohorts. Tested Fixed-price → Hourly navigation. |
| Explicit widget defaults competed with persisted state | Date and career/pagination numeric controls initialize state once. Cleared compensation limits are explicitly rejected. |
| Analytical tables transferred every row and only scrolled | Tables above 100 rows now have numbered pages; all-row exports remain unchanged. Checked first, second and final partial pages. |
| Unexpected page errors gave only generic recovery advice | Error messages include the failing page, exception type and cause; full tracebacks remain logged. Missing rates display **Not available**, never an invented zero. |

Reconciliation: 244,828 raw records → 244,311 unique postings; 517 canonical-ID duplicates removed; 212 nonmissing country labels; 140,603 hourly and 103,708 fixed-price postings; 98,545 usable hourly midpoints; mean 29.7099396215, median 22.50 source units/hour; 8,193 remote-title signals. Country totals exclude Unknown. Remote status remains an estimate. Workspace metrics reflect header filters; local Explorer searches and Career preferences have their own result counts.

The original 40 full-source AppTest checkpoints passed again. An additional 46-check audit covered single/multiple countries and categories, all payment modes, complete/partial/empty date ranges, all pagination sizes, page changes, sorting, activity intervals, changed country scopes, three recommendation queries, hourly/fixed compensation limits, invalid bounds and empty titles, related titles, short-window refusal, seven-day forecasts, model metric selection, remote/rate outlooks and full-source reprocessing. Holdout MAE/RMSE/R² were recomputed from predictions and chronological validation/holdout separation asserted. A chart/table inventory checked 36 nonempty Plotly specifications and 19 populated tables across all eleven pages (including content inside tabs/expanders). This checks generated chart structure; it does not assert every possible filter combination or browser interaction. Additional repair checks exercised no-data routes, cleared numeric inputs, missing-rate KPIs and conflicting salary cohorts. Malformed-source checks covered invalid dates/payment types/numerics, reversed bounds, duplicates, aliases, unsafe links, required columns and CSV formula escaping.

The browser opened all eleven analytical destinations with the full source. Downloaded model evaluations (10 rows), recommendations (10 rows) and the filtered Data scientist directory (175 rows) were read back and checked. The sidebar updated to 175 postings, 30 countries, 139 hourly and 36 fixed-price jobs for that search. Desktop and 768-pixel tablet checks confirmed visible metrics and no horizontal overflow in the main snapshot; collapsed navigation retained the compact brand and main-dashboard metric access. The local server was restarted, the browser refreshed, the session recovery explanation checked and the full dataset reconnected. No browser application errors or new server Python errors appeared during normal interactions. The deliberate restart produced temporary browser health-check/disconnection messages while the server was stopped; the app reconnected successfully afterward. Automated model execution emitted a Windows physical-core detection warning and safely used logical cores; this did not prevent forecasting. Browser verification and operational limitations are reported separately from automated checks. A passing route or calculated chart does not establish forecast reliability, job availability, production capacity or compatibility with every browser/device. Original Upwork URLs are validated as HTTPS job links; historical listings may have expired. Browser session state is memory-only and is not durable storage.

## 22. Future improvements

Acquire a longer, documented collection history and labeled title taxonomy/relevance judgments; validate multilingual preprocessing; collect currency, descriptions, project duration and verified work-arrangement fields with permission; add true multi-horizon holdouts and interval calibration; evaluate per-category errors against business thresholds; measure recommendation precision at k; and benchmark memory/concurrency on the chosen deployment host. If the four-file restriction is relaxed, move to modules, automated CI, a persistent search index and explicit dependency lockfiles. These are future work, not implemented claims.
