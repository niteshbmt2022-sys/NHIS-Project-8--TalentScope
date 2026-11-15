"""TalentScope: freelance market intelligence, title matching and forecasting.

Importable functions have no UI side effects. Run with `streamlit run app.py`.
The analytical API and all callable product pages share this four-file package.
"""
from __future__ import annotations

import hashlib
import html
import io
import logging
import math
import re
import unicodedata
from contextlib import contextmanager
from urllib.parse import urlsplit, unquote

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pycountry
from scipy import sparse, stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.pairwise import linear_kernel
import streamlit as st

REQUIRED = ['title', 'link', 'published_date', 'is_hourly', 'hourly_low', 'hourly_high', 'budget', 'country']
# Ordered, first-match rules. No learned labels or undisclosed semantic claims.
RULES = [
 ('Data & AI', 'AI / Machine Learning', r'\b(ai|artificial intelligence|machine learning|deep learning|llm|gpt|chatgpt|nlp|computer vision)\b'),
 ('Data & AI', 'Data Engineering / Analytics', r'\b(data|analytics|analyst|tableau|power bi|sql|scrap\w*|statistic\w*)\b'),
 ('Development', 'Mobile Development', r'\b(ios|android|flutter|react native|mobile app|swift|kotlin)\b'),
 ('Development', 'Web Development', r'\b(web\w*|wordpress|shopify|react|frontend|front end|backend|back end|full stack|javascript|php|html|css|bubble|wix|squarespace)\b'),
 ('Development', 'Software / Automation', r'\b(develop\w*|programmer|software|python|java|coding|automation|api|bot|devops|cloud|aws|azure|linux|cybersecurity|blockchain)\b'),
 ('Design & Creative', 'Video / Animation', r'\b(video|animation|animator|youtube|reels|tiktok|motion|3d|blender|after effects|film)\b'),
 ('Design & Creative', 'Graphics / UX', r'\b(design\w*|logo|graphic\w*|illustrat\w*|figma|ux|ui|photoshop|photo\w*|canva|branding|artist)\b'),
 ('Sales & Marketing', 'Digital Marketing', r'\b(marketing|seo|ads|advertis\w*|social media|media buyer|facebook|google ads|email campaign|smma)\b'),
 ('Sales & Marketing', 'Sales / Lead Generation', r'\b(sales|lead generation|appointment setter|cold call\w*|outreach|business development)\b'),
 ('Writing & Translation', 'Translation / Languages', r'\b(translat\w*|transcri\w*|localization|subtitl\w*|proofread\w*)\b'),
 ('Writing & Translation', 'Writing / Content', r'\b(writ\w*|copywrit\w*|content|blog|article|editor|editing|scriptwriter|author)\b'),
 ('Admin & Support', 'Customer Support', r'\b(customer|support|service agent|call center)\b'),
 ('Admin & Support', 'Administrative', r'\b(assistant|virtual|admin\w*|data entry|excel|research|typing|spreadsheet)\b'),
 ('Finance & Legal', 'Finance / Accounting', r'\b(account\w*|bookkeep\w*|financ\w*|tax|quickbooks|cpa|audit)\b'),
 ('Finance & Legal', 'Legal', r'\b(legal|lawyer|attorney|contract|paralegal|law)\b'),
 ('Engineering', 'Engineering / Architecture', r'\b(engineer\w*|architect\w*|cad|autocad|solidworks|electrical|mechanical|civil|pcb)\b'),
 ('Education & HR', 'Education / People', r'\b(tutor|teach\w*|coach\w*|training|recruit\w*|human resources|hr|lesson|course)\b'),
]
REMOTE_RE = r'\b(remote|work from home|wfh|virtual|online|anywhere)\b'
EXTRA_STOP = {'need', 'needed', 'looking', 'seeking', 'expert', 'experienced', 'job', 'work', 'help', 'wanted', 'required', 'project', 'freelancer'}
STOP_WORDS = sorted(set(ENGLISH_STOP_WORDS) | EXTRA_STOP)
STOP_SET = frozenset(STOP_WORDS)
COUNTRY_ALIASES = {'usa': 'United States', 'us': 'United States', 'united states of america': 'United States',
 'uk': 'United Kingdom', 'uae': 'United Arab Emirates', 'south korea': 'Korea, Republic of',
 'north korea': "Korea, Democratic People's Republic of", 'russia': 'Russian Federation',
 'vietnam': 'Viet Nam', 'turkey': 'Türkiye', 'taiwan': 'Taiwan, Province of China',
 'palestinian territories': 'Palestine, State of', 'macedonia': 'North Macedonia',
 "cote d'ivoire": "Côte d'Ivoire", 'reunion': 'Réunion', 'curacao': 'Curaçao',
 'laos': "Lao People's Democratic Republic", 'swaziland': 'Eswatini',
 'british virgin islands': 'Virgin Islands, British', 'united states virgin islands': 'Virgin Islands, U.S.',
 'aland islands': 'Åland Islands', 'saint helena': 'Saint Helena, Ascension and Tristan da Cunha'}


def normalize_text(value) -> str:
    """Decode HTML, normalize Unicode and preserve useful technical tokens."""
    if pd.isna(value):
        return ''
    text = unicodedata.normalize('NFKC', html.unescape(str(value))).casefold()
    text = text.replace('c++', ' cpp ').replace('c#', ' csharp ').replace('.net', ' dotnet ')
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s+#.-]', ' ', text)).strip()


def country_details(value):
    if pd.isna(value) or not str(value).strip():
        return ('Unknown', None)
    clean = html.unescape(str(value)).strip()
    lookup = COUNTRY_ALIASES.get(clean.casefold(), clean)
    try:
        c = pycountry.countries.lookup(lookup)
        return c.name, c.alpha_3
    except LookupError:
        return clean, None  # Keep unresolved labels, omit them from maps.


def safe_job_url(value) -> str:
    """Only expose genuine HTTPS Upwork job links; uploaded text is untrusted."""
    try:
        u = urlsplit(str(value))
        if u.scheme == 'https' and u.hostname in {'upwork.com', 'www.upwork.com'} and u.path.startswith('/jobs/'):
            return u._replace(query='', fragment='').geturl()
    except ValueError:
        pass
    return ''


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate raw columns, retain their original values, add analysis features.

    Returns (cleaned_rows, audit). Invalid salary values become missing only in
    derived fields. Exact duplicates and canonical URL duplicates are removed.
    Invalid dates/titles remain visible in the audit and cleaned dataset.
    """
    missing = sorted(set(REQUIRED) - set(raw.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    if raw.empty:
        raise ValueError('The CSV contains no job postings.')
    d = raw.copy()
    audit = {'input_rows': len(d), 'input_dtypes': raw.dtypes.astype(str).to_dict(),
             'missing_original': raw[REQUIRED].isna().sum().to_dict(),
             'exact_duplicates': int(d.duplicated().sum())}
    d = d.drop_duplicates().copy()
    d['job_url'] = d.link.map(safe_job_url)
    # Upwork's ~ identifier is more stable than its human-readable URL slug.
    ids = d.job_url.map(lambda x: unquote(x).split('~')[-1] if '~' in unquote(x) else x)
    duplicate = ids.ne('') & ids.duplicated(keep='first')
    audit['duplicate_job_urls'] = int(duplicate.sum())
    d = d.loc[~duplicate].reset_index(drop=True)
    d['job_title_clean'] = d.title.map(normalize_text)
    d['date'] = pd.to_datetime(d.published_date, format='mixed', errors='coerce', utc=True).dt.tz_convert(None).dt.floor('D')
    audit['invalid_dates'] = int(d.date.isna().sum())
    audit['missing_titles'] = int(d.job_title_clean.eq('').sum())
    audit['unusable_urls'] = int(d.job_url.eq('').sum())
    hourly = d.is_hourly.astype('string').str.strip().str.lower().map({'true': True, 'false': False, '1': True, '0': False, '1.0': True, '0.0': False})
    d['salary_type'] = hourly.map({True: 'Hourly', False: 'Fixed-price'}).fillna('Unknown')
    audit['invalid_payment_type'] = int(hourly.isna().sum())
    nums = d[['hourly_low', 'hourly_high', 'budget']].apply(pd.to_numeric, errors='coerce')
    audit['invalid_numeric_or_nonpositive'] = {c: int((d[c].notna() & (~np.isfinite(nums[c]) | nums[c].le(0))).sum()) for c in nums}
    nums = nums.where(np.isfinite(nums) & nums.gt(0))
    reversed_bounds = nums.hourly_low.gt(nums.hourly_high)
    audit['reversed_hourly_bounds'] = int(reversed_bounds.sum())
    audit['payment_conflicts'] = int(((hourly.eq(True) & nums.budget.notna()) | (hourly.eq(False) & nums[['hourly_low', 'hourly_high']].notna().any(axis=1))).sum())
    d['hourly_low_valid'] = nums.hourly_low.where(hourly.eq(True) & ~reversed_bounds)
    d['hourly_high_valid'] = nums.hourly_high.where(hourly.eq(True) & ~reversed_bounds)
    d['hourly_midpoint'] = (d.hourly_low_valid + d.hourly_high_valid) / 2
    d['fixed_budget_valid'] = nums.budget.where(hourly.eq(False))
    audit['hourly_one_bound_only'] = int((hourly.eq(True) & (d.hourly_low_valid.notna() ^ d.hourly_high_valid.notna())).sum())
    audit['hourly_midpoints_available'] = int(d.hourly_midpoint.notna().sum())
    audit['outliers_iqr'] = {}
    for col in ['hourly_midpoint', 'fixed_budget_valid']:
        s = d[col].dropna()
        q1, q3 = s.quantile([.25, .75]) if len(s) else (np.nan, np.nan)
        d[col + '_outlier'] = d[col].gt(q3 + 1.5 * (q3 - q1)) | d[col].lt(q1 - 1.5 * (q3 - q1))
        audit['outliers_iqr'][col] = int(d[col + '_outlier'].sum())
    mapping = {x: country_details(x) for x in d.country.dropna().unique()}
    d['country_clean'] = d.country.map(lambda x: mapping.get(x, ('Unknown', None))[0])
    d['country_iso3'] = d.country.map(lambda x: mapping.get(x, ('Unknown', None))[1])
    audit['country_labels_normalized'] = {str(x): y[0] for x, y in mapping.items() if str(x) != y[0]}
    audit['unmapped_countries'] = sorted(d.loc[d.country_iso3.isna() & d.country_clean.ne('Unknown'), 'country_clean'].unique().tolist())
    d['job_category'], d['job_subcategory'] = 'Other / Unclassified', 'Other / Unclassified'
    for category, subcategory, pattern in RULES:
        hit = d.job_subcategory.eq('Other / Unclassified') & d.job_title_clean.str.contains(pattern.replace('(', '(?:'), regex=True)
        d.loc[hit, ['job_category', 'job_subcategory']] = [category, subcategory]
    for name, values in {'year': d.date.dt.year, 'month': d.date.dt.strftime('%Y-%m'),
                         'week': (d.date - pd.to_timedelta(d.date.dt.dayofweek, unit='D')).dt.strftime('%Y-%m-%d'),
                         'day': d.date.dt.day, 'day_of_week': d.date.dt.day_name()}.items():
        d[name] = values
    d['remote_indicator'] = d.job_title_clean.str.contains(REMOTE_RE.replace('(', '(?:'), regex=True)
    d['job_keyword_features'] = d.job_title_clean.map(lambda t: ' '.join(w for w in t.split() if w not in STOP_SET))
    # Low-cardinality derived fields avoid repeating strings on every posting.
    for col in ['country_clean', 'country_iso3', 'job_category', 'job_subcategory',
                'salary_type', 'month', 'week', 'day_of_week']:
        d[col] = d[col].astype('category')
    audit['output_rows'] = len(d)
    return d, audit


@st.cache_data(max_entries=2, ttl=1800, show_spinner=False)
def load_csv(content: bytes):
    """Cached uploader parser. No files are written to disk."""
    try:
        raw = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig', low_memory=False)
    except (UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError('Upload a nonempty UTF-8 CSV with the eight required columns.') from exc
    return prepare_data(raw)


def filter_jobs(d, start=None, end=None, categories=None, countries=None, payment='Any',
                min_rate=None, max_rate=None, min_budget=None, max_budget=None):
    """Inclusive date, category, country and same-unit compensation filters."""
    mask = pd.Series(True, index=d.index)
    if start is not None:
        mask &= d.date.ge(pd.Timestamp(start))
    if end is not None:
        mask &= d.date.le(pd.Timestamp(end))
    if categories:
        mask &= d.job_category.isin(categories)
    if countries:
        mask &= d.country_clean.isin(countries)
    if payment != 'Any':
        mask &= d.salary_type.eq(payment)
    if min_rate is not None or max_rate is not None:
        mask &= d.salary_type.eq('Hourly') & d.hourly_midpoint.between(min_rate or 0, np.inf if max_rate is None else max_rate)
    if min_budget is not None or max_budget is not None:
        mask &= d.salary_type.eq('Fixed-price') & d.fixed_budget_valid.between(min_budget or 0, np.inf if max_budget is None else max_budget)
    return d.loc[mask].copy()


def analyze_market(data: pd.DataFrame) -> dict:
    """Summarize prepared data; return KPI dict and daily/monthly/category tables."""
    valid = data.dropna(subset=['date'])
    monthly = valid.groupby('month', observed=True).agg(postings=('title', 'size'), hourly_rate=('hourly_midpoint', 'mean'),
                 salary_n=('hourly_midpoint', 'count'), remote_titles=('remote_indicator', 'sum')).reset_index()
    return {'kpis': {'postings': len(data), 'countries': data.loc[data.country_clean.ne('Unknown'), 'country_clean'].nunique(),
             'hourly_pct': data.salary_type.eq('Hourly').mean() * 100,
             'fixed_pct': data.salary_type.eq('Fixed-price').mean() * 100,
             'average_hourly': data.hourly_midpoint.mean(),
             'top_category': data.job_category.mode().iloc[0] if len(data) else 'None'},
            'daily': valid.groupby('date', observed=True).size().rename('postings').reset_index(), 'monthly': monthly,
            'categories': data.groupby('job_category', observed=True).size().rename('postings').sort_values(ascending=False).reset_index()}


def analyze_country_salary(data: pd.DataFrame, min_samples: int = 20) -> pd.DataFrame:
    """Hourly records only; preserve separate denominators for bounds/midpoints."""
    if min_samples < 1:
        raise ValueError('min_samples must be positive.')
    h = data.loc[data.salary_type.eq('Hourly')]
    result = h.groupby('country_clean', observed=True).agg(hourly_postings=('title', 'size'),
        mean_low=('hourly_low_valid', 'mean'), low_n=('hourly_low_valid', 'count'),
        mean_high=('hourly_high_valid', 'mean'), high_n=('hourly_high_valid', 'count'),
        mean_midpoint=('hourly_midpoint', 'mean'), median_midpoint=('hourly_midpoint', 'median'),
        salary_n=('hourly_midpoint', 'count'), country_iso3=('country_iso3', 'first')).reset_index()
    # An empty categorical aggregation can narrow codes while retaining a large
    # category vocabulary; string labels also make this API table portable.
    result['country_clean'] = result.country_clean.astype('string')
    result['eligible_ranking'] = result.salary_n.ge(min_samples) & result.country_clean.ne('Unknown')
    return result.sort_values('mean_midpoint', ascending=False)


def salary_keywords(data, payment='Hourly', min_samples=30):
    """Sparse TF-IDF, term presence Spearman tests, FDR and Kruskal-Wallis."""
    target = 'hourly_midpoint' if payment == 'Hourly' else 'fixed_budget_valid'
    d = data.loc[data.salary_type.eq(payment)].copy()
    corpus = d.job_title_clean.tolist()
    if len(d) < 3 or not any(corpus):
        return {'keywords': pd.DataFrame(), 'correlation': pd.DataFrame(), 'test': {}, 'n': 0}
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2,
                               max_features=1500, dtype=np.float32, token_pattern=r'(?u)\b\w[\w+#.-]+\b')
    try:
        x = vectorizer.fit_transform(corpus)
    except ValueError:
        return {'keywords': pd.DataFrame(), 'correlation': pd.DataFrame(), 'test': {}, 'n': 0}
    presence = x.copy(); presence.data[:] = 1
    observed = d[target].notna().to_numpy()
    y = d.loc[observed, target].to_numpy()
    p = presence[observed].tocsc()
    counts = np.asarray(p.sum(axis=0)).ravel()
    sums = np.asarray(p.T @ y).ravel()
    # Spearman of a binary presence feature and salary equals Pearson with salary ranks.
    ranks = stats.rankdata(y)
    n = len(y)
    with np.errstate(divide='ignore', invalid='ignore'):
        centered = np.asarray(p.T @ (ranks - ranks.mean())).ravel() if n else np.zeros(x.shape[1])
        denom = np.sqrt(counts * (1 - counts / max(n, 1)) * ((ranks - ranks.mean()) ** 2).sum())
        rho = centered / denom
        t = rho * np.sqrt((n - 2) / (1 - rho ** 2)) if n > 2 else np.full_like(rho, np.nan)
        pv = 2 * stats.t.sf(np.abs(t), max(1, n - 2))
    table = pd.DataFrame({'keyword': vectorizer.get_feature_names_out(),
        'postings': np.asarray(presence.sum(axis=0)).ravel().astype(int),
        'mean_tfidf': np.asarray(x.mean(axis=0)).ravel(), 'salary_n': counts.astype(int),
        'mean_salary': np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0),
        'spearman_rho': rho, 'p_value': pv})
    table.loc[(table.salary_n < min_samples) | ((n - table.salary_n) < min_samples), ['spearman_rho', 'p_value']] = np.nan
    table['q_value_bh'] = np.nan
    validp = table.p_value.dropna().sort_values()
    if len(validp):
        q = np.minimum.accumulate((validp.to_numpy() * len(validp) / np.arange(1, len(validp) + 1))[::-1])[::-1]
        table.loc[validp.index, 'q_value_bh'] = np.minimum(q, 1)
    groups = [g[target].dropna().to_numpy() for _, g in d.groupby('job_category', observed=True) if g[target].count() >= min_samples]
    test = {'groups': len(groups), 'salary_n': n, 'missing_salary': int(d[target].isna().sum())}
    if len(groups) >= 2 and np.unique(np.concatenate(groups)).size > 1:
        h, probability = stats.kruskal(*groups)
        test.update(H=float(h), p_value=float(probability))
    numeric = d[['hourly_low_valid', 'hourly_high_valid', 'hourly_midpoint']] if payment == 'Hourly' else d[['fixed_budget_valid', 'remote_indicator']].astype(float)
    return {'keywords': table.sort_values('postings', ascending=False),
            'correlation': numeric.corr(method='spearman'), 'test': test, 'n': n}


def category_growth(data, start=None, end=None, min_samples=30):
    """Compare adjacent complete Monday-Sunday weeks within an explicit window."""
    d = data.dropna(subset=['date'])
    if start is not None:
        d = d.loc[d.date.ge(pd.Timestamp(start))]
    if end is not None:
        d = d.loc[d.date.le(pd.Timestamp(end))]
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    lo, hi = d.date.min(), d.date.max()
    counts = d.groupby(['week', 'job_category'], observed=True).size().unstack(fill_value=0)
    full = [w for w in counts.index if pd.Timestamp(w) >= lo and pd.Timestamp(w) + pd.Timedelta(days=6) <= hi and
            d.loc[d.date.between(pd.Timestamp(w), pd.Timestamp(w) + pd.Timedelta(days=6)), 'date'].nunique() == 7]
    if len(full) < 2 or (pd.Timestamp(full[-1]) - pd.Timestamp(full[-2])).days != 7:
        return pd.DataFrame(), counts
    a, b = counts.loc[full[-2]], counts.loc[full[-1]]
    out = pd.DataFrame({'job_category': counts.columns, 'previous': a.values, 'recent': b.values})
    out['growth_pct'] = np.where(out.previous.ge(min_samples), 100 * (out.recent - out.previous) / out.previous.replace(0, np.nan), np.nan)
    out['share_change_pp'] = 100 * (out.recent / max(b.sum(), 1) - out.previous / max(a.sum(), 1))
    out['status'] = np.select([out.previous.eq(0) & out.recent.ge(min_samples),
        out.previous.ge(min_samples) & out.recent.ge(min_samples) & out.growth_pct.gt(0),
        out.previous.ge(min_samples) & out.growth_pct.lt(0)],
        ['Newly observed in comparison', 'Growing', 'Declining'], default='Insufficient / stable')
    out['comparison'] = full[-2] + ' vs ' + full[-1]
    return out.sort_values('growth_pct', ascending=False), counts


@st.cache_resource(max_entries=2, ttl=1800, show_spinner=False)
def build_recommendation_index(titles: tuple[str, ...]):
    """Fit on unique titles, retaining float32 sparse matrices; never dense N x N."""
    words = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), max_features=24000,
                           sublinear_tf=True, dtype=np.float32)
    chars = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2,
                           max_features=28000, sublinear_tf=True, dtype=np.float32)
    # Character vocabulary is still usable for corpora consisting of stop words.
    try:
        wm = words.fit_transform(titles)
    except ValueError:
        words = None; wm = sparse.csr_matrix((len(titles), 0), dtype=np.float32)
    try:
        cm = chars.fit_transform(titles)
    except ValueError:
        chars = None; cm = sparse.csr_matrix((len(titles), 0), dtype=np.float32)
    return words, wm, chars, cm


def recommend_jobs(data: pd.DataFrame, desired_title: str, category=None, country=None,
                   payment='Any', min_hourly=None, max_hourly=None, min_budget=None,
                   max_budget=None, top_k=10) -> pd.DataFrame:
    """Content ranking on titles, with hard preferences and a small category bonus.

    Match percentage is a similarity score, not an employment probability.
    Salary filters use midpoint (hourly) or total project budget (fixed).
    """
    query = normalize_text(desired_title)
    if not query:
        raise ValueError('Enter a desired job title.')
    if not 1 <= top_k <= 100:
        raise ValueError('top_k must be between 1 and 100.')
    if (min_hourly is not None or max_hourly is not None) and (min_budget is not None or max_budget is not None):
        raise ValueError('Use hourly bounds or fixed-budget bounds, not both.')
    for low, high in [(min_hourly, max_hourly), (min_budget, max_budget)]:
        if (low is not None and low < 0) or (high is not None and high < 0) or (low is not None and high is not None and low > high):
            raise ValueError('Salary bounds must be nonnegative and minimum must not exceed maximum.')
    d = filter_jobs(data, categories=[category] if category else None, countries=[country] if country else None,
        payment=payment, min_rate=min_hourly, max_rate=max_hourly, min_budget=min_budget, max_budget=max_budget)
    d = d.loc[d.job_title_clean.ne('') & d.job_url.ne('')].copy()
    if d.empty:
        return d
    # Index all supplied titles, independently of salary/country preferences, for stable IDF.
    titles = tuple(sorted(data.loc[data.job_title_clean.ne(''), 'job_title_clean'].unique()))
    word, wm, char, cm = build_recommendation_index(titles)
    score = np.zeros(len(titles), dtype=np.float32)
    if word is not None:
        score += .65 * linear_kernel(word.transform([query]), wm).ravel()
    if char is not None:
        score += .35 * linear_kernel(char.transform([query]), cm).ravel()
    lookup = pd.Series(score, index=titles)
    d['text_similarity'] = d.job_title_clean.map(lookup).fillna(0)
    d = d.loc[d.text_similarity.gt(.015)].copy()
    if d.empty:
        return d
    inferred = next((c for c, _, pattern in RULES if re.search(pattern, query)), None)
    preferred = category or inferred
    bonus = d.job_category.eq(preferred).astype(float) if preferred else 0
    d['match_percentage'] = 100 * (.9 * d.text_similarity + .1 * bonus)
    d = d.sort_values(['match_percentage', 'date'], ascending=[False, False])
    d = d.drop_duplicates('job_url').drop_duplicates(['job_title_clean', 'country_clean', 'salary_type', 'hourly_midpoint', 'fixed_budget_valid']).head(top_k)
    qtokens = set(query.split()) - set(STOP_WORDS)
    def explain(row):
        overlap = sorted(qtokens & set(row.job_title_clean.split()))
        reasons = ['shared title terms: ' + ', '.join(overlap)] if overlap else ['character n-gram title similarity']
        if row.job_category == preferred:
            reasons.append('category alignment')
        if country:
            reasons.append('country preference satisfied')
        if payment != 'Any':
            reasons.append('payment preference satisfied')
        return '; '.join(reasons)
    d['explanation'] = d.apply(explain, axis=1)
    return d


def suggest_model_window(data):
    """Heuristic latest dense contiguous run, excluding the possibly partial last day."""
    daily = data.dropna(subset=['date']).groupby('date', observed=True).size().sort_index()
    if daily.empty:
        return None, None, 0
    threshold = max(1, int(daily.tail(28).median() * .15))
    dense = daily.loc[(daily >= threshold) & (daily.index < daily.index.max())]
    if dense.empty:
        return daily.index.min(), daily.index.max(), threshold
    end = dense.index.max(); start = end
    while start - pd.Timedelta(days=1) in dense.index:
        start -= pd.Timedelta(days=1)
    return start, end, threshold


def _features(hist, date, epoch, categories):
    features = []
    for j in range(hist.shape[1]):
        last = hist[-7:, j]
        previous = hist[-14:-7, j] if len(hist) >= 14 else last
        features.append([hist[-1, j], hist[-7, j], last.mean(), last.std(), last.mean() - previous.mean(),
            np.sin(2 * np.pi * date.dayofweek / 7), np.cos(2 * np.pi * date.dayofweek / 7),
            (date - epoch).days] + [int(j == k) for k in range(len(categories))])
    return np.asarray(features, dtype=float)


def _new_model(name):
    if name == 'Random Forest':
        return RandomForestRegressor(n_estimators=80, max_depth=7, min_samples_leaf=3, random_state=42, n_jobs=2)
    if name == 'Extra Trees':
        return ExtraTreesRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=2)
    if name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=10, l2_regularization=5, random_state=42)
    return Ridge(alpha=10)


def _fit_predict(values, dates, future_dates, categories, name):
    hist = values.copy().astype(float)
    model = None
    if name != 'Seasonal naive (7d)':
        x = np.vstack([_features(hist[:i], dates[i], dates[0], categories) for i in range(7, len(hist))])
        y = hist[7:].reshape(-1)
        model = _new_model(name).fit(x, y)
    predictions = []
    for date in future_dates:
        pred = hist[-7] if model is None else model.predict(_features(hist, date, dates[0], categories))
        pred = np.maximum(pred, 0)
        predictions.append(pred)
        hist = np.vstack([hist, pred])
    return np.asarray(predictions), model


def _metrics(actual, predicted):
    a, p = np.asarray(actual).ravel(), np.asarray(predicted).ravel()
    return {'MAE': float(mean_absolute_error(a, p)), 'RMSE': float(np.sqrt(mean_squared_error(a, p))),
            'R2': float(r2_score(a, p)) if len(a) > 1 and np.var(a) > 0 else np.nan}


def predict_job_demand(data: pd.DataFrame, horizon=30, start=None, end=None, group_col='job_category',
                       value_col=None) -> dict:
    """Recursive daily panel regression with expanding-origin selection and holdout.

    Count forecasts use globally observed days only. Missing global dates block
    modeling rather than silently becoming zero. value_col produces daily means.
    At least 35 consecutive dates required. No accuracy claim for a 30-day horizon
    unless a separate historical 30-day backtest is feasible (>=65 days).
    """
    if horizon not in (7, 30):
        raise ValueError('horizon must be 7 or 30.')
    d = data.dropna(subset=['date']).copy()
    if start is None or end is None:
        auto_start, auto_end, _ = suggest_model_window(d)
        start = start if start is not None else auto_start
        end = end if end is not None else auto_end
    if start is None or end is None:
        return {'status': 'insufficient', 'reason': 'No valid dates.'}
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError('Model start must precede model end.')
    d = d.loc[d.date.between(start, end)]
    dates = pd.date_range(start, end, freq='D')
    if len(dates) < 35:
        return {'status': 'insufficient', 'reason': 'At least 35 consecutive observed days are required for training, validation and a 7-day holdout.'}
    if len(dates) > 1095:
        return {'status': 'insufficient', 'reason': 'Select a modeling window of at most 1,095 days.'}
    if not dates.isin(d.date.unique()).all():
        return {'status': 'insufficient', 'reason': 'The selected window has globally unobserved dates. Choose a contiguous collection window; missing dates are not zero demand.'}
    if group_col is None:
        d['_series'] = 'All postings'; group_col = '_series'
    if value_col:
        panel = d.groupby(['date', group_col], observed=True)[value_col].mean().unstack().reindex(dates)
        if panel.isna().any().any():
            return {'status': 'insufficient', 'reason': 'Daily salary observations are missing in this window; no salary values have been imputed.'}
    else:
        panel = d.groupby(['date', group_col], observed=True).size().unstack(fill_value=0).reindex(dates, fill_value=0)
    categories = panel.columns.astype(str).tolist()
    values = panel.to_numpy(dtype=float)
    if len(categories) > 40:
        raise ValueError('Select a grouping with at most 40 series.')
    n = len(dates); holdout_start = n - 7
    origins = list(range(21, holdout_start - 6, 7))[-3:]
    names = ['Seasonal naive (7d)', 'Ridge', 'Random Forest', 'Extra Trees', 'HistGradientBoosting']
    records, residuals = [], {}
    for name in names:
        aa, pp = [], []
        for origin in origins:
            pred, _ = _fit_predict(values[:origin], dates[:origin], dates[origin:origin + 7], categories, name)
            aa.append(values[origin:origin + 7]); pp.append(pred)
        actual, predicted = np.vstack(aa), np.vstack(pp)
        residuals[name] = actual - predicted
        records.append({'model': name, 'stage': 'Rolling validation (7d)', 'folds': len(origins),
                        'train_end': str(dates[origins[-1] - 1].date()), **_metrics(actual, predicted)})
    selection = pd.DataFrame(records).sort_values('MAE')
    best = selection.iloc[0]['model']
    hold_records, actual_predicted = [], []
    for name in names:
        pred, _ = _fit_predict(values[:holdout_start], dates[:holdout_start], dates[holdout_start:], categories, name)
        hold_records.append({'model': name, 'stage': 'Untouched holdout (7d)', 'folds': 1,
                             'train_end': str(dates[holdout_start - 1].date()), **_metrics(values[holdout_start:], pred)})
        if name == best:
            for i, date in enumerate(dates[holdout_start:]):
                for j, c in enumerate(categories):
                    actual_predicted.append({'date': date, 'series': c, 'actual': values[holdout_start + i, j], 'predicted': pred[i, j]})
    future_dates = pd.date_range(end + pd.Timedelta(days=1), periods=horizon)
    predictions, model = _fit_predict(values, dates, future_dates, categories, best)
    # Descriptive absolute-error bands, calibrated before holdout. Not nominal CIs.
    width = np.quantile(np.abs(residuals[best]), .9, axis=0)
    future = []
    for i, date in enumerate(future_dates):
        for j, c in enumerate(categories):
            w = width[j] * np.sqrt(1 + i / 7)
            future.append({'date': date, 'series': c, 'projection': predictions[i, j],
                           'lower_error_band': max(0, predictions[i, j] - w), 'upper_error_band': predictions[i, j] + w})
    long_validation = None
    if n >= 65:
        lp, _ = _fit_predict(values[:-30], dates[:-30], dates[-30:], categories, best)
        bp, _ = _fit_predict(values[:-30], dates[:-30], dates[-30:], categories, names[0])
        long_validation = pd.DataFrame([{'model': best, **_metrics(values[-30:], lp)}, {'model': names[0], **_metrics(values[-30:], bp)}])
        # Model choice uses observations within this interval; call this diagnostic.
    importance = None
    if model is not None and hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({'feature': ['lag1', 'lag7', 'mean7', 'std7', 'change7', 'weekday_sin', 'weekday_cos', 'time_index'] + categories,
                                   'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    history = panel.rename_axis(index='date', columns='series').stack().rename('actual').reset_index()
    return {'status': 'ok', 'selected_model': best, 'metrics': pd.concat([selection, pd.DataFrame(hold_records)], ignore_index=True),
        'forecast': pd.DataFrame(future), 'history': history, 'holdout': pd.DataFrame(actual_predicted),
        'importance': importance, 'window_start': start, 'window_end': end, 'days': n,
        'validation_origins': [str(dates[i].date()) for i in origins], 'diagnostic_30d': long_validation,
        'limitations': 'Collection completeness is unverified. Forecasts describe this CSV stream, not the whole labor market. '
          'Error bands are heuristic, correlated across categories and not calibrated confidence intervals. '
          + ('No historical 30-day accuracy estimate: insufficient history.' if n < 65 else '30-day diagnostic overlaps selection history and is not an independent holdout.')}


def csv_bytes(frame):
    """In-memory export with spreadsheet formula injection mitigation."""
    out = frame.copy()
    for col in out.select_dtypes(include=['object', 'string']).columns:
        out[col] = out[col].map(lambda x: "'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@', '\t', '\r')) else x)
    return out.to_csv(index=False).encode('utf-8-sig')


# ---------------------------------------------------------------------------
# TalentScope product shell and reusable presentation components
# ---------------------------------------------------------------------------
BLUE, NAVY, GREEN, AMBER = '#2563EB', '#14243A', '#16A36A', '#D99A27'
PALETTE = [BLUE, '#7096E8', GREEN, '#718096', AMBER, '#6673AC', '#4E9EA6']
px.defaults.color_discrete_sequence = PALETTE


def brand_svg(variant='dark', compact=False, descriptor=True):
    """Original Orbit / Opportunity identity. All artwork lives in this source file.

    The globe's meridians establish global reach. A rising route terminates in
    a geometric briefcase node: an opportunity revealed through intelligence.
    The same geometry is used for every lockup and the browser favicon.
    """
    white = variant in ('white', 'badge')
    ink, accent = ('#FFFFFF', '#91B8FF') if white else (NAVY, BLUE)
    ground = NAVY if variant == 'badge' else ('#14243A' if white else '#FFFFFF')
    badge = '<rect width="76" height="76" rx="18" fill="#14243A"/>' if variant == 'badge' else ''
    geometry = f'''<g fill="none" stroke-linecap="round" stroke-linejoin="round">
<circle cx="34" cy="40" r="25" stroke="{ink}" stroke-width="3"/>
<path d="M9 40h50M34 15c-16 12-16 38 0 50M34 15c16 12 16 38 0 50" stroke="{ink}" stroke-width="1.5" opacity=".42"/>
<path d="M14 53l15-17 10 7 16-22" stroke="{ground}" stroke-width="8"/>
<path d="M14 53l15-17 10 7 16-22" stroke="{accent}" stroke-width="3.8"/>
<path d="M48 12V9a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v3" stroke="{accent}" stroke-width="2.4"/>
<rect x="44" y="12" width="20" height="16" rx="3" fill="{ground}" stroke="{accent}" stroke-width="2.7"/>
<path d="M45 18h18M54 17v4" stroke="{accent}" stroke-width="2"/>
</g>'''
    wordmark = ''
    if not compact:
        wordmark = f'<text x="84" y="39" fill="{ink}" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="32" font-weight="650" letter-spacing="-1">TalentScope</text>'
        if descriptor:
            wordmark += f'<text x="85" y="57" fill="{ink}" opacity=".72" font-family="Inter, Segoe UI, Arial, sans-serif" font-size="8.6" font-weight="600" letter-spacing=".75">FREELANCE MARKET INTELLIGENCE</text>'
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {76 if compact else 300} 76" role="img" aria-label="TalentScope — Freelance Market Intelligence" data-ts-logo="{variant}" preserveAspectRatio="xMinYMid meet"><title>TalentScope — Freelance Market Intelligence</title>{badge}{geometry}{wordmark}</svg>'


@contextmanager
def branded_loading(message):
    """Show the real processing state with the white brand lockup."""
    status = st.empty()
    status.markdown(f'<div class="ts-loading">{brand_svg("white")}<div><strong>Turning data into perspective</strong><span>{esc(message)}</span></div></div>', unsafe_allow_html=True)
    try:
        with st.spinner(message):
            yield
    finally:
        status.empty()


def enter_workspace():
    request_route('overview')


def brand_footer():
    with st.expander('About TalentScope'):
        st.markdown('<div class="ts-about-grid"><div class="ts-about-light">' + brand_svg() + '<h3>A wider view of opportunity.</h3><p>Explore the global freelance economy through historical postings, transparent compensation comparisons and explainable title matching.</p></div><div class="ts-about-dark">' + brand_svg('white') + '<h3>Global perspective. Clearer decisions.</h3><p>Discover Opportunities. Understand Markets. Make Smarter Decisions.</p></div></div>', unsafe_allow_html=True)
        st.caption('The globe represents global reach. Its ascending route connects to an opportunity node, shaped as a briefcase. Categories and remote signals are inferred; forecasts are exploratory. TalentScope is independent of Upwork.')
    st.markdown('<div class="ts-footer"><div class="ts-footer-brand">' + brand_svg(compact=True) + '<span><b>TalentScope</b><br>Freelance Market Intelligence</span></div><span>Historical observations.<br>Informed decisions.</span></div>', unsafe_allow_html=True)


PRODUCT_CSS = '''<style>
:root {--ts-ink:#172B46;--ts-muted:#718096;--ts-line:#E3E9F1;--ts-blue:#2563EB;}
html,body,.stApp {font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#172B46;background:#F5F7FA;}
[data-testid="stAppViewContainer"] {background:#F5F7FA;}
[data-testid="stHeader"] {background:rgba(245,247,250,.96);height:2.7rem;border-bottom:1px solid #e8edf3;}
[data-testid="stAppDeployButton"],[data-testid="stMainMenu"],[data-testid="stToolbarActions"] {display:none;}
.block-container {max-width:1510px;padding:3.35rem 2.2rem 2rem;}
[data-testid="stMain"] h1 {font-size:1.72rem;letter-spacing:-.045rem;line-height:1.25;font-weight:650;color:#14243A;padding:0 0 .45rem;}
[data-testid="stMain"] h2 {font-size:1.13rem;letter-spacing:-.02rem;font-weight:650;color:#172B46;}
[data-testid="stMain"] h3 {font-size:1.02rem;font-weight:650;color:#172B46;padding-top:.15rem;}
[data-testid="stMain"] p {font-size:.88rem;line-height:1.55;}
[data-testid="stMain"] label,[data-testid="stMain"] [data-testid="stMarkdownContainer"] {color:#172B46;}
[data-testid="stCaptionContainer"] p {color:#64748B !important;font-size:.77rem;}
[data-testid="stSidebar"] {background:#FFFFFF;border-right:1px solid #E3E9F1;}
[data-testid="stSidebar"][aria-expanded="true"] {min-width:255px;max-width:255px;}
[data-testid="stSidebar"][aria-expanded="false"] {min-width:0;max-width:0;border:0;}
[data-testid="stSidebarContent"] {padding-top:.3rem;}
[data-testid="stSidebarCollapseButton"] {visibility:visible !important;opacity:1 !important;}
[data-testid="stSidebarUserContent"] {padding:1.1rem 1rem 1rem;}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] summary {color:#475569;}
[data-testid="stSidebar"] [data-testid="stPageLink"] {margin:0 -.15rem;}
[data-testid="stSidebar"] [data-testid="stPageLink"] a {border-radius:7px;padding:.54rem .6rem;min-height:37px;color:#526176;font-size:.83rem;}
[data-testid="stSidebar"] [data-testid="stPageLink"] p {font-size:.79rem;line-height:1.25;white-space:normal;}
[data-testid="stSidebar"] [data-testid="stIconMaterial"] {font-size:1.15rem;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.45rem;}
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {background:#F2F6FD;color:#2563EB;}
[data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {background:#EDF3FF;color:#2563EB;font-weight:600;}
[data-testid="stSidebar"] [data-testid="stExpander"] {border:0;background:transparent;}
[data-testid="stSidebar"] [data-testid="stExpander"] details {border:0;background:transparent;}
[data-testid="stSidebar"] [data-testid="stExpander"] details summary {padding:.2rem 0;}
[data-testid="stSidebar"] [data-testid="stExpander"] details summary p {font-size:.62rem;font-weight:700;letter-spacing:.09rem;text-transform:uppercase;color:#94A0B1;}
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {padding:.1rem 0 .5rem;}
[data-testid="stSidebar"] [data-testid="stExpander"] details>div {padding:.1rem 0 .4rem;}
.ts-brand {padding:0 0 14px;}.ts-brand svg{display:block;width:100%;height:auto;}.ts-brand-descriptor{display:block;font-size:.49rem;letter-spacing:.045rem;font-weight:600;color:#64748B;white-space:nowrap;}
.ts-header-logo{position:fixed;z-index:1000000;top:3px;left:273px;height:36px;pointer-events:none;}
.ts-header-logo svg{height:36px;width:143px;}.ts-header-compact{display:none;}
.ts-header-compact svg{width:36px;}
body:has([data-testid="stSidebar"][aria-expanded="false"]) .ts-header-logo{left:58px;}
body:has([data-testid="stSidebar"][aria-expanded="false"]) .ts-header-full{display:none;}
body:has([data-testid="stSidebar"][aria-expanded="false"]) .ts-header-compact{display:block;width:36px;}
.ts-footer{align-items:center;gap:20px;}.ts-footer-brand{display:flex;align-items:center;gap:9px;}.ts-footer-brand svg{width:35px;height:35px;}.ts-footer-brand b{color:#334155;font-weight:650;}
.ts-loading{display:flex;align-items:center;gap:24px;background:#14243A;border:1px solid #233C5C;border-radius:10px;padding:18px 24px;margin:12px 0;color:white;}
.ts-loading svg{width:190px;flex:none;}.ts-loading strong{display:block;font-size:.86rem;}.ts-loading span{display:block;color:#B8CBE2;font-size:.77rem;margin-top:5px;}
.ts-about-grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;}.ts-about-light,.ts-about-dark{padding:24px;border-radius:9px;border:1px solid #E3E9F1;}.ts-about-light{background:white;}.ts-about-dark{background:#14243A;border-color:#14243A;}.ts-about-grid svg{width:min(100%,280px);height:auto;}.ts-about-dark h3,.ts-about-dark p{color:#E4EDFB !important;}.ts-about-grid p{font-size:.82rem !important;}
.ts-landing-hero{padding:16px 0 20px;max-width:680px;}.ts-landing-hero>svg{display:block;width:310px;max-width:100%;height:auto;margin:0 0 28px;}
.ts-eyebrow{color:#2563EB;font-size:.68rem;font-weight:700;letter-spacing:.12rem;text-transform:uppercase;margin-bottom:16px;}
[data-testid="stMain"] .ts-landing-hero h1{font-size:clamp(2rem,3.2vw,3.15rem);letter-spacing:-.07rem;line-height:1.1;max-width:610px;margin-bottom:22px;}
[data-testid="stMain"] .ts-landing-hero p{color:#64748B;max-width:550px;font-size:1rem;line-height:1.7;}.ts-landing-tagline{font-size:.72rem !important;font-weight:600;color:#334155 !important;margin-top:26px;}
.ts-preview{background:#14243A;border:1px solid #263E5B;border-radius:16px;padding:28px;color:#fff;margin:20px 0;box-shadow:0 16px 36px #14243a12;}
.ts-preview-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:24px;}.ts-preview-top svg{height:46px;width:46px;}.ts-preview-label{font-size:.63rem;letter-spacing:.1rem;color:#B5C8E4;text-transform:uppercase;}
[data-testid="stMain"] .ts-preview h2{color:#fff;font-size:1.35rem;line-height:1.35;max-width:300px;}.ts-preview p{color:#B5C8E4 !important;font-size:.77rem !important;}
.ts-preview-kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:9px;margin:22px 0;}.ts-preview-kpis>div{background:#203750;border:1px solid #35506D;border-radius:8px;padding:12px 9px;}.ts-preview-kpis b{font-size:1.15rem;display:block;}.ts-preview-kpis span{font-size:.59rem;color:#B5C8E4;}
.ts-preview-row{margin:13px 0;font-size:.72rem;}.ts-preview-row>div:first-child{display:flex;justify-content:space-between;gap:10px;margin-bottom:7px;}.ts-preview-track{height:5px;background:#314762;border-radius:3px;}.ts-preview-track i{display:block;height:5px;background:#91B8FF;border-radius:3px;}
.ts-preview-empty{border:1px dashed #49627E;border-radius:9px;padding:22px 16px;margin:20px 0;}.ts-preview-empty b{font-size:.8rem;}.ts-preview-meta{border-top:1px solid #35506D;padding-top:15px;margin-top:22px;font-size:.67rem;color:#B5C8E4;}
.ts-capability-symbol{width:32px;height:32px;color:#2563EB;margin-bottom:12px;}.ts-capability-symbol svg{width:100%;height:100%;}
@media(max-width:800px){.ts-about-grid{grid-template-columns:1fr;}.ts-loading{flex-wrap:wrap;gap:10px;}.ts-loading svg{width:150px;}.ts-preview{padding:22px;}.ts-landing-hero{padding-top:5px;}.ts-landing-hero>svg{width:270px;margin-bottom:20px;}}
.ts-workspace {margin:0 3px 17px;padding:10px 11px;border:1px solid #E5EAF1;border-radius:7px;font-size:.74rem;color:#536378;background:#F9FBFE;}
.ts-workspace b {color:#172B46;display:block;font-size:.77rem;margin-top:3px;}
.ts-breadcrumb {color:#8793A5;font-size:.7rem;letter-spacing:.02rem;margin:0 0 9px;}
.ts-breadcrumb span {color:#334155;}.ts-subtitle {color:#718096;font-size:.86rem;margin:0 0 1.2rem;}
.ts-status {display:inline-flex;align-items:center;gap:6px;background:#EDF8F2;color:#16744E;border:1px solid #D4ECDF;border-radius:30px;padding:5px 9px;font-size:.68rem;font-weight:600;}
.ts-status i {height:5px;width:5px;border-radius:50%;background:#16A36A;}.ts-muted-status{background:#F3F5F8;color:#718096;border-color:#E3E9F1;}
.ts-source {border:1px solid #E3E9F1;border-radius:8px;padding:12px;background:#F8FAFD;font-size:.72rem;color:#718096;}.ts-source b{display:block;color:#334155;font-size:.77rem;margin:6px 0;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-color:#E3E9F1 !important;border-radius:10px;}
[class*="st-key-panel_"] {background:white;border:1px solid #E3E9F1;border-radius:11px;padding:16px 18px;box-shadow:0 2px 4px #172b4603;}
[class*="st-key-kpi_"] button {background:#FFF !important;border:1px solid #E3E9F1 !important;border-radius:10px !important;min-height:109px;align-items:flex-start;justify-content:flex-start;padding:15px 17px !important;box-shadow:0 2px 4px #172b4603;transition:border-color .15s;}
[class*="st-key-kpi_"] button:hover {border-color:#87A9F3 !important;box-shadow:0 3px 10px #2563eb0b;}
[class*="st-key-kpi_"] button p {text-align:left;color:#718096 !important;font-size:.72rem !important;margin:0;line-height:1.4;}
[class*="st-key-kpi_"] button p:first-child {font-size:1.65rem !important;font-weight:650;color:#172B46 !important;letter-spacing:-.04rem;margin-bottom:7px;}
[data-testid="stMetric"] {padding:13px 16px;border:1px solid #E3E9F1;border-radius:9px;background:#FFF;}
[data-testid="stMetric"] * {color:#172B46 !important;}
[data-testid="stMetricLabel"] p {font-size:.74rem !important;color:#718096 !important;}
[data-testid="stMetricValue"] {font-size:1.65rem;font-weight:600;white-space:normal;line-height:1.2;}
[data-testid="stMetricValue"] div {white-space:normal;overflow-wrap:anywhere;}
[data-testid="stBaseButton-primary"] {background:#2563EB !important;border:1px solid #2563EB !important;color:white !important;border-radius:7px;}
[data-testid="stBaseButton-primary"] * {color:white !important;}
[data-testid="stBaseButton-primary"]:disabled {background:#DFE6F0 !important;border-color:#DFE6F0 !important;opacity:1;}
[data-testid="stBaseButton-primary"]:disabled * {color:#65758A !important;}
[data-testid="stBaseButton-secondary"],[data-testid="stBaseButton-tertiary"],[data-testid="stBaseButton-pageLink"],[data-testid="stBaseButton-download"] {border-color:#DEE5EF;background:#FFF;color:#334155;border-radius:7px;}
[data-testid="stMain"] button:hover {border-color:#8EADF2;}
[data-testid="stMain"] input,[data-testid="stMain"] [data-baseweb="select"]>div,[data-testid="stSidebar"] input {background:#FFF !important;color:#172B46 !important;}
[data-baseweb="input"],[data-baseweb="base-input"],[data-baseweb="select"]>div {background:#FFF !important;color:#172B46;border-color:#DDE5EF !important;}
[data-baseweb="input"] input,[data-baseweb="select"] input {color:#172B46 !important;caret-color:#2563EB;}
[data-baseweb="select"] span,[data-baseweb="select"] svg {color:#526176;fill:#526176;}
[data-baseweb="tag"] {background:#EDF3FF;color:#2563EB;}
[data-testid="stTextInput"] input::placeholder {color:#97A2B1 !important;}
[data-testid="stTabs"] [data-baseweb="tab-list"] {gap:20px;border-bottom:1px solid #E3E9F1;}
[data-testid="stTabs"] [data-baseweb="tab"] {background:transparent;color:#718096;font-size:.82rem;}
[data-testid="stTabs"] [aria-selected="true"] {color:#2563EB !important;}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {background:#2563EB;}
[data-testid="stExpander"] {border-color:#E3E9F1;background:#FFF;border-radius:9px;}
[data-testid="stExpander"] summary {color:#334155;}
.ts-note {background:#F0F5FF;border:1px solid #DCE7FC;border-left:3px solid #7399EA;border-radius:6px;padding:11px 13px;font-size:.77rem;color:#536987;line-height:1.5;margin:5px 0 16px;}
.ts-note.warning {background:#FFFAEE;border-color:#F2E7C9;border-left-color:#D99A27;color:#806C42;}
.ts-highlight {border-left:2px solid #2563EB;padding:2px 0 2px 12px;margin:8px 0 4px;}
.ts-highlight b {color:#172B46;font-size:.86rem;font-weight:600;display:block;}.ts-highlight p {font-size:.78rem !important;color:#718096;margin:5px 0;}
.ts-empty {padding:38px 25px;background:#FFF;border:1px dashed #CFD9E7;border-radius:11px;text-align:center;}.ts-empty h3{margin:0;color:#334155;}.ts-empty p{color:#718096;max-width:520px;margin:10px auto;}
.ts-score {float:right;padding:5px 9px;border-radius:6px;background:#ECF3FF;color:#2563EB;font-size:.8rem;font-weight:650;}
.ts-job-title {font-size:1rem;font-weight:600;color:#172B46;line-height:1.4;margin-right:90px;}
.ts-job-meta {font-size:.75rem;color:#718096;margin:9px 0 12px;line-height:1.6;}
.ts-comp {font-weight:600;color:#334155;font-size:.9rem;margin:5px 0;}
.ts-footer {border-top:1px solid #E3E9F1;color:#8A96A7;font-size:.7rem;padding-top:15px;margin-top:30px;display:flex;justify-content:space-between;}
.ts-upload-intro {background:#FFF;border:1px solid #E3E9F1;padding:34px;border-radius:12px;max-width:850px;margin:20px auto;}
@media(max-width:1000px){.block-container{padding-left:1.2rem;padding-right:1.2rem;}[class*="st-key-kpi_"] button{padding:12px !important;}[class*="st-key-kpi_"] button p:first-child{font-size:1.4rem !important;}}
@media(max-width:1100px){.block-container{padding:3rem 1rem 1rem;}[data-testid="stMain"] h1{font-size:1.5rem;}[data-testid="stHorizontalBlock"]{flex-wrap:wrap;}[data-testid="stHorizontalBlock"]>[data-testid="stColumn"]{min-width:min(100%,230px);flex:1 1 230px;}[class*="st-key-kpi_"] button{min-height:94px;}.ts-footer{display:block;}.ts-score{float:none;display:inline-block;margin-bottom:7px;}.ts-job-title{margin-right:0;}}
@media(max-width:1100px){.st-key-global_filters [data-testid="stColumn"]:nth-child(-n+2){min-width:min(100%,230px);flex:1 1 calc(50% - 8px);}.st-key-global_filters [data-testid="stColumn"]:nth-child(n+3){min-width:100px;flex:1 1 100px;}}
</style>'''


def esc(value):
    return html.escape(str(value))


def number(value, decimals=0):
    return 'Not available' if pd.isna(value) else f'{value:,.{decimals}f}'


def market_snapshot(data, scope='Current workspace filters'):
    """One calculated KPI definition shared by sidebar and main dashboard."""
    items = [('Unique postings', len(data)),
             ('Countries', data.loc[data.country_clean.ne('Unknown'), 'country_clean'].nunique()),
             ('Hourly postings', int(data.salary_type.eq('Hourly').sum())),
             ('Fixed-price postings', int(data.salary_type.eq('Fixed-price').sum())),
             ('Mean hourly rate', data.hourly_midpoint.mean()),
             ('Median hourly rate', data.hourly_midpoint.dropna().median() if data.hourly_midpoint.notna().any() else np.nan),
             ('Remote title signals', int(data.remote_indicator.sum()))]
    rows = ''.join(f'<div class="ts-snapshot-row"><span>{label}</span><strong>{number(value, 2 if "rate" in label else 0)}</strong></div>' for label, value in items)
    st.markdown(f'<section class="ts-snapshot"><h3>Market Snapshot</h3><p>{esc(scope)}</p>{rows}<p>Rates: source units/hour · {data.hourly_midpoint.count():,} valid midpoints.<br>Remote signals are title-based estimates.</p></section>', unsafe_allow_html=True)


def note(text, warning=False):
    st.markdown(f'<div class="ts-note{" warning" if warning else ""}">{esc(text)}</div>', unsafe_allow_html=True)


def empty_state(title, detail):
    st.markdown(f'<div class="ts-empty"><h3>{esc(title)}</h3><p>{esc(detail)}</p></div>', unsafe_allow_html=True)


def chart(fig, key, selectable=False, height=320):
    """A shared chart language, with supported point selections for drill-downs."""
    fig.update_layout(template='plotly_white', height=height, font=dict(family='Inter, Segoe UI, sans-serif', size=11, color='#718096'),
        title=dict(font=dict(size=13, color='#334155'), x=0), paper_bgcolor='#FFFFFF', plot_bgcolor='#FFFFFF',
        margin=dict(l=8, r=16, t=34, b=10), colorway=PALETTE, legend_title_text='',
        legend=dict(orientation='h', y=-.15, x=0, font=dict(size=10)), hoverlabel=dict(bgcolor='#14243A', font_color='white'),
        selectionrevision=key)
    fig.update_xaxes(gridcolor='#EEF2F7', zeroline=False, title_font=dict(size=10), tickfont=dict(size=10))
    fig.update_yaxes(gridcolor='#EEF2F7', zeroline=False, title_font=dict(size=10), tickfont=dict(size=10))
    ctx = st.session_state.get('_ctx', {})
    widget_key = f'{key}_{st.session_state.get("_nav_epoch", 0)}_{ctx.get("token", "empty")}'
    return st.plotly_chart(fig, width='stretch', theme=None, key=widget_key,
        on_select='rerun' if selectable else 'ignore', selection_mode='points',
        config={'displaylogo': False, 'modeBarButtonsToRemove': ['lasso2d', 'select2d'], 'scrollZoom': False})


def table(frame, filename, key, height=360):
    shown = frame
    if len(frame) > 100:
        left, right = st.columns([3, 1])
        pages = math.ceil(len(frame) / 100)
        page_key = key + '_table_page'
        if st.session_state.get(page_key, 1) > pages:
            st.session_state[page_key] = 1
        st.session_state.setdefault(page_key, 1)
        page = int(right.number_input('Table page', min_value=1, max_value=pages, value=None, step=1, key=page_key) or 1)
        left.caption(f'{len(frame):,} rows · Page {page} of {pages} · 100 rows per page. Export includes all rows.')
        shown = frame.iloc[(page - 1) * 100:page * 100]
    st.dataframe(shown, hide_index=True, width='stretch', height=min(height, 38 + 35 * len(shown)),
                 column_config={'job_url': st.column_config.LinkColumn('Upwork listing', display_text='Open listing')})
    st.download_button('Download results', csv_bytes(frame), filename, 'text/csv', key=key, icon=':material/download:')


def request_route(slug, filters=None):
    """Callback: apply scope on the next run, before global widgets are created."""
    updates = dict(filters or {})
    if slug == 'explorer':
        updates.update(explorer_query='', explorer_page=1)
    st.session_state['_pending_nav'] = (slug, updates)
    st.session_state['_nav_epoch'] = st.session_state.get('_nav_epoch', 0) + 1


def drill_chart(event, slug, field):
    points = event.get('selection', {}).get('points', []) if event else []
    if points:
        datum = points[0].get('customdata', [])
        if datum:
            request_route(slug, {field: [str(datum[0])]})
            st.rerun()


def reset_scope():
    for key, value in {'g_search': '', 'g_countries': [], 'g_categories': [], 'g_payment': 'Any', 'g_remote': False}.items():
        st.session_state[key] = value
    d = st.session_state.get('_data')
    if d is not None and d.date.notna().any():
        st.session_state['g_dates'] = (d.date.min().date(), d.date.max().date())
    st.session_state['_nav_epoch'] = st.session_state.get('_nav_epoch', 0) + 1


def apply_scope(data, start=None, end=None, search='', countries=None, categories=None, payment='Any', remote=False):
    """Reusable global filtering. Search is a literal title/category substring."""
    d = filter_jobs(data, start, end, categories, countries, payment)
    if search.strip():
        q = search.strip()
        mask = d.title.fillna('').str.contains(q, case=False, regex=False) | d.job_category.astype('string').str.contains(q, case=False, regex=False)
        d = d.loc[mask]
    if remote:
        d = d.loc[d.remote_indicator]
    return d


def ingest_upload(upload, slot='sidebar', force=False):
    """Validate before atomically replacing the active session dataset."""
    if upload is None:
        return False
    content = upload.getvalue()
    digest = hashlib.sha256(content).hexdigest()
    seen = st.session_state.setdefault('_seen_uploads', {})
    if not force and seen.get(slot) == digest:
        return False
    seen[slot] = digest
    try:
        if len(content) > 150 * 1024 * 1024:
            raise ValueError('The upload limit is 150 MB. Please choose a smaller CSV.')
        with branded_loading('Checking fields, dates and compensation…'):
            data, audit = load_csv(content)
    except (ValueError, TypeError, MemoryError) as exc:
        st.session_state['_upload_message'] = ('error', str(exc))
        return False
    changed = digest != st.session_state.get('_digest')
    st.session_state.update({'_data': data, '_audit': audit, '_digest': digest, '_source_bytes': content,
        '_source_name': getattr(upload, 'name', 'Uploaded CSV'), '_upload_message': ('success', f'{len(data):,} unique opportunities are ready to explore.')})
    for key in ['_recommendations', '_salary_cache', '_latest_run']:
        st.session_state.pop(key, None)
    st.session_state['_runs'] = {}
    if changed:
        st.session_state['_saved_jobs'] = {}
        for key in list(st.session_state):
            if key.startswith(('explorer_', 'career_', 'geo_', 'demand_', 'outlook_', 'model_', 'skills_')):
                del st.session_state[key]
        st.session_state['_reset_scope_pending'] = True
    return True


def reprocess_source():
    st.session_state['_reprocess'] = True


def get_context():
    ctx = st.session_state['_ctx']
    if ctx['data'] is None:
        welcome()
        return None
    if ctx['selected'].empty:
        empty_state('No opportunities in this view', 'Try a broader date range, another search term or fewer filters.')
        st.button('Reset all filters', on_click=reset_scope, icon=':material/filter_alt_off:', key='empty_reset')
        return None
    return ctx


def welcome():
    data = st.session_state.get('_data')
    left, right = st.columns([1.35, 1], gap='large')
    with left:
        st.markdown('<section class="ts-landing-hero">' + brand_svg() + '<div class="ts-eyebrow">A global perspective on opportunity</div><h1>Navigate the Global Freelance Economy</h1><p>Explore job opportunities, analyze compensation trends and discover emerging market opportunities through intelligent data analytics.</p><p class="ts-landing-tagline">Discover Opportunities. Understand Markets. Make Smarter Decisions.</p></section>', unsafe_allow_html=True)
        with st.container(key='panel_landing_connect'):
            st.subheader('Your intelligence workspace starts here')
            if data is None:
                st.caption('Connect your Upwork CSV. TalentScope validates its fields and prepares your market view.')
            else:
                st.caption(f'{len(data):,} unique opportunities connected. Enter your workspace or replace the snapshot below.')
            st.button('Enter analytics workspace', on_click=enter_workspace, type='primary', icon=':material/arrow_forward:', disabled=data is None, width='stretch', key='landing_enter', help='Upload a valid CSV to activate your workspace.' if data is None else 'Open Market Overview with your connected dataset.')
            upload = st.file_uploader('Connect a dataset', type=['csv'], max_upload_size=150, key='welcome_upload', help='Use the original Upwork CSV. Your file is processed in server memory.')
            if ingest_upload(upload, 'welcome') or '_upload_message' in st.session_state:
                request_route('welcome')
                st.rerun()
    with right:
        if data is not None:
            valid = data.date.dropna()
            period = f'{valid.min():%d %b %Y} – {valid.max():%d %b %Y}' if len(valid) else 'No valid publication dates'
            counts = data.job_category.value_counts().head(4)
            bars = ''.join(f'<div class="ts-preview-row"><div><span>{esc(name)}</span><span>{count:,}</span></div><div class="ts-preview-track"><i style="width:{count/counts.max()*100:.1f}%"></i></div></div>' for name, count in counts.items())
            preview = f'<div class="ts-preview-kpis"><div><b>{len(data):,}</b><span>Opportunities</span></div><div><b>{data.loc[data.country_clean.ne("Unknown"), "country_clean"].nunique():,}</b><span>Country labels</span></div><div><b>{number(data.hourly_midpoint.mean(),2)}</b><span>Mean hourly rate</span></div></div>{bars}<div class="ts-preview-meta">{esc(period)} · Historical source<br>Inferred categories · Compensation in source units</div>'
            label = 'Connected snapshot'
        else:
            preview = '<div class="ts-preview-kpis"><div><b>—</b><span>Opportunities</span></div><div><b>—</b><span>Country labels</span></div><div><b>—</b><span>Mean hourly rate</span></div></div><div class="ts-preview-empty"><b>Your data. A clearer perspective.</b><p>Posting trends, global reach and compensation comparisons appear here once your snapshot is connected.</p></div><div class="ts-preview-meta">Workspace preview · Awaiting your dataset<br>No example values or invented market activity</div>'
            label = 'Market intelligence preview'
        st.markdown('<section class="ts-preview"><div class="ts-preview-top">' + brand_svg('white', compact=True) + f'<span class="ts-preview-label">{label}</span></div><h2>See the market.<br>Find your opportunity.</h2><p>A connected view of roles, locations and compensation.</p>{preview}</section>', unsafe_allow_html=True)
    st.subheader('A clearer view, from market to opportunity')
    capabilities = [
        ('Explore the market', 'Search real postings and compare global opportunity across roles, countries and time.', '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c-5 5-5 13 0 18M12 3c5 5 5 13 0 18"/>'),
        ('Understand compensation', 'Separate offered hourly rates from project budgets, with transparent samples and coverage.', '<path d="M4 20h16M6 16v-5M12 16V7M18 16V3"/>'),
        ('Discover your next move', 'Explore evaluated demand scenarios and save an explainable, personalized shortlist.', '<rect x="3" y="7" width="18" height="14" rx="2"/><path d="M8 7V3h8v4M3 12h18M11 11h2v3h-2z"/>')]
    for col, (title, detail, paths) in zip(st.columns(3), capabilities):
        with col:
            with st.container(key='panel_welcome_' + title):
                st.markdown(f'<div class="ts-capability-symbol"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg></div>', unsafe_allow_html=True)
                st.subheader(title); st.caption(detail)
    with st.expander('CSV requirements'):
        st.write('Required columns'); st.code(', '.join(REQUIRED))
        st.caption('Historical data is labeled with its actual coverage. No salary values, current market activity or job descriptions are invented.')


def metric_row(items):
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        col.metric(label, value, help=help_text)


def compensation(row):
    if row['salary_type'] == 'Hourly':
        lo, hi = row['hourly_low_valid'], row['hourly_high_valid']
        if pd.notna(lo) and pd.notna(hi):
            return f'{lo:,.0f}–{hi:,.0f} / hour'
        if pd.notna(lo): return f'From {lo:,.0f} / hour'
        if pd.notna(hi): return f'Up to {hi:,.0f} / hour'
        return 'Hourly rate undisclosed'
    if row['salary_type'] == 'Fixed-price' and pd.notna(row['fixed_budget_valid']):
        return f'{row["fixed_budget_valid"]:,.0f} fixed budget'
    return 'Compensation undisclosed'


def job_details(row, key):
    with st.container(key='panel_detail_' + key):
        st.subheader(str(row['title']) if pd.notna(row['title']) else 'Untitled opportunity')
        st.caption(f'{row["country_clean"]} · {row["job_category"]} · {pd.Timestamp(row["date"]):%d %b %Y}' if pd.notna(row['date']) else str(row['country_clean']))
        st.write(compensation(row))
        st.caption('Compensation is in source units. Currency and full job descriptions are not provided in the CSV.')
        st.write(f'Inferred role group: {row["job_subcategory"]}')
        if row['job_url']:
            st.link_button('Open original Upwork listing', row['job_url'], icon=':material/open_in_new:')
        else:
            st.caption('A valid source link is unavailable for this posting.')


def job_table(data, key='explorer'):
    """Bound table payloads and expose posting details on row or title selection."""
    if data.empty:
        empty_state('No matching postings', 'Adjust the search or filters to broaden your view.'); return
    a, b, c = st.columns([2, 1, 1])
    a.caption(f'{len(data):,} matching opportunities · Select a title or row to inspect its posting')
    size = b.selectbox('Rows per page', [25, 50, 100], key=key + '_size')
    pages = max(1, math.ceil(len(data) / size))
    page_key = key + '_page'
    if st.session_state.get(page_key, 1) > pages:
        st.session_state[page_key] = 1
    st.session_state.setdefault(page_key, 1)
    page = int(c.number_input('Page', min_value=1, max_value=pages, value=None, step=1, key=page_key) or 1)
    shown = data.iloc[(page - 1) * size: page * size]
    display = shown[['title', 'country_clean', 'job_category', 'salary_type', 'date', 'job_url']].copy()
    display.insert(4, 'compensation', shown.apply(compensation, axis=1))
    state = st.dataframe(display, hide_index=True, width='stretch', height=min(570, 38 + 35 * len(shown)),
        column_config={'title': st.column_config.TextColumn('Opportunity', width='large'), 'country_clean': 'Country',
        'job_category': 'Inferred category', 'salary_type': 'Payment', 'compensation': 'Offered compensation',
        'date': st.column_config.DateColumn('Posted', format='DD MMM YYYY'),
        'job_url': st.column_config.LinkColumn('Upwork', display_text='Open listing')},
        selection_mode=['single-row', 'single-cell'], on_select='rerun', key=f'{key}_grid_{st.session_state["_ctx"]["token"]}_{page}_{size}')
    selected_row = state.selection.cells[0][0] if state.selection.cells else (state.selection.rows[0] if state.selection.rows else None)
    if selected_row is not None:
        job_details(shown.iloc[selected_row], key)
    st.caption(f'Page {page:,} of {pages:,}. Downloads include every matching row, not only this page.')
    st.download_button('Download filtered opportunities', lambda: csv_bytes(data), 'talentscope_opportunities.csv', 'text/csv', key=key + '_download', icon=':material/download:', on_click='ignore')


@st.cache_data(max_entries=6, ttl=1800, show_spinner=False)
def cached_salary(signature, payment, minimum, _data):
    return salary_keywords(_data, payment, minimum)


@st.cache_data(max_entries=3, ttl=1800, show_spinner=False)
def title_keywords(signature, _data):
    vectorizer = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2, max_features=1500, dtype=np.float32)
    try:
        x = vectorizer.fit_transform(_data.job_title_clean)
    except ValueError:
        return pd.DataFrame(columns=['keyword', 'postings', 'mean_tfidf'])
    presence = x.copy(); presence.data[:] = 1
    return pd.DataFrame({'keyword': vectorizer.get_feature_names_out(), 'postings': np.asarray(presence.sum(axis=0)).ravel().astype(int),
                         'mean_tfidf': np.asarray(x.mean(axis=0)).ravel()}).sort_values('postings', ascending=False)


@st.cache_data(max_entries=8, ttl=1800, show_spinner=False)
def cached_projection(signature, horizon, start, end, group, value, _data):
    return predict_job_demand(_data, horizon, start, end, group, value)


def build_projection(ctx, kind, horizon, dates):
    groups = {'Category demand': 'job_category', 'Detailed roles': 'job_subcategory', 'Total opportunities': None,
              'Remote title signals': 'remote_series', 'Hourly compensation': None}
    group = groups[kind]; value = 'hourly_midpoint' if kind == 'Hourly compensation' else None
    data = ctx['selected']
    if group == 'remote_series':
        data = data.assign(remote_series=np.where(data.remote_indicator, 'Remote title signal', 'No title signal'))
    key = (ctx['signature'], kind, horizon, tuple(map(str, dates)))
    with branded_loading('Evaluating historical patterns and preparing projections…'):
        result = cached_projection(ctx['signature'], horizon, str(dates[0]), str(dates[1]), group, value, data)
    bundle = {'result': result, 'kind': kind, 'horizon': horizon, 'scope': ctx['signature'], 'key': key}
    runs = st.session_state.setdefault('_runs', {})
    runs[str(key)] = bundle
    while len(runs) > 6:
        del runs[next(iter(runs))]
    st.session_state['_latest_run'] = bundle
    return bundle


def projection_controls(ctx, prefix, kinds=None):
    kinds = kinds or ['Category demand', 'Detailed roles', 'Total opportunities']
    lo, hi, threshold = suggest_model_window(ctx['selected'])
    if lo is None:
        return None
    with st.container(key='panel_' + prefix + '_controls'):
        a, b, c = st.columns([1.2, 1.6, 1])
        kind = a.selectbox('Forecast focus', kinds, key=prefix + '_kind')
        dates = b.date_input('Analysis window', (lo.date(), hi.date()), min_value=ctx['selected'].date.min().date(),
                            max_value=ctx['selected'].date.max().date(), key=prefix + '_dates_' + ctx['token'])
        horizon = c.selectbox('Projection horizon', [7, 30], index=1, format_func=lambda x: f'Next {x} days', key=prefix + '_horizon')
        st.caption(f'Suggested coverage: {lo:%d %b %Y} – {hi:%d %b %Y}. Uses consecutive days with at least {threshold:,} postings/day and excludes the final observed day. Adjust after reviewing data coverage.')
        run = st.button('Generate outlook' if prefix == 'outlook' else 'Evaluate forecast', type='primary', key=prefix + '_run', icon=':material/insights:')
    if not dates or len(dates) != 2:
        note('Select both a start and end date.'); return None
    desired = (ctx['signature'], kind, horizon, tuple(map(str, dates)))
    if run:
        bundle = build_projection(ctx, kind, horizon, dates)
    else:
        bundle = st.session_state.get('_runs', {}).get(str(desired))
    return bundle


def forecast_chart(bundle, key, compact=False):
    result = bundle['result']
    if result['status'] != 'ok':
        note(result['reason'], True); return
    is_rate = bundle['kind'] == 'Hourly compensation'
    series = sorted(result['forecast'].series.unique())
    chosen = st.selectbox('View series', series, key=key + '_series') if len(series) > 1 else series[0]
    h = result['history'].loc[result['history'].series.eq(chosen)]
    f = result['forecast'].loc[result['forecast'].series.eq(chosen)]
    fig = go.Figure()
    fig.add_scatter(x=h.date, y=h.actual, name='Observed', mode='lines', line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=f.date, y=f.upper_error_band, line=dict(width=0), showlegend=False, hoverinfo='skip')
    fig.add_scatter(x=f.date, y=f.lower_error_band, fill='tonexty', fillcolor='rgba(37,99,235,.10)', line=dict(width=0), name='Exploratory error band', hoverinfo='skip')
    fig.add_scatter(x=f.date, y=f.projection, name='Projected', mode='lines', line=dict(color=BLUE, dash='dash', width=2.5))
    fig.add_vline(x=result['window_end'].timestamp() * 1000, line_color='#ADB9CA', line_dash='dot')
    fig.update_layout(title=str(chosen), yaxis_title='Source units / hour' if is_rate else 'Opportunities / day', xaxis_title='Date (UTC)')
    chart(fig, key + '_forecast', height=320 if compact else 370)
    st.caption(f'Projection begins {f.date.min():%d %b %Y}, after the selected training window. Bands describe historical forecast errors; they are not calibrated confidence intervals.')
    if not compact:
        summary = result['forecast'].groupby('series', observed=True).agg(
            next_7_days=('projection', lambda s: s.iloc[:7].mean() if is_rate else s.iloc[:7].sum()),
            selected_horizon=('projection', 'mean' if is_rate else 'sum')).reset_index().sort_values('next_7_days', ascending=False)
        summary.columns = ['Series', 'Next 7-day mean rate' if is_rate else 'Next 7-day opportunities',
                           f'{bundle["horizon"]}-day mean rate' if is_rate else f'{bundle["horizon"]}-day opportunities']
        table(summary.round(2), 'talentscope_projection_summary.csv', key + '_summary')
        st.download_button('Download daily projections', csv_bytes(result['forecast']), 'talentscope_daily_projections.csv', 'text/csv', key=key + '_daily', icon=':material/download:')


# Each navigation destination is a distinct callable page in this file.
def page_market_overview():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']; m = analyze_market(d); k = m['kpis']
    cards = [('Total Job Opportunities', number(len(d)), 'explorer', {}, 'Unique observed postings in the current scope.'),
             ('Countries Covered', number(k['countries']), 'global', {}, 'Nonmissing location labels; unresolved locations remain in tables.'),
             ('Hourly Job Share', f'{k["hourly_pct"]:.1f}%', 'salary', {'g_payment': 'Hourly'}, 'Share of selected postings with hourly payment.'),
             ('Average Hourly Compensation', number(k['average_hourly'], 2), 'salary', {'g_payment': 'Hourly'}, f'{d.hourly_midpoint.count():,} valid midpoints; source units per hour.'),
             ('Fixed-Price Opportunities', number(d.salary_type.eq('Fixed-price').sum()), 'explorer', {'g_payment': 'Fixed-price'}, 'Fixed budgets are total project amounts.'),
             ('Remote Opportunity Share · Estimated', f'{d.remote_indicator.mean()*100:.1f}%', 'remote', {}, 'Title keyword proxy; not independently verified.')]
    for col, (title, value, dest, filters, help_text) in zip(st.columns(6), cards):
        with col:
            with st.container(key='kpi_' + dest + title[:5]):
                st.button(f'{value}\n\n{title}', help=help_text, width='stretch', key='kpi_button_' + title,
                          on_click=request_route, args=(dest, filters))
    st.caption('Offered compensation is shown in source units. Currency is not provided. Every card opens its detailed view.')
    with st.expander('Market Snapshot · exact counts and rates'):
        market_snapshot(d)
    a, b = st.columns([1.65, 1])
    with a:
        with st.container(key='panel_market_activity'):
            st.subheader('Market activity')
            grain = st.segmented_control('Time interval', ['Daily', 'Weekly', 'Monthly'], default='Daily', key='overview_grain', label_visibility='collapsed')
            freq = {'Daily': 'D', 'Weekly': 'W-MON', 'Monthly': 'MS'}[grain or 'Daily']
            counts = d.dropna(subset=['date']).set_index('date').resample(freq).size().rename('Opportunities').reset_index()
            # Globally missing dates are absent observations, not measured zero demand.
            counts.loc[counts.Opportunities.eq(0), 'Opportunities'] = np.nan
            fig = px.area(counts, x='date', y='Opportunities', labels={'date': 'Publication date (UTC)'})
            fig.update_traces(line_color=BLUE, fillcolor='rgba(37,99,235,.08)', connectgaps=False, hovertemplate='%{x|%d %b %Y}<br>%{y:,.0f} opportunities<extra></extra>')
            chart(fig, 'overview_activity', height=290)
            st.caption('Collection coverage varies; boundary weeks and months may be incomplete.')
    with b:
        with st.container(key='panel_top_roles'):
            st.subheader('Where opportunity is concentrated')
            cats = m['categories'].head(7).sort_values('postings')
            fig = px.bar(cats, x='postings', y='job_category', orientation='h', custom_data=['job_category'], labels={'postings': 'Opportunities', 'job_category': ''})
            fig.update_traces(marker_color=BLUE, hovertemplate='%{y}<br>%{x:,.0f} opportunities<extra>Explore category</extra>')
            drill_chart(chart(fig, 'overview_categories', True, 320), 'explorer', 'g_categories')
            st.caption('Select a category bar to explore its postings.')
    a, b = st.columns(2)
    with a:
        with st.container(key='panel_overview_pay'):
            st.subheader('Compensation by category')
            pay = d.groupby('job_category', observed=True).hourly_midpoint.agg(['count', 'median']).query('count >= 20').nlargest(7, 'median').reset_index().sort_values('median')
            if len(pay):
                fig = px.bar(pay, x='median', y='job_category', orientation='h', custom_data=['job_category', 'count'], labels={'median': 'Median offered hourly midpoint · source units', 'job_category': ''})
                fig.update_traces(marker_color='#7096E8', hovertemplate='%{y}<br>Median: %{x:.2f} / hour<br>Observed salaries: %{customdata[1]:,}<extra></extra>')
                drill_chart(chart(fig, 'overview_compensation', True, 280), 'salary', 'g_categories')
            else: st.caption('At least 20 observed hourly midpoints per category are required.')
    with b:
        with st.container(key='panel_overview_geo'):
            st.subheader('Geographical reach')
            locations = d.groupby('country_clean', observed=True).size().rename('Opportunities').nlargest(7).reset_index().sort_values('Opportunities')
            fig = px.bar(locations, x='Opportunities', y='country_clean', orientation='h', custom_data=['country_clean'], labels={'country_clean': ''})
            fig.update_traces(marker_color='#4E9EA6', hovertemplate='%{y}<br>%{x:,.0f} opportunities<extra>Explore country</extra>')
            drill_chart(chart(fig, 'overview_locations', True, 280), 'global', 'g_countries')
    st.subheader('Market highlights')
    largest = m['categories'].iloc[0]
    observed = d.hourly_midpoint.count(); hourly = d.salary_type.eq('Hourly').sum()
    insights = [(f'{largest.job_category} leads observed volume', f'{largest.postings:,} postings, or {largest.postings/len(d)*100:.1f}% of this view. Categories are inferred from titles.'),
                (f'{observed/max(hourly,1)*100:.1f}% of hourly postings disclose both bounds', 'The hourly midpoint is calculated only when both rate bounds are valid.'),
                (f'{d.remote_indicator.sum():,} postings contain remote-related title terms', 'This estimate reflects title language, not verified working arrangements.')]
    for col, (title, detail) in zip(st.columns(3), insights):
        col.markdown(f'<div class="ts-highlight"><b>{esc(title)}</b><p>{esc(detail)}</p></div>', unsafe_allow_html=True)
    st.subheader('Continue exploring')
    for col, (label, dest, icon) in zip(st.columns(4), [('Explore Job Market', 'explorer', 'travel_explore'), ('Compare Salaries', 'salary', 'payments'), ('View Demand Forecast', 'demand', 'trending_up'), ('Discover Recommendations', 'career', 'auto_awesome')]):
        col.page_link(st.session_state['_pages'][dest], label=label, icon=f':material/{icon}:', width='stretch')


def page_market_explorer():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']
    with st.container(key='panel_explorer_search'):
        a, b = st.columns([3, 1])
        query = a.text_input('Title keyword', placeholder='Try Python, video editor or virtual assistant', key='explorer_query')
        order = b.selectbox('Sort postings', ['Newest first', 'Oldest first', 'Title A–Z'], key='explorer_sort')
        if query:
            d = d.loc[d.title.fillna('').str.contains(query, case=False, regex=False)]
        a.caption('Country, category, date and payment filters are available in the header. Search matches literal title text.')
    d = d.sort_values('title' if order == 'Title A–Z' else 'date', ascending=order != 'Newest first', na_position='last')
    metric_row([('Matching opportunities', number(len(d)), 'After all active filters.'), ('Hourly postings', number(d.salary_type.eq('Hourly').sum()), ''),
                ('Fixed-price postings', number(d.salary_type.eq('Fixed-price').sum()), '')])
    st.subheader('Opportunity directory')
    job_table(d)


def page_salary_intelligence():
    ctx = get_context()
    if not ctx: return
    a, b, c = st.columns([1.4, 1, 1.2])
    scope_payment = st.session_state.get('g_payment', 'Any')
    cohorts = [scope_payment] if scope_payment in ['Hourly', 'Fixed-price'] else ['Hourly', 'Fixed-price']
    if st.session_state.get('salary_kind') not in cohorts:
        st.session_state['salary_kind'] = cohorts[0]
    kind = a.selectbox('Payment cohort', cohorts, index=None, key='salary_kind') or cohorts[0]
    if len(cohorts) == 1:
        st.caption(f'The header payment filter limits this view to {scope_payment}. Choose Any in Filters to compare both payment cohorts.')
    st.session_state.setdefault('salary_minimum', 20)
    minimum = int(b.number_input('Minimum samples', 5, 1000, value=None, key='salary_minimum') or 20)
    omit = c.checkbox('Exclude IQR outliers', value=False, key='salary_outliers', help='This descriptive filter flags unusual values. It does not establish that they are incorrect.')
    target = 'hourly_midpoint' if kind == 'Hourly' else 'fixed_budget_valid'
    unit = 'Source units / hour' if kind == 'Hourly' else 'Source units / project'
    d = ctx['selected'].loc[ctx['selected'].salary_type.eq(kind)]
    if omit: d = d.loc[~d[target + '_outlier']]
    y = d[target].dropna()
    if y.empty:
        empty_state('No disclosed compensation in this cohort', 'Choose a different payment type or broaden the header filters.'); return
    metric_row([('Mean offered amount', number(y.mean(), 2), unit), ('Median offered amount', number(y.median(), 2), unit),
                ('Observed range', f'{y.min():,.0f} – {y.max():,.0f}', unit), ('Salary observations', number(len(y)), f'{len(d)-len(y):,} postings have no usable value.')])
    note(f'{kind} compensation · {unit.lower()}. {len(d)-len(y):,} postings have no usable value. Fixed budgets represent total project scope; currency is not supplied.')
    summary = d.groupby('job_category', observed=True)[target].agg(['count', 'mean', 'median', 'min', 'max']).reset_index()
    qualified = summary.loc[summary['count'].ge(minimum)].sort_values('median')
    a, b = st.columns([1.1, 1])
    with a:
        with st.container(key='panel_salary_category'):
            st.subheader('Category compensation')
            fig = px.bar(qualified, y='job_category', x='median', orientation='h', custom_data=['job_category', 'count', 'mean'], labels={'job_category': '', 'median': f'Median · {unit.lower()}'})
            fig.update_traces(marker_color=BLUE, hovertemplate='%{y}<br>Median: %{x:.2f}<br>Mean: %{customdata[2]:.2f}<br>Samples: %{customdata[1]:,}<extra></extra>')
            drill_chart(chart(fig, 'salary_categories', True), 'explorer', 'g_categories')
    sample = d.dropna(subset=[target]).sample(min(15000, len(y)), random_state=42)
    with b:
        with st.container(key='panel_salary_distribution'):
            st.subheader('Offered amount distribution')
            chart(px.histogram(sample, x=target, nbins=60, log_y=True, labels={target: unit}, color_discrete_sequence=[BLUE]), 'salary_distribution')
            st.caption('Logarithmic count axis; up to 15,000 reproducibly sampled points. Summary statistics use the full cohort.')
    tabs = st.tabs(['Category detail', 'Country comparison', 'Statistical context'])
    with tabs[0]:
        table(summary.rename(columns={'job_category': 'Category', 'count': 'Observations', 'mean': 'Mean', 'median': 'Median', 'min': 'Minimum', 'max': 'Maximum'}).round(2), 'talentscope_category_compensation.csv', 'salary_category_table')
        with st.expander('Compare category distributions'):
            chart(px.box(sample, x='job_category', y=target, points=False, labels={'job_category': 'Inferred category', target: unit}), 'salary_box', height=350)
    with tabs[1]:
        if kind == 'Hourly':
            countries = analyze_country_salary(d, minimum)
            eligible = countries.loc[countries.eligible_ranking].head(15)
            chart(px.bar(eligible.sort_values('mean_midpoint'), x='mean_midpoint', y='country_clean', orientation='h', hover_data=['salary_n', 'median_midpoint'], labels={'country_clean': '', 'mean_midpoint': 'Mean hourly midpoint · source units'}), 'salary_country', height=370)
            table(countries.round(2), 'talentscope_country_compensation.csv', 'salary_country_table')
        else:
            st.caption('Country hourly comparisons are available in the Hourly cohort. Fixed budgets remain project amounts.')
    with tabs[2]:
        st.caption('Associations describe the observed postings and are not causal estimates. No arbitrary title encoding is used.')
        if st.button('Calculate title–compensation associations', key='salary_statistics'):
            with branded_loading('Calculating associations and group comparisons…'):
                result = cached_salary((ctx['signature'], omit), kind, minimum, d)
            st.session_state['_salary_cache'] = ((ctx['signature'], kind, minimum, omit), result)
        stored = st.session_state.get('_salary_cache')
        if stored and stored[0] == (ctx['signature'], kind, minimum, omit):
            result = stored[1]
            if len(result['keywords']): table(result['keywords'].round(5), 'talentscope_keyword_associations.csv', 'salary_associations')
            st.write('Category comparison (Kruskal–Wallis)', result['test'])
            if not result['correlation'].empty:
                chart(px.imshow(result['correlation'], text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r'), 'salary_correlation', height=290)
            st.caption('Spearman correlations use term presence and salary ranks. Approximate p-values are adjusted with Benjamini–Hochberg. Hourly-bound/midpoint correlations are mechanically related. A displayed p-value of zero reflects numerical underflow.')


def page_skills_roles():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']
    note('Job-title keywords and inferred role signals are extracted from titles. The source contains no full descriptions or verified skill requirements.')
    with branded_loading('Reading job-title patterns…'):
        words = title_keywords(ctx['signature'], d)
    classified = d.job_category.ne('Other / Unclassified').sum()
    metric_row([('Distinct normalized titles', number(d.job_title_clean.nunique()), ''), ('Classified title share', f'{classified/len(d)*100:.1f}%', 'Ordered rules; not verified occupations.'),
                ('Observed keyword features', number(len(words)), 'Unigrams and bigrams retained in the vocabulary.')])
    a, b = st.columns([1.2, 1])
    with a:
        with st.container(key='panel_skills_terms'):
            st.subheader('Most frequent title keywords')
            if len(words):
                chart(px.bar(words.head(15).sort_values('postings'), x='postings', y='keyword', orientation='h', labels={'postings': 'Postings containing term', 'keyword': ''}, hover_data=['mean_tfidf']), 'skills_words', height=385)
            else: st.caption('Too few repeated title terms to build a useful vocabulary.')
    with b:
        with st.container(key='panel_skills_composition'):
            st.subheader('Role composition')
            counts = d.groupby(['job_category', 'job_subcategory'], observed=True).size().reset_index(name='postings')
            chart(px.treemap(counts, path=['job_category', 'job_subcategory'], values='postings', color_discrete_sequence=PALETTE), 'skills_treemap', height=385)
    tabs = st.tabs(['Keyword compensation', 'Related titles', 'Emerging role signals', 'Classification rules'])
    with tabs[0]:
        kind = st.selectbox('Compensation cohort', ['Hourly', 'Fixed-price'], key='skills_kind')
        with branded_loading('Comparing observed keyword compensation…'):
            result = cached_salary(ctx['signature'], kind, 30, d)
        eligible = result['keywords']
        if len(eligible):
            eligible = eligible.loc[eligible.salary_n.ge(30)].sort_values('mean_salary', ascending=False)
            chart(px.bar(eligible.head(15).sort_values('mean_salary'), x='mean_salary', y='keyword', orientation='h', hover_data=['salary_n', 'spearman_rho', 'q_value_bh'], labels={'mean_salary': 'Mean hourly midpoint' if kind == 'Hourly' else 'Mean fixed project budget', 'keyword': ''}), 'skills_pay', height=370)
            table(eligible, 'talentscope_keyword_compensation.csv', 'skills_salary_download')
        else: st.caption('No supported keyword comparisons in this cohort.')
    with tabs[1]:
        query = st.text_input('Find titles related to', 'Python data engineer', key='skills_related_query')
        if st.button('Find related titles', key='skills_related_run'):
            with branded_loading('Matching word and character n-grams…'):
                try:
                    related = recommend_jobs(d, query, top_k=15)
                    st.session_state['_related'] = (ctx['signature'], query, related)
                except ValueError as exc: st.error(str(exc))
        related = st.session_state.get('_related')
        if related and related[:2] == (ctx['signature'], query):
            if len(related[2]): table(related[2][['title', 'job_category', 'match_percentage', 'explanation', 'job_url']], 'talentscope_related_titles.csv', 'skills_related_download')
            else: st.caption('No related titles found. Try a broader role name.')
        st.caption('Related titles use the same word/character TF-IDF similarity as Career Recommendations.')
    with tabs[2]:
        lo, hi, _ = suggest_model_window(d)
        growth, weekly = category_growth(d, lo, hi)
        render_growth(growth, weekly, 'skills_growth')
    with tabs[3]:
        st.write('The first matching rule assigns one category and role group; unmatched titles remain unclassified.')
        table(pd.DataFrame(RULES, columns=['Category', 'Role group', 'Title pattern']), 'talentscope_role_rules.csv', 'skills_rules')


def render_growth(growth, weekly, key):
    if growth.empty:
        note('Two adjacent complete observed weeks with sufficient category samples are required for a growth comparison.'); return
    st.caption('Comparison: ' + str(growth.comparison.iloc[0]) + '. These are source posting trends, not proof of new occupations.')
    valid = growth.dropna(subset=['growth_pct']).sort_values('growth_pct')
    fig = px.bar(valid, x='growth_pct', y='job_category', orientation='h', color='status',
                 color_discrete_map={'Growing': GREEN, 'Declining': '#718096', 'Insufficient / stable': AMBER},
                 hover_data=['previous', 'recent', 'share_change_pp'], labels={'growth_pct': 'Observed weekly change (%)', 'job_category': ''})
    chart(fig, key, height=330)
    with st.expander('Weekly counts and comparison detail'):
        table(growth, 'talentscope_category_growth.csv', key + '_table')
        if len(weekly):
            long = weekly.reset_index().melt(id_vars='week', var_name='Category', value_name='Opportunities')
            chart(px.line(long, x='week', y='Opportunities', color='Category', labels={'week': 'Week starting (UTC)'}), key + '_weekly', height=340)


def page_global_markets():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']
    countries = d.groupby('country_clean', observed=True).agg(postings=('title', 'size'), iso3=('country_iso3', 'first'),
                       mean_hourly=('hourly_midpoint', 'mean'), median_hourly=('hourly_midpoint', 'median'), salary_n=('hourly_midpoint', 'count')).reset_index()
    metric_row([('Locations represented', number(countries.country_clean.ne('Unknown').sum()), ''),
                ('Largest source market', str(countries.nlargest(1, 'postings').country_clean.iloc[0]), 'Posting/client location; not verified worker location.'),
                ('Missing location', f'{d.country_clean.eq("Unknown").mean()*100:.1f}%', 'Missing countries are excluded from the map.')])
    with st.container(key='panel_global_map'):
        a, b = st.columns([2, 1])
        a.subheader('The opportunity map')
        measure = b.selectbox('Map measure', ['Opportunity volume', 'Median hourly rate'], key='geo_measure', label_visibility='collapsed')
        col = 'postings' if measure == 'Opportunity volume' else 'median_hourly'
        mapped = countries.dropna(subset=['iso3', col])
        if col == 'median_hourly': mapped = mapped.loc[mapped.salary_n.ge(20)]
        if len(mapped):
            fig = px.choropleth(mapped, locations='iso3', color=col, hover_name='country_clean', custom_data=['country_clean'],
                hover_data={'iso3': False, 'postings': ':,', 'salary_n': ':,', 'median_hourly': ':.2f'}, color_continuous_scale=['#EAF1FC', '#8CB0EE', '#2563EB', '#14243A'],
                labels={'postings': 'Opportunities', 'median_hourly': 'Median hourly rate', 'salary_n': 'Hourly observations'})
            fig.update_geos(showframe=False, showcoastlines=False, projection_type='natural earth', bgcolor='white', landcolor='#F3F5F9')
            drill_chart(chart(fig, 'geo_map', True, 370), 'global', 'g_countries')
        else: st.caption('No mapped countries meet this measure’s observation threshold.')
        st.caption('ISO country identifiers only. Select a country to scope this workspace. Hourly rates require at least 20 midpoints; all values are in source units.')
    tabs = st.tabs(['Country comparison', 'Country detail'])
    with tabs[0]:
        top = countries.nlargest(15, 'postings').sort_values('postings')
        event = chart(px.bar(top, x='postings', y='country_clean', orientation='h', custom_data=['country_clean'], labels={'postings': 'Opportunities', 'country_clean': ''}), 'geo_volume', True, 380)
        drill_chart(event, 'global', 'g_countries')
        table(countries.sort_values('postings', ascending=False).round(2), 'talentscope_country_markets.csv', 'geo_table')
        with st.expander('Hourly lower, upper and midpoint comparisons'):
            table(analyze_country_salary(d, 20).round(2), 'talentscope_hourly_country_detail.csv', 'geo_bounds')
    with tabs[1]:
        country = st.selectbox('Inspect a location', sorted(d.country_clean.unique()), key='geo_country')
        local = d.loc[d.country_clean.eq(country)]
        metric_row([('Local opportunities', number(len(local)), ''), ('Mean hourly midpoint', number(local.hourly_midpoint.mean(), 2), ''),
                    ('Median hourly midpoint', number(local.hourly_midpoint.median(), 2), '')])
        counts = local.groupby('job_category', observed=True).size().reset_index(name='Opportunities')
        chart(px.bar(counts.sort_values('Opportunities'), x='Opportunities', y='job_category', orientation='h', labels={'job_category': ''}), 'geo_local_categories')
        st.button('Explore postings in this country', on_click=request_route, args=('explorer', {'g_countries': [country]}), icon=':material/arrow_forward:')


def page_remote_insights():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']; signal = d.loc[d.remote_indicator]
    note('Remote classification is inferred from job title keywords and is not independently verified.')
    metric_row([('Estimated remote share', f'{len(signal)/len(d)*100:.1f}%', 'Title signal only.'), ('Title-signal opportunities', number(len(signal)), ''),
                ('Countries with title signals', number(signal.loc[signal.country_clean.ne('Unknown'), 'country_clean'].nunique()), '')])
    labels = d.assign(signal_label=np.where(d.remote_indicator, 'Remote title signal', 'No title signal'))
    a, b = st.columns([1, 1.7])
    with a:
        with st.container(key='panel_remote_mix'):
            st.subheader('Remote opportunity estimate')
            counts = labels.signal_label.value_counts().rename_axis('Signal').reset_index(name='Opportunities')
            fig = px.pie(counts, names='Signal', values='Opportunities', hole=.73, color='Signal', color_discrete_map={'Remote title signal': BLUE, 'No title signal': '#DCE5F2'})
            fig.update_traces(textinfo='percent', hovertemplate='%{label}<br>%{value:,.0f} postings<extra></extra>')
            chart(fig, 'remote_mix', height=290)
    with b:
        with st.container(key='panel_remote_monthly'):
            st.subheader('Monthly title signals')
            monthly = labels.groupby(['month', 'signal_label'], observed=True).size().reset_index(name='Opportunities')
            chart(px.bar(monthly, x='month', y='Opportunities', color='signal_label', color_discrete_map={'Remote title signal': BLUE, 'No title signal': '#DCE5F2'}, labels={'month': 'Month', 'signal_label': ''}), 'remote_monthly', height=290)
            st.caption('Incomplete months are descriptive snapshots. No title signal does not establish on-site work.')
    tabs = st.tabs(['Locations & roles', 'Compensation comparison', 'Signal definition'])
    with tabs[0]:
        if signal.empty:
            empty_state('No remote-related title signals', 'Broaden your filters or inspect another date range.')
        else:
            a, b = st.columns(2)
            for col, field, title in [(a, 'country_clean', 'Locations'), (b, 'job_category', 'Inferred categories')]:
                with col:
                    counts = signal.groupby(field, observed=True).size().nlargest(10).reset_index(name='Opportunities').sort_values('Opportunities')
                    chart(px.bar(counts, x='Opportunities', y=field, orientation='h', labels={field: ''}, title=title), 'remote_' + field)
            st.button('Explore remote-title opportunities', on_click=request_route, args=('explorer', {'g_remote': True}), icon=':material/arrow_forward:')
    with tabs[1]:
        comp = labels.groupby('signal_label', observed=True).agg(postings=('title', 'size'), observations=('hourly_midpoint', 'count'),
                  mean_hourly=('hourly_midpoint', 'mean'), median_hourly=('hourly_midpoint', 'median')).reset_index()
        chart(px.bar(comp, x='signal_label', y=['mean_hourly', 'median_hourly'], barmode='group', hover_data=['observations'], labels={'signal_label': '', 'value': 'Hourly midpoint · source units', 'variable': ''}), 'remote_pay')
        table(comp.round(2), 'talentscope_remote_comparison.csv', 'remote_comparison')
    with tabs[2]:
        st.write('Title terms: remote, work from home, WFH, virtual, online and anywhere.')
        st.caption('Online products and virtual technologies can create false positives. Remote roles without these words create false negatives. These estimates cannot measure the true remote-work share.')
        st.page_link(st.session_state['_pages']['outlook'], label='Explore remote-market scenarios', icon=':material/trending_up:')


def page_demand_forecast():
    ctx = get_context()
    if not ctx: return
    bundle = projection_controls(ctx, 'demand')
    if not bundle:
        with st.container(key='panel_demand_history'):
            st.subheader('Observed demand before projection')
            daily = analyze_market(ctx['selected'])['daily']
            chart(px.line(daily, x='date', y='postings', labels={'date': 'Date (UTC)', 'postings': 'Observed opportunities'}), 'demand_initial')
        note('Evaluate a forecast to see seven- and thirty-day exploratory scenarios. Model selection uses earlier validation data and a separate final seven-day holdout.'); return
    r = bundle['result']
    if r['status'] != 'ok': note(r['reason'], True); return
    selected = r['metrics'].loc[(r['metrics'].model == r['selected_model']) & r['metrics'].stage.str.startswith('Untouched')].iloc[0]
    metric_row([('Selected model', r['selected_model'], 'Chosen by validation MAE.'), ('Observed training window', f'{r["days"]} days', ''),
                ('Holdout MAE', number(selected.MAE, 2), 'Mean absolute error per category/day, not total market error.')])
    with st.container(key='panel_demand_projection'):
        st.subheader('Observed demand & exploratory projection')
        forecast_chart(bundle, 'demand')
    a, b = st.columns([1.6, 1])
    with a:
        with st.container(key='panel_demand_holdout'):
            st.subheader('How the forecast performed')
            h = r['holdout'].groupby('date', observed=True)[['actual', 'predicted']].sum().reset_index()
            chart(px.line(h, x='date', y=['actual', 'predicted'], labels={'date': 'Holdout date', 'value': 'Total daily opportunities', 'variable': ''}), 'demand_holdout', height=270)
    with b:
        st.subheader('Read the forecast carefully')
        st.write(f'This run has {len(r["validation_origins"])} validation origin(s). The final seven days were reserved for evaluation.')
        st.caption('The available collection history is short and uneven. Projections describe this source stream; thirty-day reliability is not established by a seven-day holdout.')
        st.page_link(st.session_state['_pages']['models'], label='Inspect model performance', icon=':material/arrow_forward:')
    note(r['limitations'], True)


def page_market_outlook():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']; lo, hi, _ = suggest_model_window(d)
    growth, weekly = category_growth(d, lo, hi)
    st.subheader('What the observed market is signaling')
    if len(growth):
        growing = growth.loc[growth.status.eq('Growing')]; declining = growth.loc[growth.status.eq('Declining')]
        metric_row([('Growing categories', str(len(growing)), 'Supported positive change between adjacent complete weeks.'),
                    ('Declining categories', str(len(declining)), 'Supported negative change in observed counts.'),
                    ('New in comparison', str(growth.status.eq('Newly observed in comparison').sum()), 'Absent in the prior comparison week, not a newly created occupation.')])
        if len(growing):
            leader = growing.iloc[0]
            note(f'{leader.job_category} has the strongest supported positive change in this comparison: {leader.growth_pct:.1f}% ({leader.previous:,} to {leader.recent:,} postings). Source collection changes may contribute.')
        else: note('No category meets the current sample threshold for positive growth in the latest complete-week comparison.')
    tabs = st.tabs(['Emerging & declining', 'Monthly dynamics', 'Exploratory scenarios'])
    with tabs[0]: render_growth(growth, weekly, 'outlook_growth')
    with tabs[1]:
        monthly = analyze_market(d)['monthly']
        chart(px.bar(monthly, x='month', y='postings', labels={'month': 'Month', 'postings': 'Observed opportunities'}), 'outlook_monthly')
        a, b = st.columns(2)
        with a:
            composition = d.groupby(['month', 'salary_type'], observed=True).size().reset_index(name='Opportunities')
            chart(px.bar(composition, x='month', y='Opportunities', color='salary_type', labels={'month': 'Month', 'salary_type': ''}, title='Payment mix'), 'outlook_mix')
        with b:
            chart(px.line(monthly, x='month', y='hourly_rate', markers=True, hover_data=['salary_n'], labels={'month': 'Month', 'hourly_rate': 'Mean hourly midpoint · source units'}, title='Offered compensation'), 'outlook_rate_history')
        counts = d.groupby(['month', 'job_category'], observed=True).size().unstack(fill_value=0)
        chart(px.imshow(counts.T, aspect='auto', color_continuous_scale='Blues', labels={'x': 'Month', 'y': 'Inferred category', 'color': 'Postings'}), 'outlook_heatmap')
        locations = d.loc[d.country_clean.isin(d.country_clean.value_counts().head(8).index)].groupby(['month', 'country_clean'], observed=True).size().reset_index(name='Opportunities')
        with st.expander('Country changes and monthly detail'):
            chart(px.line(locations, x='month', y='Opportunities', color='country_clean', labels={'month': 'Month', 'country_clean': ''}), 'outlook_countries')
            table(monthly, 'talentscope_monthly_dynamics.csv', 'outlook_monthly_download')
        note('These are incomplete historical monthly snapshots. Month-to-month totals should not be treated as comparable full-month growth rates.', True)
    with tabs[2]:
        st.caption('Explore category demand, remote-title frequency or offered hourly-rate composition. These projections are scenarios, not observed future outcomes.')
        bundle = projection_controls(ctx, 'outlook', ['Category demand', 'Remote title signals', 'Hourly compensation'])
        if bundle:
            forecast_chart(bundle, 'outlook')
            r = bundle['result']
            if r['status'] == 'ok':
                f = r['forecast']; is_rate = bundle['kind'] == 'Hourly compensation'
                if not is_rate:
                    top = f.groupby('series', observed=True).projection.sum().idxmax()
                    note(f'{top} has the largest projected volume within this scenario. This ranking reflects the selected model, filters and history; it is not a guarantee of future demand.')
                else: note(f'The projected daily hourly midpoint averages {f.projection.mean():.2f} source units over the selected horizon. Changes may reflect posting composition rather than changes in pay for a fixed role.')
                note(r['limitations'], True)


def clear_career_filters():
    for key, value in {'career_title': '', 'career_category': 'Any', 'career_country': 'Any', 'career_payment': 'Any',
                       'career_bounds': False, 'career_min': 0., 'career_max': 100., 'career_count': 10}.items():
        st.session_state[key] = value
    st.session_state.pop('_recommendations', None)


def toggle_saved(row):
    saved = st.session_state.setdefault('_saved_jobs', {})
    url = row['job_url']
    if url in saved: del saved[url]
    else: saved[url] = row


def render_job_cards(frame, prefix):
    for _, row in frame.iterrows():
        item = row.to_dict(); identifier = hashlib.sha256(str(row.job_url).encode()).hexdigest()[:12]
        with st.container(key='panel_card_' + prefix + identifier):
            st.markdown(f'<div class="ts-score">{row.match_percentage:.1f}% match</div><div class="ts-job-title">{esc(row.title)}</div>'
                        f'<div class="ts-job-meta">{esc(row.country_clean)} &nbsp;·&nbsp; {esc(row.salary_type)} &nbsp;·&nbsp; {row.date:%d %b %Y}</div>'
                        f'<div class="ts-comp">{esc(compensation(item))}</div>', unsafe_allow_html=True)
            st.caption(row.explanation)
            a, b = st.columns([1, 1])
            is_saved = row.job_url in st.session_state.get('_saved_jobs', {})
            a.button('Remove saved' if is_saved else 'Save opportunity', key=prefix + '_save_' + identifier,
                on_click=toggle_saved, args=(item,), icon=':material/bookmark_remove:' if is_saved else ':material/bookmark_add:', width='stretch')
            b.link_button('Open Upwork listing', row.job_url, icon=':material/open_in_new:', width='stretch')
            with st.expander('Why this opportunity matches'):
                st.write(row.explanation)
                st.caption(f'Inferred category: {row.job_category} · Role group: {row.job_subcategory}. Text similarity: {row.text_similarity*100:.1f}%.')
                st.caption('Final score = 90% title similarity + 10% category alignment. Title similarity combines 65% word and 35% character n-gram cosine similarity. The score is not a hiring probability.')


def page_career_recommendations():
    ctx = get_context()
    if not ctx: return
    d = ctx['selected']
    with st.container(key='panel_career_preferences'):
        st.subheader('Build your opportunity shortlist')
        st.caption('Tell us what you are looking for. Your matches use the opportunities available in the current workspace filters.')
        st.session_state.setdefault('career_title', 'Data Scientist Python')
        title = st.text_input('Desired job title', placeholder='e.g. Python developer or brand designer', key='career_title')
        a, b, c = st.columns(3)
        category = a.selectbox('Job category', ['Any'] + sorted(d.job_category.unique()), key='career_category')
        country = b.selectbox('Preferred country', ['Any'] + sorted(d.country_clean.unique()), key='career_country')
        pay = c.selectbox('Payment preference', ['Any', 'Hourly', 'Fixed-price'], key='career_payment')
        a, b, c, e = st.columns([1.3, 1, 1, 1])
        bounds = a.checkbox('Filter compensation', key='career_bounds', help='Choose a payment type. Unknown compensation is excluded when bounds are active.')
        unit = 'hourly midpoint' if pay != 'Fixed-price' else 'fixed budget'
        st.session_state.setdefault('career_min', 0.)
        st.session_state.setdefault('career_max', 100.)
        low = b.number_input('Minimum ' + unit, min_value=0., value=None, key='career_min', disabled=not bounds)
        high = c.number_input('Maximum ' + unit, min_value=0., value=None, key='career_max', disabled=not bounds)
        st.session_state.setdefault('career_count', 10)
        count = e.selectbox('Results', [5, 10, 20, 30], index=None, key='career_count') or 10
        a, b = st.columns([1, 3])
        run = a.button('Find my matches', type='primary', icon=':material/search:', width='stretch', key='career_run')
        b.button('Clear preferences', on_click=clear_career_filters, key='career_clear', icon=':material/restart_alt:')
    if run:
        if bounds and (low is None or high is None):
            st.error('Enter both compensation limits, or turn off Filter compensation.')
        elif bounds and pay == 'Any':
            st.error('Choose Hourly or Fixed-price before applying compensation limits.')
        else:
            kwargs = (dict(min_hourly=low, max_hourly=high) if pay == 'Hourly' else dict(min_budget=low, max_budget=high)) if bounds else {}
            try:
                with branded_loading('Finding title matches across your market…'):
                    result = recommend_jobs(d, title, None if category == 'Any' else category, None if country == 'Any' else country, pay, top_k=count, **kwargs)
                st.session_state['_recommendations'] = (ctx['signature'], result, (title, category, country, pay, bounds, low, high, count))
            except ValueError as exc: st.error(str(exc))
    saved = st.session_state.get('_saved_jobs', {})
    tabs = st.tabs(['Your matches', f'Saved opportunities · {len(saved)}'])
    with tabs[0]:
        stored = st.session_state.get('_recommendations')
        if stored and stored[0] == ctx['signature']:
            result = stored[1]
            current = (title, category, country, pay, bounds, low, high, count)
            if current != stored[2]: note('Preferences have changed. Select Find my matches to update the shortlist below.')
            if result.empty: empty_state('No relevant opportunities found', 'Broaden your role title, remove a preference or adjust compensation bounds.')
            else:
                st.caption(f'{len(result)} matches for “{stored[2][0]}” · Ranked by title similarity and category alignment')
                st.download_button('Download matches', csv_bytes(result), 'talentscope_recommendations.csv', 'text/csv', key='career_download', icon=':material/download:')
                render_job_cards(result, 'matches')
        else:
            empty_state('Your shortlist is ready to be discovered', 'Set your preferences and find matching opportunities. Saved jobs remain available during this session.')
    with tabs[1]:
        if saved:
            saved_frame = pd.DataFrame(saved.values())
            st.caption('Saved in this session only. Replacing the dataset clears this shortlist.')
            st.download_button('Download saved opportunities', csv_bytes(saved_frame), 'talentscope_saved_opportunities.csv', 'text/csv', key='career_saved_download', icon=':material/download:')
            render_job_cards(saved_frame, 'saved')
        else: empty_state('Keep the opportunities that interest you', 'Select Save opportunity on any recommendation to build your session shortlist.')
    st.caption('Historical postings may have expired. Compensation is shown in source units; currency is unavailable. Matching uses titles, not job descriptions.')


def page_data_quality():
    ctx = st.session_state['_ctx']
    if ctx['data'] is None:
        welcome(); return
    d, audit = ctx['data'], ctx['audit']
    note('This page audits the entire uploaded source, independent of workspace filters.')
    metric_row([('Uploaded records', number(audit['input_rows']), ''), ('Unique opportunities', number(audit['output_rows']), ''),
                ('Duplicates removed', number(audit['exact_duplicates'] + audit['duplicate_job_urls']), 'Exact rows plus canonical Upwork job identifiers.'),
                ('Invalid dates', number(audit['invalid_dates']), 'Invalid dates are retained in the audit and excluded from dated analysis.')])
    tabs = st.tabs(['Coverage & completeness', 'Validation detail', 'Replace dataset'])
    with tabs[0]:
        a, b = st.columns([1.4, 1])
        with a:
            with st.container(key='panel_quality_coverage'):
                st.subheader('Publication coverage')
                coverage = analyze_market(d)['daily']
                if len(coverage): chart(px.bar(coverage, x='date', y='postings', labels={'date': 'Date (UTC)', 'postings': 'Observed postings'}), 'quality_coverage')
                st.caption('Sparse dates and partial boundary days can distort observed growth. A day with no collected records is not proof of zero demand.')
        with b:
            with st.container(key='panel_quality_missing'):
                st.subheader('Missing source values')
                missing = pd.DataFrame({'Field': audit['missing_original'].keys(), 'Missing': audit['missing_original'].values()})
                chart(px.bar(missing.sort_values('Missing'), x='Missing', y='Field', orientation='h'), 'quality_missing')
                st.caption('Hourly fields are expected to be empty for fixed-price jobs, and vice versa.')
        metric_row([('Known country labels', number(d.loc[d.country_clean.ne('Unknown'), 'country_clean'].nunique()), ''),
                    ('Valid hourly midpoints', number(audit['hourly_midpoints_available']), ''),
                    ('Single-bound hourly records', number(audit['hourly_one_bound_only']), 'No midpoint is invented for these records.')])
        mix = d.salary_type.value_counts().rename_axis('Payment type').reset_index(name='Postings')
        st.dataframe(mix, hide_index=True, width='stretch')
    with tabs[1]:
        checks = pd.DataFrame({'Check': ['Invalid payment type', 'Reversed hourly bounds', 'Payment-field conflicts', 'Missing titles', 'Unusable links'] + [f'Invalid / nonpositive {k}' for k in audit['invalid_numeric_or_nonpositive']],
            'Flagged records': [audit['invalid_payment_type'], audit['reversed_hourly_bounds'], audit['payment_conflicts'], audit['missing_titles'], audit['unusable_urls']] + list(audit['invalid_numeric_or_nonpositive'].values())})
        table(checks, 'talentscope_validation_checks.csv', 'quality_checks')
        st.subheader('Field types & missingness')
        types = pd.DataFrame({'Field': audit['input_dtypes'].keys(), 'Input type': audit['input_dtypes'].values()})
        types['Missing values'] = types.Field.map(audit['missing_original'])
        st.dataframe(types, hide_index=True, width='stretch')
        with st.expander('Country normalization and outlier policy'):
            st.write('Country labels are normalized against the country registry. Unmapped labels remain in tables and are excluded from maps.')
            st.write('Unmapped locations:', ', '.join(audit['unmapped_countries']) or 'None')
            st.dataframe(pd.DataFrame(audit['country_labels_normalized'].items(), columns=['Source label', 'Normalized label']), hide_index=True)
            st.write('IQR flags (retained by default)', audit['outliers_iqr'])
        st.download_button('Download prepared source', lambda: csv_bytes(d), 'talentscope_prepared_source.csv', 'text/csv', key='quality_source_download', icon=':material/download:', on_click='ignore')
    with tabs[2]:
        st.subheader('Connect a new market snapshot')
        st.write('A replacement is validated before it becomes active. An invalid upload leaves the current dataset available.')
        upload = st.file_uploader('Replace the current CSV', type=['csv'], max_upload_size=150, key='quality_upload')
        if ingest_upload(upload, 'quality') or '_upload_message' in st.session_state:
            st.rerun()
        st.caption('Replacing the dataset resets filters, forecasts, recommendation results and saved opportunities. No source files are written to the project directory.')
        st.button('Reprocess current source', on_click=reprocess_source, icon=':material/refresh:', key='quality_refresh')


def page_model_performance():
    ctx = get_context()
    if not ctx: return
    runs = [v for v in st.session_state.get('_runs', {}).values() if v['scope'] == ctx['signature'] and v['result']['status'] == 'ok']
    if not runs:
        empty_state('Evaluate a forecast to inspect its evidence', 'Compare genuine validation and holdout results from the current dataset and filters.')
        bundle = projection_controls(ctx, 'model', ['Category demand', 'Total opportunities', 'Remote title signals', 'Hourly compensation'])
        if not bundle or bundle['result']['status'] != 'ok':
            if bundle: note(bundle['result']['reason'], True)
            return
    else:
        choice = st.selectbox('Evaluation run', range(len(runs)), index=len(runs)-1,
            format_func=lambda i: f'{runs[i]["kind"]} · {runs[i]["horizon"]}-day scenario · {runs[i]["result"]["window_start"]:%d %b}–{runs[i]["result"]["window_end"]:%d %b %Y}', key='model_run_choice')
        bundle = runs[choice]
    r = bundle['result']; metrics = r['metrics']
    hold = metrics.loc[metrics.stage.str.startswith('Untouched')]
    selected = hold.loc[hold.model.eq(r['selected_model'])].iloc[0]
    base = hold.loc[hold.model.eq('Seasonal naive (7d)')].iloc[0]
    metric_row([('Selected model', r['selected_model'], 'Chosen using validation MAE, never the holdout.'), ('Holdout MAE', number(selected.MAE, 2), ''),
                ('Holdout RMSE', number(selected.RMSE, 2), ''), ('Holdout R²', number(selected.R2, 4), 'Pooled across series. Category size differences can inflate this number.')])
    change = (base.MAE-selected.MAE)/base.MAE*100 if base.MAE > 0 else np.nan
    note(f'The selected model has {number(abs(change),1)}% {"lower" if change >= 0 else "higher"} holdout MAE than the seasonal baseline in this split. {len(r["validation_origins"])} validation origin(s) are available. This is not evidence of reliable long-horizon performance.')
    a, b = st.columns([1.25, 1])
    with a:
        with st.container(key='panel_model_comparison'):
            st.subheader('Model comparison')
            metric = st.segmented_control('Error metric', ['MAE', 'RMSE'], default='MAE', key='model_metric', label_visibility='collapsed') or 'MAE'
            chart(px.bar(metrics, x='model', y=metric, color='stage', barmode='group', labels={'model': '', 'stage': ''}), 'model_comparison', height=340)
    with b:
        with st.container(key='panel_model_actual'):
            st.subheader('Holdout: actual vs predicted')
            series = st.selectbox('Evaluation series', sorted(r['holdout'].series.unique()), key='model_series')
            h = r['holdout'].loc[r['holdout'].series.eq(series)]
            chart(px.line(h, x='date', y=['actual', 'predicted'], markers=True, labels={'date': 'Holdout date', 'value': 'Daily observed / predicted value', 'variable': ''}), 'model_actual', height=300)
    tabs = st.tabs(['Evaluation results', 'Training & test periods', 'Methodology'])
    with tabs[0]:
        table(metrics.round(5), 'talentscope_model_evaluation.csv', 'model_metrics')
        if r['importance'] is not None:
            with st.expander('Model feature importance'):
                chart(px.bar(r['importance'].head(15).sort_values('importance'), x='importance', y='feature', orientation='h'), 'model_importance')
                st.caption('Tree impurity importance is descriptive and can favor correlated or high-variance features.')
        if r['diagnostic_30d'] is not None:
            st.caption('30-day diagnostic overlaps model-selection history; it is not an independent test.')
            st.dataframe(r['diagnostic_30d'], hide_index=True)
    with tabs[1]:
        rows = []
        for i, origin in enumerate(r['validation_origins']):
            date = pd.Timestamp(origin)
            rows.append({'Split': f'Validation {i+1}', 'Train start': r['window_start'], 'Train end': date-pd.Timedelta(days=1), 'Test start': date, 'Test end': date+pd.Timedelta(days=6)})
        rows.append({'Split': 'Untouched holdout', 'Train start': r['window_start'], 'Train end': r['window_end']-pd.Timedelta(days=7), 'Test start': r['window_end']-pd.Timedelta(days=6), 'Test end': r['window_end']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption('Each forecast is recursive: actual values within its future horizon are not supplied to the model. Future scenarios refit on all observations in the selected window.')
    with tabs[2]:
        st.write('Expanding chronological validation compares a seven-day seasonal baseline with Ridge, Random Forest, Extra Trees and HistGradientBoosting. Validation MAE selects the model; the last seven dates are held out.')
        st.write('Features use past posting counts, trailing means, previous-week change, weekday patterns and series identity. Fixed configurations avoid extensive tuning on short history. A minimum of 35 consecutive observed days is required.')
        st.caption('For category demand, MAE and RMSE are pooled category/day errors, not total-market errors. R² is not an accuracy percentage. No recommendation quality score is invented without relevance labels.')
    note(r['limitations'], True)


PAGE_SPECS = [
    ('Overview', 'overview', 'Market Overview', 'space_dashboard', page_market_overview, 'Freelance Market Overview', 'Understand hiring activity, compensation patterns and emerging opportunities across the freelance economy.'),
    ('Market Intelligence', 'explorer', 'Market Explorer', 'travel_explore', page_market_explorer, 'Explore the freelance market', 'Find the roles, locations and opportunities that matter to you.'),
    ('Market Intelligence', 'salary', 'Salary Intelligence', 'payments', page_salary_intelligence, 'Compensation, with context', 'Compare offered rates and project budgets with clear sample sizes.'),
    ('Market Intelligence', 'skills', 'Skills & Role Insights', 'hub', page_skills_roles, 'Understand the language of demand', 'Explore job-title keywords, related roles and emerging signals.'),
    ('Market Intelligence', 'global', 'Global Market Insights', 'public', page_global_markets, 'A global view of opportunity', 'Understand where postings originate and how offered compensation varies.'),
    ('Market Intelligence', 'remote', 'Remote Work Insights', 'laptop_chromebook', page_remote_insights, 'Read the remote-work signals', 'Track remote-related title language across locations, roles and time.'),
    ('Predictive Analytics', 'demand', 'Demand Forecast', 'trending_up', page_demand_forecast, 'Make sense of future demand', 'Explore historical patterns and carefully evaluated demand projections.'),
    ('Predictive Analytics', 'outlook', 'Future Market Outlook', 'insights', page_market_outlook, 'Turn market signals into perspective', 'Separate observed shifts from exploratory future scenarios.'),
    ('Career Tools', 'career', 'Career Recommendations', 'work_outline', page_career_recommendations, 'Find your next opportunity', 'A personalized shortlist, grounded in the job titles and preferences you choose.'),
    ('Data & Model Insights', 'quality', 'Data Quality', 'fact_check', page_data_quality, 'Know the data behind your decisions', 'Inspect coverage, missing values and validation before interpreting the market.'),
    ('Data & Model Insights', 'models', 'Model Performance', 'monitoring', page_model_performance, 'See the evidence behind the forecast', 'Compare models, inspect holdouts and understand the limits of each prediction.'),
]


def main():
    st.set_page_config(page_title='TalentScope | Freelance Market Intelligence', page_icon=brand_svg('badge', compact=True), layout='wide', initial_sidebar_state='expanded')
    st.markdown(PRODUCT_CSS, unsafe_allow_html=True)
    st.markdown('''<style>
    .ts-snapshot{background:#F7F9FC;border:1px solid #DFE6EF;border-radius:10px;padding:12px;margin:12px 0;color:#172B46;}
    .ts-snapshot h3{font-size:15px!important;margin:0 0 6px!important;padding:0!important;}
    .ts-snapshot p{font-size:11px!important;line-height:1.5;margin:5px 0;color:#53647A;}
    .ts-snapshot-row{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid #E6EBF2;align-items:baseline;}
    .ts-snapshot-row span{font-size:12px;min-width:0;overflow-wrap:anywhere;}
    .ts-snapshot-row strong{font-size:14px;font-variant-numeric:tabular-nums;text-align:right;flex-shrink:0;}
    [data-testid="stMetricValue"]{overflow:visible;white-space:normal;overflow-wrap:anywhere;}
    [data-testid="stMetricLabel"]{height:auto;white-space:normal;}
    </style>''', unsafe_allow_html=True)
    st.markdown('<div class="ts-header-logo"><div class="ts-header-full">' + brand_svg(descriptor=False) + '</div><div class="ts-header-compact">' + brand_svg(compact=True) + '</div></div>', unsafe_allow_html=True)
    # Entrypoint-owned controls persist across callable pages. Keep selected
    # page-specific input preferences when their page is temporarily unmounted.
    keep = ['career_title', 'career_category', 'career_country', 'career_payment', 'career_bounds', 'career_min', 'career_max', 'career_count',
            'explorer_query', 'explorer_sort', 'explorer_size', 'skills_related_query', 'geo_country', 'salary_kind', 'salary_minimum', 'salary_outliers',
            'g_search', 'g_dates', 'g_countries', 'g_categories', 'g_payment', 'g_remote']
    for key in keep:
        if key in st.session_state: st.session_state[key] = st.session_state[key]
    landing_page = st.Page(welcome, title='Product home', icon=':material/home:', url_path='welcome')
    groups, pages = {'Product': [landing_page]}, {'welcome': landing_page}
    for group, slug, title, icon, func, _, _ in PAGE_SPECS:
        page = st.Page(func, title=title, icon=f':material/{icon}:', url_path=slug, default=slug == 'overview')
        groups.setdefault(group, []).append(page); pages[slug] = page
    st.session_state['_pages'] = pages
    navigation = st.navigation(groups, position='hidden')
    pending = st.session_state.pop('_pending_nav', None)
    if pending:
        for key, value in pending[1].items(): st.session_state[key] = value
        st.switch_page(pages[pending[0]])
    spec = next((s for s in PAGE_SPECS if s[2] == navigation.title), ('Product', 'welcome', 'Product home'))
    active_path = '' if spec[1] == 'overview' else spec[1]
    st.markdown(f'<style>[data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{active_path}"] {{background:#EDF3FF;color:#2563EB;font-weight:600;}} [data-testid="stSidebar"] a[data-testid="stPageLink-NavLink"][href="{active_path}"] p {{color:#2563EB;}}</style>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="ts-brand">' + brand_svg(descriptor=False) + '<span class="ts-brand-descriptor">FREELANCE MARKET INTELLIGENCE</span></div><div class="ts-workspace">WORKSPACE<b>Freelance markets</b></div>', unsafe_allow_html=True)
        st.page_link(landing_page, label='Product home', icon=':material/home:', width='stretch')
        snapshot_slot = st.empty()
        for group, destinations in groups.items():
            if group == 'Product':
                continue
            with st.expander(group, expanded=True):
                for page in destinations:
                    st.page_link(page, label=page.title, icon=page.icon, width='stretch')
        st.divider()
        with st.expander('Data source', expanded=st.session_state.get('_data') is None):
            upload = st.file_uploader('Upload Upwork CSV', type=['csv'], max_upload_size=150, key='source_upload', label_visibility='collapsed')
            ingest_upload(upload)
        if st.session_state.get('_data') is not None:
            data = st.session_state['_data']
            valid = data.date.dropna()
            period = f'{valid.min():%d %b %Y} – {valid.max():%d %b %Y}' if len(valid) else 'No valid publication dates'
            st.markdown(f'<div class="ts-source"><span class="ts-status"><i></i>Dataset connected</span><b>{len(data):,} unique opportunities</b>{esc(period)}<br>Historical CSV snapshot</div>', unsafe_allow_html=True)
            st.page_link(pages['quality'], label='Manage dataset', icon=':material/database:', width='stretch')
    if st.session_state.pop('_reprocess', False) and '_source_bytes' in st.session_state:
        load_csv.clear(); cached_projection.clear(); cached_salary.clear(); title_keywords.clear()
        ingest_upload(io.BytesIO(st.session_state['_source_bytes']), 'refresh', force=True)
    if st.session_state.pop('_reset_scope_pending', False):
        reset_scope()
    data = st.session_state.get('_data'); audit = st.session_state.get('_audit', {})
    if spec[1] == 'welcome':
        if data is not None:
            with snapshot_slot.container(): market_snapshot(data, 'Entire connected source')
        message = st.session_state.pop('_upload_message', None)
        if message:
            (st.success if message[0] == 'success' else st.error)(message[1])
        welcome()
        brand_footer()
        return
    if data is None:
        st.title(spec[2])
        st.info('Connect a dataset to open this analysis. Uploads and saved preferences are held in this browser session; a browser refresh or server restart may require uploading again.')
        upload = st.file_uploader('Connect a dataset', type=['csv'], max_upload_size=150, key='route_upload')
        if ingest_upload(upload, 'route'):
            st.rerun()
        message = st.session_state.pop('_upload_message', None)
        if message: (st.success if message[0] == 'success' else st.error)(message[1])
        st.page_link(landing_page, label='Back to Product home', icon=':material/home:')
        brand_footer()
        return
    a, b = st.columns([5, 1.5])
    a.markdown(f'<div class="ts-breadcrumb">Workspace &nbsp;/&nbsp; {esc(spec[0])} &nbsp;/&nbsp; <span>{esc(spec[2])}</span></div>', unsafe_allow_html=True)
    b.markdown('<span class="ts-status"><i></i>Historical dataset</span>' if data is not None else '<span class="ts-status ts-muted-status">No dataset connected</span>', unsafe_allow_html=True)
    st.title(spec[5]); st.markdown(f'<div class="ts-subtitle">{esc(spec[6])}</div>', unsafe_allow_html=True)
    selected = data
    signature = (st.session_state.get('_digest', ''),)
    if data is not None and data.date.notna().any():
        with st.container(key='global_filters'):
            a, b, c, e, f = st.columns([2.3, 1.7, .95, .65, .5])
        search = a.text_input('Global search', placeholder='Search job titles or categories', key='g_search', label_visibility='collapsed', help='Literal text search across titles and inferred categories. Applies across analytical pages.')
        if 'g_dates' not in st.session_state:
            st.session_state['g_dates'] = (data.date.min().date(), data.date.max().date())
        dates = b.date_input('Date range', min_value=data.date.min().date(), max_value=data.date.max().date(), key='g_dates', label_visibility='collapsed')
        with c.popover('Filters', icon=':material/tune:', width='stretch'):
            st.multiselect('Countries', sorted(data.country_clean.unique()), key='g_countries')
            st.multiselect('Categories', sorted(data.job_category.unique()), key='g_categories')
            st.selectbox('Payment type', ['Any', 'Hourly', 'Fixed-price', 'Unknown'], key='g_payment')
            st.checkbox('Remote title signals only', key='g_remote')
        e.button('Reset', on_click=reset_scope, key='global_reset', width='stretch')
        f.button('', icon=':material/refresh:', help='Reprocess the current source and clear analytical results', on_click=reprocess_source, key='global_refresh', width='stretch')
        categories, countries, payment, remote = (st.session_state.get('g_categories', []), st.session_state.get('g_countries', []), st.session_state.get('g_payment', 'Any'), st.session_state.get('g_remote', False))
        if dates and len(dates) == 2:
            selected = apply_scope(data, dates[0], dates[1], search, countries, categories, payment, remote)
        else:
            selected = data.iloc[0:0]
            st.info('Select both a start and end date to calculate this view.')
        signature = (st.session_state['_digest'], tuple(map(str, dates or ())), search, tuple(countries), tuple(categories), payment, remote)
        scope_labels = countries + categories + ([payment] if payment != 'Any' else []) + (['Remote title signals'] if remote else [])
        caption = f'{len(selected):,} opportunities in view · Source coverage {data.date.min():%d %b %Y} – {data.date.max():%d %b %Y} UTC'
        st.caption(caption + (' · ' + ', '.join(scope_labels) if scope_labels else ''))
    elif data is not None:
        selected = data.iloc[0:0]
        st.warning('No valid publication dates are available. Open Data Quality to inspect the source or upload a CSV with valid dates.')
    token = hashlib.sha256(repr(signature).encode()).hexdigest()[:12]
    st.session_state['_ctx'] = {'data': data, 'audit': audit, 'selected': selected, 'signature': signature, 'token': token}
    with snapshot_slot.container(): market_snapshot(selected)
    message = st.session_state.pop('_upload_message', None)
    if message:
        (st.success if message[0] == 'success' else st.error)(message[1])
    try:
        navigation.run()
    except Exception as exc:
        logging.exception('TalentScope view failed: %s', spec[1])
        st.error(f'{spec[2]} could not be prepared ({type(exc).__name__}): {exc}. Reset filters or reprocess the source. Your dataset remains available in Data Quality; the server log contains the traceback.')
    brand_footer()


if __name__ == '__main__':
    main()

