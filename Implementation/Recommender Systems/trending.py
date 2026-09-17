"""
app.py  —  ShopMind · Community-Based Product Recommender
Run with:  streamlit run app.py
"""

import streamlit as st
import pickle, os, re, urllib.parse, time, random
import pandas as pd
import numpy as np
from collections import defaultdict
import requests

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ShopMind · Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
#  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d0f14 !important; }
section[data-testid="stMain"] { background: #0d0f14; }

[data-testid="stSidebar"] { background: #13161f !important; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span { color: #c8cad8 !important; }

[data-testid="stMetricValue"] { color: #9d96ff !important; font-family: 'Syne', sans-serif; font-size: 1.5rem !important; }
[data-testid="stMetricLabel"] { color: #555977 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.06em; }

[data-testid="stExpander"] { background: #13161f; border: 1px solid #1e2130 !important; border-radius: 12px; }
div[data-baseweb="select"] > div { background: #13161f !important; border-color: #2a2d40 !important; color: #e8eaf0 !important; }

/* ── Unified product image card ── */
.img-wrapper {
    width: 100%;
    aspect-ratio: 4 / 3;
    overflow: hidden;
    border-radius: 12px;
    background: #13161f;
    display: flex;
    align-items: center;
    justify-content: center;
}
.img-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    border-radius: 12px;
}

hr { border-color: #1e2130 !important; }
p, li { color: #c8cad8; }
h1,h2,h3,h4 { font-family: 'Syne', sans-serif; color: #fff !important; }

.pill {
    display: inline-block; background: #6c63ff22; color: #9d96ff;
    border: 1px solid #6c63ff44; border-radius: 99px;
    padding: 2px 12px; font-size: 0.78rem; font-weight: 600; margin-right: 4px;
}
.pill-red   { background: #ff636322; color: #ff9d96; border-color: #ff636344; }
.pill-green { background: #22ff8322; color: #96ffb4; border-color: #44ff8344; }

.pcard-body { padding: 0.6rem 0.2rem 0; }
.pcard-name {
    font-family: 'Syne', sans-serif; font-size: 0.78rem; font-weight: 600;
    color: #c8cad8; line-height: 1.35; min-height: 2.3rem; text-transform: capitalize;
}
.pcard-code  { font-size: 0.67rem; color: #454869; margin-top: 4px; }
.pcard-price { font-family: 'Syne', sans-serif; font-size: 0.9rem; font-weight: 700; color: #6c63ff; margin-top: 6px; }
.pcard-score { font-size: 0.68rem; color: #454869; margin-top: 2px; }

/* ── Rich Notification Panel ── */
@keyframes slideInRight {
    from { transform: translateX(110%); opacity: 0; }
    to   { transform: translateX(0);    opacity: 1; }
}
.notif-stack {
    position: fixed;
    bottom: 24px;
    right: 24px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 12px;
    width: 320px;
}
.notif-card {
    background: #12141e;
    border: 1px solid #252840;
    border-radius: 18px;
    overflow: hidden;
    box-shadow: 0 16px 48px rgba(0,0,0,0.6), 0 0 0 1px rgba(108,99,255,0.08);
    animation: slideInRight 0.45s cubic-bezier(.22,.68,0,1.15) forwards;
}
.notif-card::before {
    content: '';
    display: block;
    height: 3px;
    background: linear-gradient(90deg, #6c63ff, #a78bfa);
}
.notif-card.notif-green::before { background: linear-gradient(90deg, #22c55e, #4ade80); }
.notif-card.notif-amber::before { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.notif-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 14px 6px;
}
.notif-badge {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #6c63ff;
    background: #6c63ff18;
    border: 1px solid #6c63ff30;
    border-radius: 99px;
    padding: 2px 8px;
}
.notif-badge.green { color: #4ade80; background: #22c55e18; border-color: #22c55e30; }
.notif-badge.amber { color: #fbbf24; background: #f59e0b18; border-color: #f59e0b30; }
.notif-product {
    display: flex;
    gap: 12px;
    align-items: center;
    padding: 8px 14px 12px;
}
.notif-img-wrap {
    width: 68px;
    height: 68px;
    flex-shrink: 0;
    border-radius: 12px;
    overflow: hidden;
    background: #1a1d2e;
    position: relative;
}
.notif-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.notif-discount {
    position: absolute;
    top: 4px;
    left: 4px;
    background: #ef4444;
    color: #fff;
    font-size: 0.6rem;
    font-weight: 800;
    border-radius: 6px;
    padding: 2px 5px;
    letter-spacing: 0.03em;
    line-height: 1.2;
}
.notif-info { flex: 1; min-width: 0; }
.notif-label {
    font-size: 0.62rem;
    color: #555977;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 3px;
}
.notif-name {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    color: #e8eaf0;
    text-transform: capitalize;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.notif-prices {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-top: 5px;
}
.notif-price-now {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: #6c63ff;
}
.notif-price-was {
    font-size: 0.72rem;
    color: #454869;
    text-decoration: line-through;
}
.notif-saving {
    font-size: 0.65rem;
    color: #4ade80;
    font-weight: 600;
}
.notif-footer {
    border-top: 1px solid #1e2130;
    padding: 8px 14px;
    font-size: 0.68rem;
    color: #555977;
    display: flex;
    align-items: center;
    gap: 6px;
}
.notif-footer b { color: #9d96ff; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════
MODEL_DIR   = "recommender_models"
PLACEHOLDER = "https://placehold.co/400x300/13161f/6c63ff?text=🛍️"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


# ══════════════════════════════════════════════════════════════
#  IMAGE FETCHING  (from tjriba.py — multi-strategy)
# ══════════════════════════════════════════════════════════════
def _clean_query(raw: str) -> str:
    q = raw.lower()
    noise = ["set of", "pack of", "box of", "bag of", "large", "small",
             "mini", "giant", "assorted", "colour", "color", "mixed", "x"]
    for n in noise:
        q = q.replace(n, " ")
    q = re.sub(r"\d+", "", q)
    q = re.sub(r"[^a-z\s]", " ", q)
    q = " ".join(q.split()[:5])
    return q.strip()


def _scrape_google(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query + " product shop")
        url = f"https://www.google.com/search?q={encoded}&tbm=isch&hl=en&safe=active"
        resp = requests.get(url, headers={
            **_HEADERS,
            "Referer": "https://www.google.com/",
            "Cookie": "CONSENT=YES+; SOCS=CAE=",
        }, timeout=6)
        if resp.status_code != 200:
            return ""
        html = resp.text
        matches = re.findall(
            r'\["(https://[^"]+\.(?:jpg|jpeg|png|webp))"(?:,\d+){2}\]', html
        )
        for m in matches:
            if "gstatic" not in m and "google" not in m and len(m) > 30:
                return m
        matches2 = re.findall(r'ou=(https?://[^&"\\]+)', html)
        for m in matches2:
            decoded = urllib.parse.unquote(m)
            if decoded.startswith("http") and any(
                ext in decoded.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]
            ):
                return decoded
        matches3 = re.findall(r'imgurl=(https?://[^&"\\]+)', html)
        for m in matches3:
            decoded = urllib.parse.unquote(m)
            if decoded.startswith("http"):
                return decoded
    except Exception:
        pass
    return ""


def _scrape_bing(query: str) -> str:
    try:
        encoded = urllib.parse.quote(query + " product")
        url = f"https://www.bing.com/images/search?q={encoded}&form=HDRSC2&first=1"
        resp = requests.get(url, headers={**_HEADERS, "Referer": "https://www.bing.com/"}, timeout=6)
        if resp.status_code != 200:
            return ""
        html = resp.text
        matches = re.findall(r'murl&quot;:&quot;(https[^&]+)&quot;', html)
        if matches:
            return urllib.parse.unquote(matches[0])
        matches2 = re.findall(r'"(https://[^"]+\.(?:jpg|jpeg|png|webp))"', html)
        for m in matches2:
            if "bing" not in m and "microsoft" not in m:
                return m
    except Exception:
        pass
    return ""


def _duckduckgo(query: str) -> str:
    try:
        s = requests.Session()
        s.headers.update(_HEADERS)
        r = s.get(
            "https://duckduckgo.com/",
            params={"q": query, "iax": "images", "ia": "images"},
            timeout=5,
        )
        vqd = ""
        for pattern in [
            r'vqd=(["\'])([^"\']+)\1',
            r'"vqd"\s*:\s*"([^"]+)"',
            r'vqd=([\w-]+)',
        ]:
            m = re.search(pattern, r.text)
            if m:
                vqd = m.group(2) if m.lastindex == 2 else m.group(1)
                break
        if not vqd:
            return ""
        r2 = s.get(
            "https://duckduckgo.com/i.js",
            params={"q": query + " product", "vqd": vqd, "p": "1", "f": ",,,,,"},
            headers={**_HEADERS, "Referer": "https://duckduckgo.com/"},
            timeout=5,
        )
        if r2.ok:
            for item in r2.json().get("results", [])[:5]:
                url = item.get("image", "")
                if url.startswith("http") and not url.endswith(".gif"):
                    return url
    except Exception:
        pass
    return ""


def _loremflickr(query: str) -> str:
    try:
        tags = urllib.parse.quote(query.replace(" ", ",")[:60])
        url  = f"https://loremflickr.com/400/300/{tags}"
        r = requests.head(url, allow_redirects=True, timeout=5)
        if r.ok:
            return r.url if r.url != url else url
    except Exception:
        pass
    return ""


def _unsplash(query: str) -> str:
    try:
        tags = urllib.parse.quote(query[:60])
        url  = f"https://source.unsplash.com/400x300/?{tags}"
        r = requests.head(url, allow_redirects=True, timeout=5)
        if r.ok and r.url != url:
            return r.url
    except Exception:
        pass
    return ""


@st.cache_data(show_spinner=False, ttl=86400)
def fetch_image_url(raw_name: str) -> str:
    query = _clean_query(raw_name)
    if not query:
        return PLACEHOLDER
    strategies = [
        ("Google",      lambda q: _scrape_google(q)),
        ("Bing",        lambda q: _scrape_bing(q)),
        ("DuckDuckGo",  lambda q: _duckduckgo(q)),
        ("LoremFlickr", lambda q: _loremflickr(q)),
        ("Unsplash",    lambda q: _unsplash(q)),
    ]
    for name, fn in strategies:
        try:
            url = fn(query)
            if url and url.startswith("http"):
                return url
        except Exception:
            continue
    return PLACEHOLDER


# ══════════════════════════════════════════════════════════════
#  NOTIFICATION HELPERS  — rich persistent product cards
# ══════════════════════════════════════════════════════════════
_DISCOUNT_POOL = [30, 35, 40, 45, 20, 25]   # realistic-looking discounts

def _discount_price(price: float, pct: int) -> tuple[float, float]:
    """Return (sale_price, original_price) given a discount %."""
    original = round(price / (1 - pct / 100), 2)
    return price, original


def _notif_product_html(
    img_url: str,
    name: str,
    price: float,
    discount_pct: int,
    label: str,
    footer: str,
    color_cls: str = "",
) -> str:
    """Build one rich notification card as an HTML string."""
    sale_price, orig_price = _discount_price(price, discount_pct)
    saving = orig_price - sale_price
    placeholder = "https://placehold.co/68x68/1a1d2e/6c63ff?text=🛍"

    return f"""
<div class="notif-stack">
  <div class="notif-card {color_cls}">
    <div class="notif-header">
      <span class="notif-badge {color_cls.replace('notif-','')}">{label}</span>
    </div>
    <div class="notif-product">
      <div class="notif-img-wrap">
        <img src="{img_url}"
             onerror="this.src='{placeholder}'"
             alt="{name}" />
        <div class="notif-discount">−{discount_pct}%</div>
      </div>
      <div class="notif-info">
        <div class="notif-label">You may also like</div>
        <div class="notif-name">{name[:36]}{'…' if len(name) > 36 else ''}</div>
        <div class="notif-prices">
          <span class="notif-price-now">£{sale_price:.2f}</span>
          <span class="notif-price-was">£{orig_price:.2f}</span>
          <span class="notif-saving">Save £{saving:.2f}</span>
        </div>
      </div>
    </div>
    <div class="notif-footer">
      {footer}
    </div>
  </div>
</div>
"""


def notify_user_selected(user_id: int, n_bought: int, comm_size: int,
                         top_recs: list, prod_dict: dict) -> None:
    """Rich notification shown when a customer is selected — shows top recommended product."""
    if not top_recs:
        return
    code  = top_recs[0]
    info  = prod_dict.get(str(code), {})
    name  = info.get("description", str(code))
    price = info.get("price", 2.99)
    if price < 0.5:
        price = round(random.uniform(1.5, 8.0), 2)   # avoid near-zero prices in notif
    disc  = random.choice(_DISCOUNT_POOL)
    img   = fetch_image_url(name)
    footer = f"👥 <b>{comm_size}</b> shoppers in this community · {n_bought} past purchases"
    html  = _notif_product_html(img, name, price, disc,
                                label="Community Pick 🟢",
                                footer=footer,
                                color_cls="notif-green")
    st.markdown(html, unsafe_allow_html=True)


def notify_item_selected(item_name: str, item_code: str, price: float,
                         top_recs: list, prod_dict: dict) -> None:
    """Rich notification shown when a product is selected — shows a related product deal."""
    if not top_recs:
        return
    code  = top_recs[0]
    info  = prod_dict.get(str(code), {})
    name  = info.get("description", str(code))
    rprice = info.get("price", 2.99)
    if rprice < 0.5:
        rprice = round(random.uniform(1.5, 8.0), 2)
    disc  = random.choice(_DISCOUNT_POOL)
    img   = fetch_image_url(name)
    footer = f"🛍️ Customers who bought <b>{item_name[:22]}{'…' if len(item_name)>22 else ''}</b> also got this"
    html  = _notif_product_html(img, name, rprice, disc,
                                label="Customers Also Bought",
                                footer=footer,
                                color_cls="")
    st.markdown(html, unsafe_allow_html=True)


def notify_trending(top_recs: list, prod_dict: dict) -> None:
    """Rich notification for a trending item in the community with a deal badge."""
    if len(top_recs) < 2:
        return
    code  = top_recs[1]   # use 2nd rec to avoid repeating the first notif
    info  = prod_dict.get(str(code), {})
    name  = info.get("description", str(code))
    price = info.get("price", 2.99)
    if price < 0.5:
        price = round(random.uniform(1.5, 8.0), 2)
    disc  = random.choice(_DISCOUNT_POOL)
    img   = fetch_image_url(name)
    footer = "🔥 Trending in your community right now"
    html  = _notif_product_html(img, name, price, disc,
                                label="🔥 Hot Deal",
                                footer=footer,
                                color_cls="notif-amber")
    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  STUB CLASSES — required for pickle to deserialize the models
# ══════════════════════════════════════════════════════════════
class SVDRecommender:
    def __init__(self):
        self.user_factors = None
        self.item_factors = None

class ALSRecommender:
    def __init__(self):
        self.user_factors = None
        self.item_factors = None

class BM25Recommender:
    def __init__(self):
        self.item_scores = None


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    with open(os.path.join(MODEL_DIR, "models.pkl"), "rb") as f:
        data = pickle.load(f)
    product_info = pd.read_pickle(os.path.join(MODEL_DIR, "product_info.pkl"))
    prod_dict = {
        str(row["StockCode"]): {
            "description": str(row["Description"]).strip().title(),
            "price": float(row["UnitPrice"]),
        }
        for _, row in product_info.iterrows()
    }
    return data, prod_dict


# ══════════════════════════════════════════════════════════════
#  COMMUNITY RECOMMENDER
# ══════════════════════════════════════════════════════════════
class CommunityRecommender:
    def __init__(self, sim, partition, mode="user"):
        self.sim   = sim
        self.part  = partition
        self.comms = defaultdict(list)
        for idx, comm in partition.items():
            self.comms[comm].append(idx)

    def user_recs(self, user_idx, uim, top_n=5):
        if user_idx not in self.part:
            return []
        peers  = [u for u in self.comms[self.part[user_idx]] if u != user_idx]
        scores = defaultdict(float)
        for other in peers:
            w = self.sim[user_idx, other]
            if w:
                for idx in uim[other].nonzero()[1]:
                    scores[idx] += w
        owned = set(uim[user_idx].nonzero()[1])
        recs  = [(i, s) for i, s in scores.items() if i not in owned]
        return sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]

    def item_recs(self, seeds, exclude=None, top_n=5):
        exclude = exclude or set()
        scores  = defaultdict(float)
        for seed in seeds:
            if seed not in self.part:
                continue
            for cand in self.comms[self.part[seed]]:
                if cand != seed and cand not in exclude:
                    scores[cand] += self.sim[seed, cand]
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]


# ══════════════════════════════════════════════════════════════
#  SVD COMMUNITY RECOMMENDER
# ══════════════════════════════════════════════════════════════
class SVDCommunityRecommender:
    def __init__(self, svd_rec, user_partition, item_partition):
        self.svd    = svd_rec
        self.u_part = user_partition
        self.i_part = item_partition
        self._u_comms = defaultdict(list)
        for idx, c in user_partition.items():
            self._u_comms[c].append(idx)
        self._i_comms = defaultdict(list)
        for idx, c in item_partition.items():
            self._i_comms[c].append(idx)

    def recommend(self, user_idx, uim, top_n=10):
        all_scores     = self.svd.user_factors[user_idx].dot(self.svd.item_factors.T)
        already_bought = set(uim[user_idx].nonzero()[1])
        if user_idx in self.u_part:
            comm = self.u_part[user_idx]
            peer_items = set()
            for peer in self._u_comms[comm]:
                if peer != user_idx:
                    peer_items.update(uim[peer].nonzero()[1])
            candidates = peer_items - already_bought
        else:
            candidates = set(range(len(all_scores))) - already_bought
        if not candidates:
            candidates = set(range(len(all_scores))) - already_bought
        clist   = list(candidates)
        cscores = all_scores[clist]
        order   = np.argsort(cscores)[::-1][:top_n]
        return [(clist[i], float(cscores[i])) for i in order]


# ══════════════════════════════════════════════════════════════
#  ALS COMMUNITY RECOMMENDER
# ══════════════════════════════════════════════════════════════
class ALSCommunityRecommender:
    def __init__(self, als_rec, user_partition, item_partition):
        self.als    = als_rec
        self.u_part = user_partition
        self.i_part = item_partition
        self._u_comms = defaultdict(list)
        for idx, c in user_partition.items():
            self._u_comms[c].append(idx)

    def recommend(self, user_idx, uim, top_n=10):
        all_scores     = self.als.user_factors[user_idx].dot(self.als.item_factors.T)
        already_bought = set(uim[user_idx].nonzero()[1])
        if user_idx in self.u_part:
            comm = self.u_part[user_idx]
            peer_items = set()
            for peer in self._u_comms[comm]:
                if peer != user_idx:
                    peer_items.update(uim[peer].nonzero()[1])
            candidates = peer_items - already_bought
        else:
            candidates = set(range(len(all_scores))) - already_bought
        if not candidates:
            candidates = set(range(len(all_scores))) - already_bought
        clist   = list(candidates)
        cscores = all_scores[clist]
        order   = np.argsort(cscores)[::-1][:top_n]
        return [(clist[i], float(cscores[i])) for i in order]


# ══════════════════════════════════════════════════════════════
#  BM25 COMMUNITY RECOMMENDER
# ══════════════════════════════════════════════════════════════
class BM25CommunityRecommender:
    def __init__(self, bm25_rec, item_partition):
        self.bm25   = bm25_rec
        self.i_part = item_partition
        self._i_comms = defaultdict(list)
        for idx, c in item_partition.items():
            self._i_comms[c].append(idx)

    def recommend(self, user_idx, uim, top_n=10):
        seed_items = list(uim[user_idx].nonzero()[1])
        if not seed_items or self.bm25.item_scores is None:
            return []
        comm_candidates = set()
        for seed in seed_items:
            if seed in self.i_part:
                comm_candidates.update(self._i_comms[self.i_part[seed]])
        comm_candidates -= set(seed_items)
        if not comm_candidates:
            comm_candidates = set(range(self.bm25.item_scores.shape[1])) - set(seed_items)
        scores = np.zeros(self.bm25.item_scores.shape[1], dtype=np.float32)
        for item_idx in seed_items:
            rating = uim[user_idx, item_idx]
            scores += rating * self.bm25.item_scores[item_idx]
        clist   = list(comm_candidates)
        cscores = scores[clist]
        order   = np.argsort(cscores)[::-1][:top_n]
        return [(clist[i], float(cscores[i])) for i in order]


# ══════════════════════════════════════════════════════════════
#  PRODUCT CARD  (tjriba.py style — URL + .img-wrapper CSS)
# ══════════════════════════════════════════════════════════════
def product_card(col, code, prod_dict, score=None):
    info  = prod_dict.get(str(code), {})
    name  = info.get("description", str(code))
    price = info.get("price", 0.0)
    with col:
        img_url    = fetch_image_url(name)
        score_html = f'<div class="pcard-score">score {score:.3f}</div>' if score else ""
        st.markdown(
            f"""
            <div class="img-wrapper">
                <img src="{img_url}"
                     onerror="this.src='{PLACEHOLDER}'"
                     alt="{name}" />
            </div>
            <div class="pcard-body">
                <div class="pcard-name">{name[:58]}{'…' if len(name) > 58 else ''}</div>
                <div class="pcard-code">{code}</div>
                <div class="pcard-price">£{price:.2f}</div>
                {score_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def product_grid(items_scored, prod_dict, ncols=4, show_score=True):
    rows = [items_scored[i:i+ncols] for i in range(0, len(items_scored), ncols)]
    for row in rows:
        cols = st.columns(len(row))
        for col, (code, score) in zip(cols, row):
            product_card(col, code, prod_dict, score=score if show_score else None)
        st.markdown("<div style='margin-bottom:0.5rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR HELPERS
# ══════════════════════════════════════════════════════════════
MODEL_TYPES = ["Community (Jaccard)", "SVD + Community", "ALS + Community", "BM25 + Community"]
ALGOS       = ["Louvain", "Leiden", "Infomap", "FluidC"]

_MODEL_DESCRIPTIONS = {
    "Community (Jaccard)": "Pure community filtering using Jaccard similarity between users or items.",
    "SVD + Community":     "Matrix factorisation (SVD) scores filtered to community peers for diversity.",
    "ALS + Community":     "Implicit ALS latent factors filtered by community membership.",
    "BM25 + Community":    "BM25 item-item relevance scores restricted to co-purchase communities.",
}


def _show_metrics(metrics, model_type, algo, mode_key):
    st.markdown("### 📊 Metrics")
    if model_type == "Community (Jaccard)":
        m = metrics.get(mode_key, {}).get(algo, {})
    else:
        prefix = model_type.split(" ")[0]
        m = metrics.get("hybrid", {}).get(f"{prefix}+{algo}", {})

    ca, cb = st.columns(2)
    if model_type == "Community (Jaccard)":
        ca.metric("Modularity",  f"{m.get('modularity', 0):.3f}")
        cb.metric("Communities", int(m["n_communities"]) if "n_communities" in m else "—")
    if "precision@5" in m:
        cc, cd = st.columns(2)
        cc.metric("Precision@5", f"{m['precision@5']:.3f}")
        cd.metric("F1@5",        f"{m['f1@5']:.3f}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    st.markdown(
        """<div style="background:linear-gradient(135deg,#1a1d2e,#0f1118);
                    border:1px solid #1e2130;border-radius:18px;
                    padding:1.6rem 2.4rem;margin-bottom:1.4rem;">
            <span style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#fff;">
                Shop<span style="color:#6c63ff;">Mind</span>
            </span>
            <span style="color:#454869;font-size:0.88rem;margin-left:1rem;">
                Community-Based Product Recommender &nbsp;·&nbsp; Online Retail Dataset
            </span>
        </div>""",
        unsafe_allow_html=True,
    )

    if not os.path.exists(os.path.join(MODEL_DIR, "models.pkl")):
        st.error("⚠️ Models not found. Run the save snippet in the notebook first.")
        st.stop()

    with st.spinner("Loading models…"):
        data, prod_dict = load_models()

    uim         = data["user_item_matrix"]
    u_sim       = data["user_similarity"]
    i_sim       = data["item_similarity"]
    users       = data["users"]
    items       = data["items"]
    user_to_idx = data["user_to_idx"]
    item_to_idx = data["item_to_idx"]
    idx_to_item = data["idx_to_item"]
    u_parts     = data["user_partitions"]
    i_parts     = data["item_partitions"]
    svd_rec     = data["svd"]
    als_rec     = data["als"]
    bm25_rec    = data["bm25"]
    metrics     = data["metrics"]

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.divider()

        mode = st.radio(
            "Mode",
            ["👤 User-Based", "📦 Item-Based"],
            help=(
                "**User-Based** — recommend products to a customer based on community peers.\n\n"
                "**Item-Based** — recommend items similar to a selected product."
            ),
        )

        model_type = st.selectbox(
            "Model",
            MODEL_TYPES,
            help="\n\n".join(f"**{k}**: {v}" for k, v in _MODEL_DESCRIPTIONS.items()),
        )

        algo  = st.selectbox("Community Algorithm", ALGOS)
        top_n = st.slider("# Recommendations", 3, 12, 5)

 

 

    if "User" in mode:
        page_user(uim, u_sim, users, user_to_idx, idx_to_item,
                  u_parts, i_parts, prod_dict, algo, top_n,
                  model_type, svd_rec, als_rec, bm25_rec)
    else:
        page_item(uim, i_sim, items, item_to_idx, idx_to_item,
                  i_parts, prod_dict, algo, top_n,
                  model_type, svd_rec, als_rec, bm25_rec, user_to_idx, u_parts)


# ══════════════════════════════════════════════════════════════
#  USER-BASED PAGE
# ══════════════════════════════════════════════════════════════
def page_user(uim, u_sim, users, user_to_idx, idx_to_item,
              u_parts, i_parts, prod_dict, algo, top_n,
              model_type, svd_rec, als_rec, bm25_rec):

    pill_model = model_type.replace(" + ", "+")
    st.markdown(
        f"#### 👤 Pick a Customer &nbsp;"
        f"<span class='pill'>{algo}</span>"
        f"<span class='pill pill-green'>{pill_model}</span>"
        f"<span class='pill pill-red'>User-Based</span>",
        unsafe_allow_html=True,
    )

    user_ids = sorted([int(u) for u in users])
    sel_user = st.selectbox("Customer ID", user_ids, label_visibility="collapsed")
    user_idx = user_to_idx[float(sel_user)]

    user_item_idx   = list(uim[user_idx].nonzero()[1])
    user_item_codes = [str(idx_to_item[i]) for i in user_item_idx]

    partition = u_parts[algo]
    comm_id   = partition.get(user_idx, "—")
    comm_size = sum(1 for v in partition.values() if v == comm_id)

    # ── Notification: user confirmed ──────────────────────────
    _prev_user_key = f"_prev_user_{algo}_{model_type}"
    if st.session_state.get(_prev_user_key) != sel_user:
        st.session_state[_prev_user_key] = sel_user
        # Lightweight quick-peek at top recs for the notification pills
        try:
            _quick_rec = CommunityRecommender(u_sim, partition)
            _quick_recs = _quick_rec.user_recs(user_idx, uim, top_n=3)
            _preview_codes = [str(idx_to_item[i]) for i, _ in _quick_recs]
        except Exception:
            _preview_codes = user_item_codes[:3]
        notify_user_selected(sel_user, len(user_item_codes), comm_size,
                             _preview_codes, prod_dict)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customer ID",     sel_user)
    c2.metric("Products Bought", len(user_item_codes))
    c3.metric("Community #",     comm_id)
    c4.metric("Community Size",  comm_size)

    st.divider()
    with st.expander(f"🛒 Purchase history — {len(user_item_codes)} items", expanded=False):
        if user_item_codes:
            product_grid([(c, None) for c in user_item_codes[:8]], prod_dict, ncols=4, show_score=False)
        else:
            st.write("No purchase history found.")

    st.divider()
    st.markdown(f"#### ✨ Top {top_n} Recommendations")

    with st.spinner("Finding recommendations…"):
        if model_type == "Community (Jaccard)":
            rec  = CommunityRecommender(u_sim, partition)
            recs = rec.user_recs(user_idx, uim, top_n=top_n)
        elif model_type == "SVD + Community":
            rec  = SVDCommunityRecommender(svd_rec, u_parts[algo], i_parts[algo])
            recs = rec.recommend(user_idx, uim, top_n=top_n)
        elif model_type == "ALS + Community":
            rec  = ALSCommunityRecommender(als_rec, u_parts[algo], i_parts[algo])
            recs = rec.recommend(user_idx, uim, top_n=top_n)
        else:  # BM25 + Community
            rec  = BM25CommunityRecommender(bm25_rec, i_parts[algo])
            recs = rec.recommend(user_idx, uim, top_n=top_n)

    if not recs:
        st.warning("No recommendations found. Try a different algorithm, model, or customer.")
        return

    scored = [(str(idx_to_item[i]), s) for i, s in recs]
    product_grid(scored, prod_dict, ncols=min(top_n, 5))

    # ── Notification: trending in community ───────────────────
    rec_codes = [c for c, _ in scored]
    notify_trending(rec_codes, prod_dict)


# ══════════════════════════════════════════════════════════════
#  ITEM-BASED PAGE
# ══════════════════════════════════════════════════════════════
def page_item(uim, i_sim, items, item_to_idx, idx_to_item,
              i_parts, prod_dict, algo, top_n,
              model_type, svd_rec, als_rec, bm25_rec, user_to_idx, u_parts):

    pill_model = model_type.replace(" + ", "+")
    st.markdown(
        f"#### 📦 Pick a Product &nbsp;"
        f"<span class='pill'>{algo}</span>"
        f"<span class='pill pill-green'>{pill_model}</span>"
        f"<span class='pill pill-red'>Item-Based</span>",
        unsafe_allow_html=True,
    )

    options = {}
    for code in items:
        if str(code) not in item_to_idx:
            continue
        info  = prod_dict.get(str(code), {})
        label = f"{info.get('description', str(code))[:55]}  [{code}]"
        options[label] = str(code)

    if not options:
        st.warning("No products available in the similarity matrix.")
        return

    sel_label = st.selectbox("Product", sorted(options.keys()), label_visibility="collapsed")
    sel_code  = options[sel_label]

    item_idx  = item_to_idx[sel_code]
    info      = prod_dict.get(sel_code, {})
    name      = info.get("description", sel_code)
    price     = info.get("price", 0.0)
    partition = i_parts[algo]
    comm_id   = partition.get(item_idx, "—")
    comm_size = sum(1 for v in partition.values() if v == comm_id)

    # ── Notification: item confirmed ──────────────────────────
    _prev_item_key = f"_prev_item_{algo}_{model_type}"
    if st.session_state.get(_prev_item_key) != sel_code:
        st.session_state[_prev_item_key] = sel_code
        try:
            _quick_rec = CommunityRecommender(i_sim, partition, mode="item")
            _quick_recs = _quick_rec.item_recs({item_idx}, exclude={item_idx}, top_n=3)
            _preview_codes = [str(idx_to_item[i]) for i, _ in _quick_recs]
        except Exception:
            _preview_codes = []
        notify_item_selected(name, sel_code, price, _preview_codes, prod_dict)

    st.divider()
    col_img, col_info = st.columns([1, 3], gap="large")
    with col_img:
        img_url = fetch_image_url(name)
        st.markdown(
            f"""
            <div class="img-wrapper">
                <img src="{img_url}"
                     onerror="this.src='{PLACEHOLDER}'"
                     alt="{name}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_info:
        st.markdown(
            f"""
            <div style="color:#6c63ff;font-size:0.72rem;font-weight:600;
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">
                Selected Product
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:1.55rem;
                        font-weight:800;color:#fff;text-transform:capitalize;line-height:1.2;">
                {name}
            </div>
            <div style="color:#9d96ff;margin-top:0.6rem;font-size:0.9rem;">
                £{price:.2f} &nbsp;·&nbsp; Code: {sel_code}
                &nbsp;·&nbsp; Community #{comm_id} ({comm_size} items)
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(f"#### ✨ Customers Also Bought — Top {top_n}")

    def _most_frequent_buyer():
        col = np.array(uim.getcol(item_idx).todense()).flatten()
        if col.max() == 0:
            return 0
        return int(np.argmax(col))

    with st.spinner("Finding similar products…"):
        if model_type == "Community (Jaccard)":
            rec  = CommunityRecommender(i_sim, partition, mode="item")
            recs = rec.item_recs({item_idx}, exclude={item_idx}, top_n=top_n)
        elif model_type == "SVD + Community":
            proxy_user = _most_frequent_buyer()
            rec  = SVDCommunityRecommender(svd_rec, u_parts[algo], i_parts[algo])
            recs = rec.recommend(proxy_user, uim, top_n=top_n)
        elif model_type == "ALS + Community":
            proxy_user = _most_frequent_buyer()
            rec  = ALSCommunityRecommender(als_rec, u_parts[algo], i_parts[algo])
            recs = rec.recommend(proxy_user, uim, top_n=top_n)
        else:  # BM25 + Community
            proxy_user = _most_frequent_buyer()
            rec  = BM25CommunityRecommender(bm25_rec, i_parts[algo])
            recs = rec.recommend(proxy_user, uim, top_n=top_n)

    if not recs:
        st.warning("No similar items found. Try a different algorithm, model, or product.")
        return

    scored = [(str(idx_to_item[i]), s) for i, s in recs]
    product_grid(scored, prod_dict, ncols=min(top_n, 5))

    # ── Notification: trending deal nudge ─────────────────────
    rec_codes = [c for c, _ in scored]
    notify_trending(rec_codes, prod_dict)


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
