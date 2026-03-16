import streamlit as st
import polars as pl
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ Thống Gợi Ý Sản Phẩm",
    layout="wide",
    page_icon="🛒"
)

# --- HÀM LOAD CSS TỪ FILE RIÊNG ---
def local_css(file_name):
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Lỗi khi tải file CSS: {e}")

base_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(base_dir, 'style.css')

if os.path.exists(css_path):
    local_css(css_path)
else:
    st.warning("Khong tim thay file style.css de lam dep giao dien.")

# --- 1. CÁC HÀM THUẬT TOÁN DỰA TRÊN LÝ THUYẾT TẬP HỢP (SET THEORY - JACCARD SIMILARITY) ---

def validate_columns(df, required_columns, df_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Thieu cot trong {df_name}: {', '.join(missing)}")


def get_similar_products(target_item_id, items_df, top_n=5, price_tolerance=0.2):
    """
    Gợi ý sản phẩm tương tự dựa trên Set Theory.
    So khớp các thuộc tính: category_l1, category_l2, category_l3, brand, manufacturer.
    Mỗi thuộc tính khớp được cộng 1 điểm.
    """
    source_item = items_df.filter(pl.col("item_id") == target_item_id)
    if source_item.is_empty():
        return pl.DataFrame()

    source = source_item.row(0, named=True)
    target_price = float(source.get("price", 0))

    min_p = target_price * (1 - price_tolerance)
    max_p = target_price * (1 + price_tolerance)

    conditions = [
        pl.col("category") == source.get("category", ""),
        pl.col("item_id") != target_item_id,
    ]
    if price_tolerance > 0.0:
        conditions.append(pl.col("price").is_between(min_p, max_p))

    candidates = items_df.filter(pl.all_horizontal(conditions))

    score_parts = [
        (pl.col("category_l1") == source.get("category_l1", "")).cast(pl.Int32),
        (pl.col("category_l2") == source.get("category_l2", "")).cast(pl.Int32),
        (pl.col("category_l3") == source.get("category_l3", "")).cast(pl.Int32),
        (pl.col("brand") == source.get("brand", "")).cast(pl.Int32),
    ]
    if "manufacturer" in items_df.columns:
        score_parts.append(
            (pl.col("manufacturer") == source.get("manufacturer", "")).cast(pl.Int32)
        )

    score_expr = score_parts[0]
    for part in score_parts[1:]:
        score_expr = score_expr + part

    reason_parts = [
        pl.when(pl.col("category_l3") == source.get("category_l3", "")).then(pl.lit("Cung category_l3")).otherwise(None),
        pl.when(pl.col("category_l2") == source.get("category_l2", "")).then(pl.lit("Cung category_l2")).otherwise(None),
        pl.when(pl.col("brand") == source.get("brand", "")).then(pl.lit("Cung thuong hieu")).otherwise(None),
    ]

    result_df = (
        candidates.with_columns(match_score=score_expr)
        .with_columns(recommendation_reason=pl.concat_str(reason_parts, separator=" | ", ignore_nulls=True))
        .sort(by=["match_score", "price"], descending=[True, False])
        .unique(subset=["item_id"], keep="first", maintain_order=True)
        .limit(top_n)
    )

    return result_df


def get_frequently_bought_together(target_item_id, transactions_df, items_df, top_n=5, min_co_purchases=2):
    """
    Gợi ý sản phẩm thường mua cùng dựa trên Jaccard Similarity Index.
    J(A, B) = |A ∩ B| / |A ∪ B| = |A ∩ B| / (|A| + |B| - |A ∩ B|)
    """
    basket_parts = [pl.col("customer_id").cast(pl.String), pl.lit("_")]

    if "updated_date" in transactions_df.columns:
        basket_parts.append(pl.col("updated_date").dt.date().cast(pl.String))

    if "channel" in transactions_df.columns:
        basket_parts.extend([pl.lit("_"), pl.col("channel").cast(pl.String)])

    df_orders = transactions_df.with_columns(
        pl.concat_str(basket_parts).alias("basket_id")
    )

    baskets_with_A = df_orders.filter(pl.col("item_id") == target_item_id).select("basket_id").unique()
    freq_A = baskets_with_A.height

    if freq_A == 0:
        return pl.DataFrame()

    co_purchases = (
        df_orders.join(baskets_with_A, on="basket_id", how="inner")
        .filter(pl.col("item_id") != target_item_id)
        .group_by("item_id").len().rename({"len": "intersection_count"})
        .filter(pl.col("intersection_count") >= min_co_purchases)
    )

    if co_purchases.is_empty():
        return pl.DataFrame()

    valid_b_items = co_purchases["item_id"].to_list()
    freq_B_df = (
        df_orders.filter(pl.col("item_id").is_in(valid_b_items))
        .group_by("item_id").agg(pl.col("basket_id").n_unique().alias("freq_B"))
    )

    fbt_stats = (
        co_purchases.join(freq_B_df, on="item_id", how="inner")
        .with_columns(
            jaccard_score=(
                pl.col("intersection_count") /
                (freq_A + pl.col("freq_B") - pl.col("intersection_count"))
            )
        )
        .sort(by=["jaccard_score", "intersection_count"], descending=[True, True])
        .limit(top_n)
    )

    result_df = fbt_stats.join(items_df, on="item_id", how="inner")

    return result_df


# --- 2. HÀM RENDER PRODUCT CARDS ---

def render_product_card(rank, name, brand, price, score_value, score_label, reason=""):
    price_fmt = f"{float(price):,.0f}"
    reason_html = f'<div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">{reason}</div>' if reason else ""
    return f'''
    <div class="result-card" style="animation-delay:{rank * 0.07}s;">
        <div class="rc-rank">#{rank}</div>
        <div class="rc-info">
            <div class="rc-name">{name}</div>
            <div class="rc-brand">{brand}</div>
            {reason_html}
        </div>
        <div class="rc-price">{price_fmt} &#8363;</div>
        <div class="rc-score">
            <div class="rc-score-value">{score_value}</div>
            <div class="rc-score-label">{score_label}</div>
        </div>
    </div>
    '''


# --- 3. GIAO DIỆN VÀ TẢI DỮ LIỆU ---

st.markdown('<div class="main-title">Product Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">He thong goi y thong minh dua tren <b>Jaccard Similarity Index</b> '
    '&ndash; do luong do tuong dong giua cac tap hop du lieu</div>',
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_items = os.path.join(base_dir, 'items.parquet')
    path_trans = os.path.join(base_dir, 'transactions-2025-12.parquet')

    if not os.path.exists(path_items) or not os.path.exists(path_trans):
        raise FileNotFoundError("Khong tim thay cac file du lieu .parquet trong thu muc.")

    df_items = pl.read_parquet(path_items)
    df_trans = pl.read_parquet(path_trans)

    validate_columns(
        df_items,
        ["item_id", "category", "price", "category_l1", "category_l2", "category_l3", "brand"],
        "items",
    )
    validate_columns(df_trans, ["item_id", "customer_id"], "transactions")

    return df_items, df_trans

try:
    items, transactions = load_data()
except Exception as e:
    st.error(f"Loi tai du lieu: {e}")
    st.stop()

try:
    # --- SIDEBAR ---
    st.sidebar.markdown("## Cai dat goi y")
    st.sidebar.markdown("---")

    if "sale_status" in items.columns:
        active_items = items.filter(pl.col("sale_status") == 1)
    else:
        active_items = items

    product_list = (
        active_items
        .with_columns(
            display_name=pl.concat_str([
                pl.col("category"),
                pl.lit(" ("),
                pl.col("brand"),
                pl.lit(")")
            ])
        )
        .select(["item_id", "display_name", "category", "brand", "price"])
        .unique(subset=["display_name"], keep="first")
        .sort("display_name")
    )

    if product_list.is_empty():
        st.warning("Khong co san pham nao de hien thi goi y.")
        st.stop()

    display_names = product_list["display_name"].to_list()
    name_to_id = dict(zip(
        product_list["display_name"].to_list(),
        product_list["item_id"].to_list()
    ))

    selected_name = st.sidebar.selectbox("Chon san pham", display_names, index=0)
    selected_id = name_to_id[selected_name]

    st.sidebar.markdown("---")
    n_results = st.sidebar.slider("So luong goi y", 1, 10, 5)
    tolerance = st.sidebar.slider("Dung sai gia (%)", 0, 100, 20) / 100

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div class="sidebar-footer">Powered by <span>Polars</span> &middot; <span>Jaccard</span></div>',
        unsafe_allow_html=True
    )

    # --- PRODUCT HERO CARD ---
    source_data = items.filter(pl.col("item_id") == selected_id).row(0, named=True)
    price_fmt = f"{float(source_data['price']):,.0f}"

    cat_l1 = source_data.get('category_l1', '')
    cat_l2 = source_data.get('category_l2', '')
    cat_l3 = source_data.get('category_l3', '')

    st.markdown(f'''
    <div class="product-hero">
        <div class="hero-label">San pham dang xem</div>
        <div class="hero-name">{source_data['category']}</div>
        <div class="hero-details">
            <div class="detail-item">
                <div class="detail-label">Thuong hieu</div>
                <div class="detail-value">{source_data['brand']}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Gia ban</div>
                <div class="detail-value price">{price_fmt} &#8363;</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Danh muc</div>
                <div class="detail-value">{cat_l1} &rsaquo; {cat_l2} &rsaquo; {cat_l3}</div>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # --- STATS ROW ---
    total_products = items.height
    total_transactions = transactions.height
    total_brands = items["brand"].n_unique()

    st.markdown(f'''
    <div class="stats-row">
        <div class="stat-card">
            <div class="stat-icon">📦</div>
            <div class="stat-value">{total_products:,}</div>
            <div class="stat-label">San pham</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🛍️</div>
            <div class="stat-value">{total_transactions:,}</div>
            <div class="stat-label">Giao dich</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🏷️</div>
            <div class="stat-value">{total_brands:,}</div>
            <div class="stat-label">Thuong hieu</div>
        </div>
        <div class="stat-card">
            <div class="stat-icon">🎯</div>
            <div class="stat-value">{n_results}</div>
            <div class="stat-label">Goi y</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # --- TABS ---
    tab1, tab2 = st.tabs(["  San pham tuong tu  ", "  Thuong duoc mua cung  "])

    with tab1:
        st.markdown(
            '<div class="tab-desc">Goi y dua tren <b>Set Theory</b>: so khop <b>danh muc</b>, '
            '<b>thuong hieu</b> va <b>tam gia</b> tuong dong. Moi thuoc tinh khop = 1 diem.</div>',
            unsafe_allow_html=True
        )
        similar_res = get_similar_products(selected_id, items, top_n=n_results, price_tolerance=tolerance)
        if not similar_res.is_empty():
            max_score = 5 if "manufacturer" in items.columns else 4
            count = similar_res.height
            st.markdown(
                f'<div class="result-count">Tim thay {count} san pham tuong tu</div>',
                unsafe_allow_html=True
            )

            cards_html = ""
            for i, row in enumerate(similar_res.iter_rows(named=True)):
                score_display = f"{row['match_score']}/{max_score}"
                reason = row.get('recommendation_reason', '')
                cards_html += render_product_card(
                    rank=i + 1,
                    name=row['category'],
                    brand=row['brand'],
                    price=row['price'],
                    score_value=score_display,
                    score_label="Diem",
                    reason=reason,
                )
            st.markdown(cards_html, unsafe_allow_html=True)

            with st.expander("Xem bang du lieu chi tiet"):
                display_cols = [c for c in ["item_id", "category", "brand", "price", "match_score", "recommendation_reason"] if c in similar_res.columns]
                st.dataframe(
                    similar_res.select(display_cols),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "item_id": st.column_config.TextColumn("Ma SP"),
                        "category": st.column_config.TextColumn("Ten san pham"),
                        "brand": st.column_config.TextColumn("Thuong hieu"),
                        "price": st.column_config.NumberColumn("Gia", format="%,.0f"),
                        "match_score": st.column_config.ProgressColumn(
                            "Diem tuong dong", min_value=0, max_value=max_score, format="%d"
                        ),
                        "recommendation_reason": st.column_config.TextColumn("Ly do goi y"),
                    }
                )
        else:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">🔍</div>'
                '<div class="empty-text">Khong tim thay san pham tuong tu phu hop.</div>'
                '</div>',
                unsafe_allow_html=True
            )

    with tab2:
        st.markdown(
            '<div class="tab-desc">Goi y dua tren <b>Jaccard Similarity Index</b> tu lich su gio hang: '
            'J(A,B) = |A&cap;B| / |A&cup;B|. Diem cang cao = hai san pham cang hay duoc mua cung nhau.</div>',
            unsafe_allow_html=True
        )
        fbt_res = get_frequently_bought_together(selected_id, transactions, items, top_n=n_results)
        if not fbt_res.is_empty():
            count = fbt_res.height
            st.markdown(
                f'<div class="result-count">Tim thay {count} san pham thuong mua kem</div>',
                unsafe_allow_html=True
            )

            cards_html = ""
            for i, row in enumerate(fbt_res.iter_rows(named=True)):
                score_display = f"{row['jaccard_score']:.4f}"
                reason = f"Mua kem {row['intersection_count']} lan"
                cards_html += render_product_card(
                    rank=i + 1,
                    name=row['category'],
                    brand=row['brand'],
                    price=row['price'],
                    score_value=score_display,
                    score_label="Jaccard",
                    reason=reason,
                )
            st.markdown(cards_html, unsafe_allow_html=True)

            with st.expander("Xem bang du lieu chi tiet"):
                display_cols = [c for c in ["item_id", "category", "brand", "price", "jaccard_score", "intersection_count"] if c in fbt_res.columns]
                st.dataframe(
                    fbt_res.select(display_cols),
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "item_id": st.column_config.TextColumn("Ma SP"),
                        "category": st.column_config.TextColumn("Ten san pham"),
                        "brand": st.column_config.TextColumn("Thuong hieu"),
                        "price": st.column_config.NumberColumn("Gia", format="%,.0f"),
                        "jaccard_score": st.column_config.NumberColumn("Jaccard Score", format="%.4f"),
                        "intersection_count": st.column_config.NumberColumn("Lan mua kem", format="%d"),
                    }
                )
        else:
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">🛒</div>'
                '<div class="empty-text">San pham nay chua co du lieu mua kem.</div>'
                '</div>',
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"Loi xu ly goi y: {e}")
