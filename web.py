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
    st.warning("⚠️ Không tìm thấy file style.css để làm đẹp giao diện.")

# --- 1. CÁC HÀM THUẬT TOÁN DỰA TRÊN LÝ THUYẾT TẬP HỢP (SET THEORY - JACCARD SIMILARITY) ---

def validate_columns(df, required_columns, df_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột trong {df_name}: {', '.join(missing)}")


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

    # Lọc cơ bản: cùng category, khác sản phẩm gốc, trong khoảng giá
    min_p = target_price * (1 - price_tolerance)
    max_p = target_price * (1 + price_tolerance)

    conditions = [
        pl.col("category") == source.get("category", ""),
        pl.col("item_id") != target_item_id,
    ]
    if price_tolerance > 0.0:
        conditions.append(pl.col("price").is_between(min_p, max_p))

    candidates = items_df.filter(pl.all_horizontal(conditions))

    # Chấm điểm từ các cột phụ (Set Theory cơ bản) - Khớp cột nào được cộng 1 điểm
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

    # Tạo cột diễn giải lý do gợi ý
    reason_parts = [
        pl.when(pl.col("category_l3") == source.get("category_l3", "")).then(pl.lit("Cùng category_l3")).otherwise(None),
        pl.when(pl.col("category_l2") == source.get("category_l2", "")).then(pl.lit("Cùng category_l2")).otherwise(None),
        pl.when(pl.col("brand") == source.get("brand", "")).then(pl.lit("Cùng thương hiệu")).otherwise(None),
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
    # 1. Gom nhóm tạo Giỏ hàng (Basket)
    # Logic: Cùng customer_id, cùng ngày updated_date, cùng channel
    basket_parts = [pl.col("customer_id").cast(pl.String), pl.lit("_")]

    if "updated_date" in transactions_df.columns:
        basket_parts.append(pl.col("updated_date").dt.date().cast(pl.String))

    if "channel" in transactions_df.columns:
        basket_parts.extend([pl.lit("_"), pl.col("channel").cast(pl.String)])

    df_orders = transactions_df.with_columns(
        pl.concat_str(basket_parts).alias("basket_id")
    )

    # 2. Tìm danh sách các giỏ hàng chứa sản phẩm gốc (Tập A)
    baskets_with_A = df_orders.filter(pl.col("item_id") == target_item_id).select("basket_id").unique()
    freq_A = baskets_with_A.height

    if freq_A == 0:
        return pl.DataFrame()

    # 3. Lấy các sản phẩm B nằm trong các giỏ hàng trên (Giao của A và B)
    co_purchases = (
        df_orders.join(baskets_with_A, on="basket_id", how="inner")
        .filter(pl.col("item_id") != target_item_id)
        .group_by("item_id").len().rename({"len": "intersection_count"})
        .filter(pl.col("intersection_count") >= min_co_purchases)
    )

    if co_purchases.is_empty():
        return pl.DataFrame()

    # 4. Tìm tổng số lần bán của các sản phẩm B (Tập B)
    valid_b_items = co_purchases["item_id"].to_list()
    freq_B_df = (
        df_orders.filter(pl.col("item_id").is_in(valid_b_items))
        .group_by("item_id").agg(pl.col("basket_id").n_unique().alias("freq_B"))
    )

    # 5. Áp dụng Jaccard Similarity để tính điểm
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

    # 6. Join với bảng items để lấy thông tin hiển thị
    result_df = fbt_stats.join(items_df, on="item_id", how="inner")

    return result_df


# --- 2. GIAO DIỆN VÀ TẢI DỮ LIỆU ---

st.markdown('<div class="main-title">🛒 Product Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Hệ thống gợi ý thông minh dựa trên <b>Jaccard Similarity Index</b> '
    '– đo lường độ tương đồng giữa các tập hợp dữ liệu</div>',
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_items = os.path.join(base_dir, 'items.parquet')
    path_trans = os.path.join(base_dir, 'transactions-2025-12.parquet')

    if not os.path.exists(path_items) or not os.path.exists(path_trans):
        raise FileNotFoundError("Không tìm thấy các file dữ liệu .parquet trong thư mục.")

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
    st.error(f"❌ Lỗi tải dữ liệu: {e}")
    st.stop()

try:
    # --- SIDEBAR ---
    st.sidebar.markdown("## ⚙️ Cài đặt gợi ý")
    st.sidebar.markdown("---")

    # Lọc sản phẩm đang bán (nếu có cột sale_status)
    if "sale_status" in items.columns:
        active_items = items.filter(pl.col("sale_status") == 1)
    else:
        active_items = items

    # Tạo danh sách sản phẩm với tên hiển thị: "Tên sản phẩm (Thương hiệu)"
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
        st.warning("⚠️ Không có sản phẩm nào để hiển thị gợi ý.")
        st.stop()

    display_names = product_list["display_name"].to_list()
    name_to_id = dict(zip(
        product_list["display_name"].to_list(),
        product_list["item_id"].to_list()
    ))

    selected_name = st.sidebar.selectbox("🔎 Chọn sản phẩm", display_names, index=0)
    selected_id = name_to_id[selected_name]

    st.sidebar.markdown("---")
    n_results = st.sidebar.slider("📊 Số lượng gợi ý", 1, 10, 5)
    tolerance = st.sidebar.slider("💰 Dung sai giá (%)", 0, 100, 20) / 100

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="color:#4b5563;font-size:0.75rem;text-align:center;">'
        'Powered by Polars · Jaccard Similarity</div>',
        unsafe_allow_html=True
    )

    # --- HIỂN THỊ SẢN PHẨM ĐANG CHỌN ---
    source_data = items.filter(pl.col("item_id") == selected_id).row(0, named=True)

    st.markdown('<div class="section-tag">📦 Sản phẩm đang xem</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tên sản phẩm", source_data['category'])
    col2.metric("Thương hiệu", source_data['brand'])
    col3.metric("Giá bán", f"{float(source_data['price']):,.0f} ₫")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- TABS GỢI Ý ---
    tab1, tab2 = st.tabs(["🔍  Sản phẩm tương tự", "🤝  Thường được mua cùng"])

    with tab1:
        st.markdown(
            '<div class="tab-desc">💡 Gợi ý dựa trên <b>Set Theory</b>: so khớp <b>danh mục</b>, '
            '<b>thương hiệu</b> và <b>tầm giá</b> tương đồng. Mỗi thuộc tính khớp = 1 điểm.</div>',
            unsafe_allow_html=True
        )
        similar_res = get_similar_products(selected_id, items, top_n=n_results, price_tolerance=tolerance)
        if not similar_res.is_empty():
            max_score = 5 if "manufacturer" in items.columns else 4
            display_cols = [c for c in ["item_id", "category", "brand", "price", "match_score", "recommendation_reason"] if c in similar_res.columns]
            st.dataframe(
                similar_res.select(display_cols),
                width="stretch",
                hide_index=True,
                column_config={
                    "item_id": st.column_config.TextColumn("Mã SP"),
                    "category": st.column_config.TextColumn("Tên sản phẩm"),
                    "brand": st.column_config.TextColumn("Thương hiệu"),
                    "price": st.column_config.NumberColumn("Giá (₫)", format="%,.0f"),
                    "match_score": st.column_config.ProgressColumn(
                        "Điểm tương đồng", min_value=0, max_value=max_score, format="%d"
                    ),
                    "recommendation_reason": st.column_config.TextColumn("Lý do gợi ý"),
                }
            )
        else:
            st.warning("⚠️ Không tìm thấy sản phẩm tương tự phù hợp.")

    with tab2:
        st.markdown(
            '<div class="tab-desc">🛒 Gợi ý dựa trên <b>Jaccard Similarity Index</b> từ lịch sử giỏ hàng: '
            'J(A,B) = |A∩B| / |A∪B|. Điểm càng cao = hai sản phẩm càng hay được mua cùng nhau.</div>',
            unsafe_allow_html=True
        )
        fbt_res = get_frequently_bought_together(selected_id, transactions, items, top_n=n_results)
        if not fbt_res.is_empty():
            display_cols = [c for c in ["item_id", "category", "brand", "price", "jaccard_score", "intersection_count"] if c in fbt_res.columns]
            st.dataframe(
                fbt_res.select(display_cols),
                width="stretch",
                hide_index=True,
                column_config={
                    "item_id": st.column_config.TextColumn("Mã SP"),
                    "category": st.column_config.TextColumn("Tên sản phẩm"),
                    "brand": st.column_config.TextColumn("Thương hiệu"),
                    "price": st.column_config.NumberColumn("Giá (₫)", format="%,.0f"),
                    "jaccard_score": st.column_config.NumberColumn("Jaccard Score", format="%.4f"),
                    "intersection_count": st.column_config.NumberColumn("Lần mua kèm", format="%d"),
                }
            )
        else:
            st.info("ℹ️ Sản phẩm này chưa có dữ liệu mua kèm.")

except Exception as e:
    st.error(f"❌ Lỗi xử lý gợi ý: {e}")
