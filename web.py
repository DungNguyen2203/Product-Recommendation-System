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
    """Đọc file CSS và nạp vào Streamlit"""
    try:
        with open(file_name, encoding="utf-8") as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Lỗi khi tải file CSS: {e}")

# Lấy đường dẫn thư mục hiện tại và nạp CSS
base_dir = os.path.dirname(os.path.abspath(__file__))
css_path = os.path.join(base_dir, 'style.css')

if os.path.exists(css_path):
    local_css(css_path)
else:
    st.warning("⚠️ Không tìm thấy file style.css để làm đẹp giao diện.")

# --- 1. CÁC HÀM THUẬT TOÁN ---

def get_similar_products(target_item_id, items_df, top_n=5, price_tolerance=0.2):
    source_item = items_df.filter(pl.col("item_id") == target_item_id)
    if source_item.is_empty():
        return pl.DataFrame()
    
    source = source_item.row(0, named=True)
    
    # Tính điểm tương đồng dựa trên phân cấp danh mục và thương hiệu
    score_expr = (
        (pl.col("category_l1") == source.get("category_l1", "")).cast(pl.Int32) * 1 +
        (pl.col("category_l2") == source.get("category_l2", "")).cast(pl.Int32) * 2 +
        (pl.col("category_l3") == source.get("category_l3", "")).cast(pl.Int32) * 3 +
        (pl.col("brand") == source.get("brand", "")).cast(pl.Int32) * 2
    )

    conditions = [
        pl.col("item_id") != target_item_id,
        pl.col("sale_status") == 1
    ]

    if price_tolerance > 0.0:
        base_price = float(source.get("price", 0))
        min_price = base_price * (1 - price_tolerance)
        max_price = base_price * (1 + price_tolerance)
        conditions.append(pl.col("price").is_between(min_price, max_price))

    similar_items = (
        items_df.filter(pl.all_horizontal(conditions))
        .with_columns(similarity_score=score_expr)
        .filter(pl.col("similarity_score") > 0)
        .sort(by=["similarity_score", "price"], descending=[True, False])
        .unique(subset=["category"], keep="first", maintain_order=True)
        .limit(top_n)
    )
    return similar_items

def get_frequently_bought_together(target_item_id, transactions_df, items_df, top_k=5):
    # Xử lý ID đơn hàng giả định từ customer_id và ngày mua
    df_orders = transactions_df.with_columns(
        pl.col("updated_date").dt.date().alias("order_date")
    ).with_columns(
        pl.concat_str([
            pl.col("customer_id").cast(pl.String),
            pl.lit("_"),
            pl.col("order_date").cast(pl.String)
        ]).alias("order_id")
    )

    total_orders = df_orders.select("order_id").n_unique()
    orders_with_A = df_orders.filter(pl.col("item_id") == target_item_id).select("order_id").unique()
    freq_A = orders_with_A.height

    if freq_A == 0:
        return pl.DataFrame()

    # Tìm các item xuất hiện chung trong cùng order_id
    co_purchases = (
        df_orders.join(orders_with_A, on="order_id", how="inner")
        .filter(pl.col("item_id") != target_item_id)
        .group_by("item_id").len().rename({"len": "co_purchase_count"})
    )

    if co_purchases.is_empty():
        return pl.DataFrame()

    freq_B_df = df_orders.filter(
        pl.col("item_id").is_in(co_purchases["item_id"])
    ).group_by("item_id").agg(pl.col("order_id").n_unique().alias("freq_B"))

    # Tính Lift Score
    fbt_stats = (
        co_purchases.join(freq_B_df, on="item_id", how="inner")
        .with_columns(
            lift_score=(pl.col("co_purchase_count") * total_orders) / (freq_A * pl.col("freq_B"))
        )
        .filter(pl.col("lift_score") > 1.0)
        .sort(["lift_score", "co_purchase_count"], descending=[True, True])
    )

    result_df = (
        fbt_stats.join(items_df, on="item_id", how="inner")
        .filter(pl.col("sale_status") == 1)
        .unique(subset=["category"], keep="first", maintain_order=True)
        .limit(top_k)
    )
    return result_df

# --- 2. GIAO DIỆN VÀ TẢI DỮ LIỆU ---

# Tiêu đề chính
st.markdown('<div class="main-title">🛒 Product Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Hệ thống gợi ý thông minh dựa trên <b>đặc tính sản phẩm</b> và <b>hành vi mua sắm</b></div>', unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_items = os.path.join(base_dir, 'items.parquet')
    path_trans = os.path.join(base_dir, 'transactions-2025-12.parquet')
    
    if not os.path.exists(path_items) or not os.path.exists(path_trans):
        raise FileNotFoundError("Không tìm thấy các file dữ liệu .parquet trong thư mục.")
        
    df_items = pl.read_parquet(path_items)
    df_trans = pl.read_parquet(path_trans)
    return df_items, df_trans

try:
    items, transactions = load_data()

    # --- SIDEBAR ---
    st.sidebar.markdown("## ⚙️ Cài đặt gợi ý")
    st.sidebar.markdown("---")

    item_list = items.filter(pl.col("sale_status") == 1)["item_id"].to_list()
    selected_id = st.sidebar.selectbox("🔎 Mã sản phẩm (item_id)", item_list, index=0)

    st.sidebar.markdown("---")
    n_results = st.sidebar.slider("📊 Số lượng gợi ý", 1, 10, 5)
    tolerance = st.sidebar.slider("💰 Dung sai giá (%)", 0, 100, 20) / 100

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="color:#4b5563;font-size:0.75rem;text-align:center;">'
        'Powered by Polars · Association Rules</div>',
        unsafe_allow_html=True
    )

    # --- HIỂN THỊ SẢN PHẨM ĐANG CHỌN ---
    source_data = items.filter(pl.col("item_id") == selected_id).row(0, named=True)

    st.markdown('<div class="section-tag">📦 Sản phẩm đang xem</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Tên sản phẩm", source_data['category'])
    col2.metric("Thương hiệu", source_data['brand'])
    col3.metric("Giá bán", f"{source_data['price']:,.0f} ₫")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- TABS GỢI Ý ---
    tab1, tab2 = st.tabs(["🔍  Sản phẩm tương tự", "🤝  Thường được mua cùng"])

    with tab1:
        st.markdown(
            '<div class="tab-desc">💡 Gợi ý dựa trên <b>danh mục</b>, <b>thương hiệu</b> '
            'và <b>tầm giá</b> tương đồng với sản phẩm đang xem.</div>',
            unsafe_allow_html=True
        )
        similar_res = get_similar_products(selected_id, items, top_n=n_results, price_tolerance=tolerance)
        if not similar_res.is_empty():
            st.dataframe(
                similar_res.select(["item_id", "category", "brand", "price", "similarity_score"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "item_id": st.column_config.TextColumn("Mã SP"),
                    "category": st.column_config.TextColumn("Danh mục"),
                    "brand": st.column_config.TextColumn("Thương hiệu"),
                    "price": st.column_config.NumberColumn("Giá (₫)", format="%,.0f"),
                    "similarity_score": st.column_config.ProgressColumn(
                        "Độ tương đồng", min_value=0, max_value=8, format="%d"
                    ),
                }
            )
        else:
            st.warning("⚠️ Không tìm thấy sản phẩm tương tự phù hợp.")

    with tab2:
        st.markdown(
            '<div class="tab-desc">🛒 Gợi ý dựa trên <b>lịch sử giỏ hàng</b> thực tế, '
            'sử dụng chỉ số <b>Lift Score</b> để đo mức độ liên kết.</div>',
            unsafe_allow_html=True
        )
        fbt_res = get_frequently_bought_together(selected_id, transactions, items, top_k=n_results)
        if not fbt_res.is_empty():
            st.dataframe(
                fbt_res.select(["item_id", "category", "price", "lift_score", "co_purchase_count"]),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "item_id": st.column_config.TextColumn("Mã SP"),
                    "category": st.column_config.TextColumn("Danh mục"),
                    "price": st.column_config.NumberColumn("Giá (₫)", format="%,.0f"),
                    "lift_score": st.column_config.NumberColumn("Lift Score", format="%.2f"),
                    "co_purchase_count": st.column_config.NumberColumn("Lần mua kèm", format="%d"),
                }
            )
        else:
            st.info("ℹ️ Sản phẩm này chưa có dữ liệu mua kèm.")

except Exception as e:
    st.error(f"❌ Lỗi hệ thống: {e}")