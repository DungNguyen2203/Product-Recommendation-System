import streamlit as st
import polars as pl
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Gợi Ý Sản Phẩm", layout="wide")

# --- 1. CÁC HÀM THUẬT TOÁN (Giữ nguyên các hàm của bạn) ---
def get_similar_products(target_item_id, items_df, top_n=5, price_tolerance=0.2):
    source_item = items_df.filter(pl.col("item_id") == target_item_id)
    if source_item.is_empty():
        return pl.DataFrame()
    
    source = source_item.row(0, named=True)
    
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

st.title("🛒 Product Recommendation System")
st.markdown("Hệ thống gợi ý sản phẩm dựa trên **Đặc tính** và **Hành vi mua sắm**.")

@st.cache_data
def load_data():
    # --- ĐOẠN SỬA LỖI ĐƯỜNG DẪN Ở ĐÂY ---
    # Lấy đường dẫn thư mục hiện tại của file app.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    path_items = os.path.join(base_dir, 'items.parquet')
    path_trans = os.path.join(base_dir, 'transactions-2025-12.parquet')
    
    # Kiểm tra xem file có thực sự tồn tại không trước khi đọc
    if not os.path.exists(path_items) or not os.path.exists(path_trans):
        raise FileNotFoundError(f"Không tìm thấy file tại: {base_dir}. Hãy đảm bảo 2 file .parquet nằm cùng thư mục với app.py")

    df_items = pl.read_parquet(path_items)
    df_trans = pl.read_parquet(path_trans)
    return df_items, df_trans

try:
    items, transactions = load_data()
    
    # Sidebar: Lựa chọn sản phẩm
    st.sidebar.header("Cài đặt gợi ý")
    item_list = items.filter(pl.col("sale_status") == 1)["item_id"].to_list()
    selected_id = st.sidebar.selectbox("Chọn mã sản phẩm (item_id):", item_list, index=0)
    
    n_results = st.sidebar.slider("Số lượng gợi ý:", 1, 10, 5)
    tolerance = st.sidebar.slider("Dung sai giá (%):", 0, 100, 20) / 100

    # Hiển thị thông tin sản phẩm gốc
    source_data = items.filter(pl.col("item_id") == selected_id).row(0, named=True)
    
    st.subheader("📦 Sản phẩm đang xem")
    col1, col2, col3 = st.columns(3)
    col1.metric("Tên sản phẩm", source_data['category'])
    col2.metric("Thương hiệu", source_data['brand'])
    col3.metric("Giá", f"{source_data['price']:,.0f} VNĐ")

    st.divider()

    # Tab phân loại gợi ý
    tab1, tab2 = st.tabs(["🔍 Sản phẩm tương tự", "🤝 Thường được mua cùng"])

    with tab1:
        st.write("Gợi ý dựa trên danh mục, thương hiệu và tầm giá.")
        similar_res = get_similar_products(selected_id, items, top_n=n_results, price_tolerance=tolerance)
        if not similar_res.is_empty():
            st.dataframe(similar_res.select(["item_id", "category", "brand", "price", "similarity_score"]), use_container_width=True)
        else:
            st.warning("Không tìm thấy sản phẩm tương tự phù hợp.")

    with tab2:
        st.write("Gợi ý dựa trên lịch sử giỏ hàng của khách hàng (Lift Score).")
        fbt_res = get_frequently_bought_together(selected_id, transactions, items, top_k=n_results)
        if not fbt_res.is_empty():
            st.dataframe(fbt_res.select(["item_id", "category", "price", "lift_score", "co_purchase_count"]), use_container_width=True)
        else:
            st.info("Sản phẩm này chưa có dữ liệu mua kèm hoặc chưa từng được bán.")

except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")