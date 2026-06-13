import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURE_INFO_PATH = BASE_DIR / "models" / "feature_info.json"
METRICS_PATH = BASE_DIR / "reports" / "model_metrics.csv"
CORR_CHART_PATH = BASE_DIR / "reports" / "correlation_chart.png"
CORR_HEATMAP_PATH = BASE_DIR / "reports" / "correlation_heatmap.png"


T = {
    "app_title": "Ứng dụng dự đoán điểm số sinh viên",
    "subtitle": "Đồ án môn Khai phá dữ liệu - Bài toán hồi quy",
    "eval_title": "Đánh giá & so sánh mô hình",
    "eval_intro": "Bảng dưới so sánh 3 mô hình hồi quy theo các độ đo MAE, MSE, RMSE và R². Mô hình có R² cao nhất được chọn để dự đoán.",
    "best_model": "Mô hình tốt nhất",
    "metrics_table": "Bảng kết quả đánh giá",
    "chart_corr": "Tương quan của các thuộc tính với điểm cuối kỳ",
    "chart_heatmap": "Ma trận tương quan Pearson",
    "no_metrics": "Chưa có kết quả đánh giá. Hãy chạy: python train_model.py trước.",

    "input_title": "Nhập thông tin sinh viên",
    "predict_button": "Dự đoán điểm",
    "note_title": "Ghi chú",
    "note_text": "Ứng dụng sử dụng mô hình hồi quy đã huấn luyện từ dữ liệu sinh viên. Kết quả dự đoán chỉ mang tính tham khảo cho mục đích học tập.",
    "result_title": "Kết quả dự đoán điểm số",
    "predicted_score": "Điểm dự đoán (thang 10)",
    "level_label": "Xếp loại",
    "input_data": "Dữ liệu đã nhập",
    "attribute": "Thuộc tính",
    "value": "Giá trị",
    "advice_title": "Gợi ý cải thiện",
    "about_title": "Mô tả đề tài",
    "goal": "Mục tiêu",
    "algorithm": "Thuật toán sử dụng",
    "evaluation": "Đánh giá mô hình",
    "level_weak": "Yếu / Cần cải thiện",
    "level_average": "Trung bình",
    "level_good": "Khá",
    "level_excellent": "Giỏi",
    "fields": {
        "study_hours": "Số giờ tự học mỗi tuần",
        "assignments_completed": "Số bài tập đã hoàn thành",
        "attendance_rate": "Tỷ lệ chuyên cần (%)",
        "midterm_score": "Điểm giữa kỳ (thang 10)",
        "final_score": "Điểm cuối kỳ",
    },
    "about_goal_text": "Xây dựng mô hình hồi quy để dự đoán điểm cuối kỳ của sinh viên dựa trên số giờ tự học, số bài tập hoàn thành, tỷ lệ chuyên cần và điểm giữa kỳ.",
    "about_algo_text": "Đề tài sử dụng 3 thuật toán hồi quy: Linear Regression, Decision Tree Regression và Random Forest Regression. Dữ liệu số được chuẩn hóa bằng Min-Max Scaling.",
    "about_eval_text": "Các mô hình được đánh giá bằng MAE, MSE, RMSE và R². Mô hình có R² cao nhất sẽ được lưu thành best_model.pkl để dùng trong web.",
    "no_model": "Chưa có file model. Hãy chạy lệnh: python train_model.py trước.",
}


def load_feature_info():
    if FEATURE_INFO_PATH.exists():
        with open(FEATURE_INFO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "target_column": "final_score",
        "feature_columns": [
            "study_hours",
            "assignments_completed",
            "attendance_rate",
            "midterm_score",
        ],
        "numeric_features": [
            "study_hours",
            "assignments_completed",
            "attendance_rate",
            "midterm_score",
        ],
        "categorical_features": [],
    }


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def score_level(score_10: float) -> str:
    if score_10 < 5:
        return T["level_weak"]
    if score_10 < 6.5:
        return T["level_average"]
    if score_10 < 8:
        return T["level_good"]
    return T["level_excellent"]


def advice_from_input(data: dict, score_10: float) -> list:
    study = float(data.get("study_hours", 0) or 0)
    assignments = float(data.get("assignments_completed", 0) or 0)
    attendance = float(data.get("attendance_rate", 0) or 0)
    midterm_10 = float(data.get("midterm_score", 0) or 0) / 10  # data lưu theo thang 100

    advice = []
    if study < 10:
        advice.append("Nên tăng số giờ tự học mỗi tuần để cải thiện kết quả.")
    if attendance < 75:
        advice.append("Nên đi học đều hơn vì tỷ lệ chuyên cần ảnh hưởng lớn đến kết quả.")
    if midterm_10 < 5:
        advice.append("Cần ôn lại kiến thức nền vì điểm giữa kỳ còn thấp.")
    if assignments < 8:
        advice.append("Nên hoàn thành thêm bài tập để củng cố kiến thức.")
    if score_10 >= 8:
        advice.append("Kết quả dự đoán tốt, nên tiếp tục duy trì thói quen học tập hiện tại.")
    if not advice:
        advice.append("Có thể cải thiện thêm bằng cách tăng số bài luyện tập và theo dõi tiến độ học tập.")
    return advice


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return None
    df = pd.read_csv(METRICS_PATH, encoding="utf-8-sig")
    return df.sort_values(by="R2", ascending=False).reset_index(drop=True)


def render_evaluation():
    """Phần đánh giá & so sánh mô hình hiển thị ở đầu trang."""
    st.subheader("" + T["eval_title"])
    st.caption(T["eval_intro"])

    metrics = load_metrics()
    if metrics is None:
        st.warning(T["no_metrics"])
        return

    best = metrics.iloc[0]
    st.success(f" {T['best_model']}: **{best['model']}**  —  R² = {best['R2']:.4f}, "
               f"MAE = {best['MAE']:.3f}, RMSE = {best['RMSE']:.3f}")

    st.markdown("**" + T["metrics_table"] + "**")
    st.dataframe(metrics, hide_index=True, use_container_width=True)

    # So sánh R² giữa các mô hình bằng biểu đồ cột.
    st.bar_chart(metrics.set_index("model")["R2"], use_container_width=True)

    # Biểu đồ phân tích tương quan (nếu đã sinh từ train_model.py).
    chart_cols = st.columns(2)
    if CORR_CHART_PATH.exists():
        chart_cols[0].image(str(CORR_CHART_PATH), caption=T["chart_corr"], use_container_width=True)
    if CORR_HEATMAP_PATH.exists():
        chart_cols[1].image(str(CORR_HEATMAP_PATH), caption=T["chart_heatmap"], use_container_width=True)


def render_input_form(feature_info):
    """Hiển thị form nhập liệu và trả về (input_data cho model, dữ liệu hiển thị)."""
    input_data = {}
    display_input_data = {}

    # Cấu hình widget số cho từng thuộc tính (min, max, mặc định, bước nhảy).
    numeric_config = {
        "study_hours": (0.0, 40.0, 15.0, 1.0),
        "assignments_completed": (0.0, 20.0, 10.0, 1.0),
        "attendance_rate": (0.0, 100.0, 85.0, 1.0),
        "midterm_score": (0.0, 10.0, 6.5, 0.5),  # nhập theo thang 10
    }

    cols = st.columns(2)
    for i, col in enumerate(feature_info["feature_columns"]):
        label = T["fields"].get(col, col)
        widget = cols[i % 2]

        min_v, max_v, default_v, step_v = numeric_config.get(col, (0.0, 100.0, 0.0, 1.0))
        value = widget.number_input(
            label, min_value=min_v, max_value=max_v, value=default_v, step=step_v, key=col
        )

        # Điểm giữa kỳ nhập theo thang 10, model dùng thang 100 nên nhân 10.
        if col == "midterm_score":
            input_data[col] = value * 10
        else:
            input_data[col] = value
        display_input_data[label] = value

    return input_data, display_input_data


def main():
    st.set_page_config(page_title="Dự đoán điểm số sinh viên", page_icon="🎓", layout="centered")

    feature_info = load_feature_info()

    st.title(" " + T["app_title"])
    st.caption(T["subtitle"])

    with st.expander("" + T["about_title"]):
        st.markdown(f"**{T['goal']}:** {T['about_goal_text']}")
        st.markdown(f"**{T['algorithm']}:** {T['about_algo_text']}")
        st.markdown(f"**{T['evaluation']}:** {T['about_eval_text']}")

    # --- Phần 1: Đánh giá mô hình (ở trên) ---
    render_evaluation()

    st.divider()

    # --- Phần 2: Dự đoán điểm (kéo xuống dưới) ---
    st.subheader(T["input_title"])

    model = load_model()
    if model is None:
        st.error(T["no_model"])
        st.stop()

    with st.form("predict_form"):
        input_data, display_input_data = render_input_form(feature_info)
        submitted = st.form_submit_button(T["predict_button"], use_container_width=True)

    if submitted:
        input_df = pd.DataFrame([input_data])

        # Model dự đoán điểm cuối kỳ theo thang 100, quy đổi sang thang 10 để hiển thị.
        predicted_score_100 = float(model.predict(input_df)[0])
        predicted_score_100 = max(0, min(100, predicted_score_100))
        predicted_score_10 = predicted_score_100 / 10

        st.subheader("" + T["result_title"])
        c1, c2 = st.columns(2)
        c1.metric(T["predicted_score"], f"{predicted_score_10:.2f}")
        c2.metric(T["level_label"], score_level(predicted_score_10))

        st.markdown("#### " + T["advice_title"])
        for item in advice_from_input(input_data, predicted_score_10):
            st.markdown(f"- {item}")

        st.markdown("#### " + T["input_data"])
        table = pd.DataFrame(
            {T["attribute"]: list(display_input_data.keys()),
             T["value"]: list(display_input_data.values())}
        )
        st.dataframe(table, hide_index=True, use_container_width=True)

    st.info(f"**{T['note_title']}:** {T['note_text']}")


if __name__ == "__main__":
    main()
