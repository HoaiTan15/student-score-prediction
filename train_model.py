import os
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

TARGET_CANDIDATES = [
    "final_score",
    "Performance Index",
    "Exam Score",
    "Final Score",
    "Final Grade",
    "Score",
    "Marks",
    "G3",
]

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def find_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Không tìm thấy cột mục tiêu dạng số. Hãy kiểm tra file CSV.")
    return numeric_cols[-1]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": name,
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def save_feature_info(X: pd.DataFrame, target_col: str):
    info = {
        "target_column": target_col,
        "feature_columns": X.columns.tolist(),
        "numeric_features": X.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": X.select_dtypes(exclude=[np.number]).columns.tolist(),
    }
    with open(MODEL_DIR / "feature_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


def plot_correlation(df: pd.DataFrame, target_col: str):
    numeric_df = df.select_dtypes(include=[np.number])
    if target_col not in numeric_df.columns or numeric_df.shape[1] < 2:
        return

    corr = numeric_df.corr(numeric_only=True)
    target_corr = corr[target_col].sort_values(ascending=False)

    # Biểu đồ cột: mức tương quan của từng thuộc tính số với biến mục tiêu.
    plt.figure(figsize=(9, 5))
    target_corr.drop(target_col, errors="ignore").plot(kind="bar")
    plt.title(f"Tương quan giữa các thuộc tính số và {target_col}")
    plt.xlabel("Thuộc tính")
    plt.ylabel("Hệ số tương quan")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_chart.png", dpi=160)
    plt.close()

    # Ma trận tương quan Pearson (heatmap) bằng Seaborn.
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Ma trận tương quan Pearson giữa các thuộc tính")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dataset: {DATA_PATH}\n"
            "Hãy đặt file CSV vào thư mục data/ và đổi tên thành student_performance.csv"
        )

    df = pd.read_csv(DATA_PATH)
    df = normalize_columns(df)

    # Kiểm tra dữ liệu thiếu và dữ liệu trùng lặp
    print("Số giá trị thiếu:", df.isnull().sum().sum())
    print("Số bản ghi trùng lặp:", df.duplicated().sum())

    # Xóa dòng trùng và dòng rỗng hoàn toàn
    df = df.drop_duplicates()
    df = df.dropna(how="all")

    target_col = find_target_column(df)
    if target_col not in df.columns:
        raise ValueError("Không tìm thấy cột mục tiêu.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    if not np.issubdtype(y.dtype, np.number):
        y = pd.to_numeric(y, errors="coerce")

    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    preprocessor = build_preprocessor(X)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=6),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=None,
            min_samples_split=2
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    trained_models = {}

    for name, regressor in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", regressor)
        ])

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(name, pipeline, X_test, y_test)
        results.append(metrics)
        trained_models[name] = pipeline

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{safe_name}.pkl")

    results_df = pd.DataFrame(results).sort_values(by="R2", ascending=False)
    results_df.to_csv(REPORT_DIR / "model_metrics.csv", index=False, encoding="utf-8-sig")

    best_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_name]

    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    save_feature_info(X, target_col)
    plot_correlation(pd.concat([X, y.rename(target_col)], axis=1), target_col)

    print("\n=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ===")
    print(results_df.to_string(index=False))
    print(f"\nModel tốt nhất: {best_name}")
    print(f"Đã lưu model tại: {MODEL_DIR / 'best_model.pkl'}")
    print(f"Đã lưu bảng đánh giá tại: {REPORT_DIR / 'model_metrics.csv'}")


if __name__ == "__main__":
    main()
