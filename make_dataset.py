"""
Sinh bộ dữ liệu mô phỏng kết quả học tập của sinh viên.

Các thuộc tính (khớp với Bảng 3.1 trong báo cáo):
    - study_hours            : Số giờ tự học mỗi tuần (float)
    - assignments_completed  : Số bài tập đã hoàn thành (int)
    - attendance_rate        : Tỷ lệ chuyên cần, % (float)
    - midterm_score          : Điểm thi giữa kỳ, thang 100 (float)
    - final_score            : Điểm cuối kỳ, thang 100 (float) -- BIẾN MỤC TIÊU

Mối quan hệ được thiết kế gần với thực tế:
    midterm_score và attendance_rate ảnh hưởng mạnh nhất tới final_score,
    tiếp theo là study_hours, rồi assignments_completed; kèm nhiễu ngẫu nhiên.
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "data" / "student_performance.csv"

N_SAMPLES = 1500
RANDOM_SEED = 42


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # --- Sinh các biến đầu vào ---
    study_hours = np.clip(rng.normal(15, 6, N_SAMPLES), 0, 40)

    # Số bài tập hoàn thành có liên hệ với thời gian tự học.
    assignments = np.clip(
        np.round(study_hours * 0.45 + rng.normal(3, 2.5, N_SAMPLES)), 0, 20
    ).astype(int)

    attendance_rate = np.clip(rng.normal(85, 9, N_SAMPLES), 50, 100)

    # Điểm giữa kỳ chịu ảnh hưởng nhẹ từ chuyên cần và giờ học.
    midterm_score = np.clip(
        50
        + 0.18 * (attendance_rate - 85)
        + 0.6 * (study_hours - 15)
        + rng.normal(0, 9, N_SAMPLES),
        0,
        100,
    )

    # --- Tính điểm cuối kỳ từ tổ hợp có trọng số (chuẩn hóa z-score) ---
    z = (
        0.45 * (midterm_score - midterm_score.mean()) / midterm_score.std()
        + 0.30 * (attendance_rate - attendance_rate.mean()) / attendance_rate.std()
        + 0.22 * (study_hours - study_hours.mean()) / study_hours.std()
        + 0.16 * (assignments - assignments.mean()) / assignments.std()
    )
    final_score = np.clip(68 + 13 * z + rng.normal(0, 4, N_SAMPLES), 0, 100)

    df = pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 1),
            "assignments_completed": assignments,
            "attendance_rate": np.round(attendance_rate, 1),
            "midterm_score": np.round(midterm_score, 1),
            "final_score": np.round(final_score, 1),
        }
    )

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")

    print(f"Đã tạo dataset: {OUT_PATH}  ({len(df)} dòng)")
    print("\n5 dòng đầu:")
    print(df.head().to_string(index=False))
    print("\nTương quan với final_score:")
    print(df.corr(numeric_only=True)["final_score"].round(3).to_string())


if __name__ == "__main__":
    main()
