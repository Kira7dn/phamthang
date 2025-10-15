INPUT (frames + numbers)
    ↓
1. Normalize Frames: xác định row_no và col_no của từng frame, san bằng w_px và h_px của các frame cùng row_no và col_no, ví dụ: cùng col thì w_px phải bằng nhau, cùng row thì h_px phải bằng nhau
    ↓
2. Estimate Scale (RANSAC)
    ↓
3. Generate Candidates (layout-aware scoring): ước lượng est_w, est_h, xác định số lần width và số lần height có thể xuất hiện trong numbers với logic (cùng cột: số width dùng chung, do đó có thể chỉ có 1 width trong numbers, và số height có thể sẽ xuất hiện đầy đủ và ngược lại với cùng hàng), kết hợp với logic score có sẵn
    ↓
4. Global Assignment
    ├─ Greedy Init
    ├─ Local Search (hill climbing)
    ├─ Consensus (majority voting) ← BUG: không apply vào return
    └─ Post-Validation (layout-aware)
    ↓
5. Recompute Scale? (nếu >2% change → quay lại bước 3)
    ↓
6. Find Inner Heights (backtracking)
    ↓
7. Calculate Unused Penalty
    ↓
8. Build Final Output
    ↓
OUTPUT (dimensions + confidences)