# Khung báo cáo (tối đa 8 trang PDF)

Mỗi người viết phần của mình vào `bao-cao/<mã>.md` (ví dụ `bao-cao/tv3.md`), dùng đúng tiêu đề mục bên dưới.
TV2 ghép thành PDF ở tuần 4. Cột "Trang" là ngân sách, để tổng không vượt 8 trang.

| Mục (đề Mục 6) | Trang | Người viết | Bắt buộc phải có |
|---|---|---|---|
| 1. Cài đặt | 1.5 | Mỗi người: phần mình cài và phần mình duyệt. TV4: bảng `pytest` | Mô tả ngắn từng thành phần, lựa chọn, **lỗi đã gặp**; bảng kết quả `pytest` |
| 2. Thiết lập huấn luyện | 1 | TV2 | Bảng cấu hình cuối cùng của mỗi màn; phần cứng; số bước; thời gian chạy; sps |
| 3. Đường cong học | 2 | TV2 vẽ; chủ lần chạy viết nhận xét | Lợi tức, vị trí x, tỉ lệ chạm cờ, entropy, approxKL, tỉ lệ cắt theo số bước; **mỗi hình một đoạn nhận xét** |
| 4. Thí nghiệm so sánh | 1.75 | TV2 (E1), TV4 (E2), TV1 (E3) | Mỗi thí nghiệm: giả thuyết, cấu hình (chỉ đổi một yếu tố so với E0), kết quả (hình và số), kết luận; 2 seed nếu có |
| 5. Phân tích thất bại | 1.25 | TV3 | Biểu đồ vị trí x cuối lượt; tác tử hay chết ở đâu, vì sao; nhóm đã thử gì |
| 6. Phân công | 0.5 | TV1 | Đóng góp của từng người |

## Hình (lưu ở `bao-cao/hinh/`, dùng .pdf để giữ nét khi in)

| Hình | Mục | Lệnh |
|---|---|---|
| Đường cong 1-1 | 3 | `python scripts/plot_curves.py runs/1-1/<E0> --out bao-cao/hinh/1-1.pdf` |
| Đường cong màn thưởng | 3 | `python scripts/plot_curves.py runs/<màn>/<B1> --out bao-cao/hinh/<màn>.pdf` |
| Mỗi thí nghiệm so sánh | 4 | `python scripts/plot_curves.py <E0-s1> <E0-s2> <E1-s1> <E1-s2> --labels "E0 γ=0,9" "E0 γ=0,9" "E1 γ=0,99" "E1 γ=0,99" --panels flag eval_flag x_pos --cols 3 --out bao-cao/hinh/e1.pdf` (nhánh đối chứng chỉ lấy 2 triệu bước đầu của E0) |
| Vị trí x cuối lượt | 5 | `scripts/plot_xpos.py` (TV3) |

## Bảng mục 2 (TV2 điền từ `ghi-chu/`)

| Màn | Checkpoint nộp | Khác `default.yaml` ở đâu | Khởi tạo | Số bước | Thời gian | sps | Máy | `flag_rate` (seed 2026) |
|---|---|---|---|---|---|---|---|---|
| 1-1 | | | từ đầu | | | | | |
| màn thưởng | | | `init_from` / từ đầu | | | | | |

## Hạn nộp bản nháp cho TV2

- Mục 1: cuối tuần 1, khi còn nhớ rõ các lỗi đã gặp.
- Mục 4, 5: cuối tuần 3.
- Bản ghép đầy đủ: giữa tuần 4. TV3 đọc soát, TV2 sửa và xuất PDF.
