# Mẫu ghi chú (chép sang `ghi-chu/<mã>.md` của mình)

Mỗi người chỉ sửa tệp của mình (quy tắc 7). TV2 lấy số liệu ở đây để làm bảng mục 2 và hình mục 3 của báo cáo,
nên giữ đúng các dòng bên dưới.

## Cài đặt (cho mục 1 của báo cáo, ghi ngay trong tuần 1)

- Phần đã cài:
- Lựa chọn quan trọng và lý do:
- Lỗi đã gặp và cách sửa:
- Phần đã duyệt chéo cho ai, điều mình hiểu ra khi duyệt:

## Lần chạy: <tên, ví dụ E0-s1>

- Mục đích: (đối chứng / thí nghiệm so sánh: đổi đúng yếu tố gì so với E0 / thăm dò màn)
- Lệnh: `python -m mario_ppo.train --config ... run_dir=runs/<màn>/<mã>-<tên>-s<seed> ...`
- Mã commit (`git rev-parse --short HEAD`, quy tắc 11):
- Khởi tạo: từ đầu / `init_from=<checkpoint>` (ghi số bước của checkpoint đó)
- Máy, `num_envs`, `torch_threads`:
- Bắt đầu, kết thúc, tổng thời gian:
- Số bước; sps trung bình (cột `sps` cuối `progress.csv`):
- Kết quả:
  - `flag_rate` checkpoint cuối (`evals.csv`):
  - `python -m mario_ppo.evaluate --ckpt ... --level ... --episodes 20 --seed 2026`:
  - `eval_seeds.py` (min các seed):
- Sự cố và cách xử lý (quy tắc 13):
- Nhận xét cho báo cáo (đường cong có gì đáng nói):
