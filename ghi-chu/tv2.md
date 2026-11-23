# Ghi chú TV2 (Công Kiên)

## Cài đặt (cho mục 1 của báo cáo)

- Phần đã cài:
  - `model.py` (nhánh `tv2/model`): `ActorCritic`. Mặc định `nature`: Nature-CNN (3 conv, FC 512). Tùy chọn `impala`: IMPALA-CNN, 3 tầng [16, 32, 32], mỗi tầng 2 khối dư, FC 256. Chọn bằng `model.arch`.
  - `train.py` (nhánh `tv2/train`):
    - lịch giảm entropy tuyến tính `algo.ent_coef_end`;
    - `init_from` cho học chuyển giao;
    - ghi cột `ent_coef` vào `progress.csv`;
    - dùng `spawn` trên Windows.
  - `scripts/plot_curves.py`, `tests/test_model_arch.py`, `tests/test_train_helpers.py`.
- Lựa chọn quan trọng và lý do:
  - **Khởi tạo trực giao:** gain √2 cho tầng ẩn, 0.01 cho đầu policy (chính sách ban đầu gần đều, khám phá tốt), 1 cho đầu value. Ảnh chia cho 255. Theo "37 Implementation Details of PPO" và CleanRL.
  - **Conv của IMPALA giữ khởi tạo mặc định của PyTorch**, như CleanRL `ppo_procgen`: khởi tạo trực giao gain √2 cộng dồn qua 6 khối dư làm biên độ kích hoạt tăng nhiều lần.
  - **Số đặc trưng sau conv tính bằng một lần chạy thử** với ảnh 84×84, thay vì ghi cứng 3136.
  - **IMPALA chậm hơn Nature-CNN khoảng 13 lần trên CPU** (forward 512 ảnh: 664 ms so với 52 ms), nên E0 dùng `nature`.
  - **`init_from` thay vì `resume` khi học chuyển giao:** `resume` nạp cả số bước của checkpoint 1-1. Khi đó lr và entropy bắt đầu đã bị giảm, và số bước của màn thưởng bị trộn với 1-1 (đề giới hạn 10 triệu bước). `init_from` chỉ nạp trọng số; optimizer mới; số bước và các lịch bắt đầu từ 0. Nếu có cả hai thì `resume` được ưu tiên, để chạy tiếp được lần chạy đã dùng `init_from`.
  - **Lịch entropy:** `ppo_update` đọc `cfg.ent_coef` ở mỗi lần gọi, nên `train.py` chỉ cần gán lại `a.ent_coef` trước mỗi lần cập nhật. Giá trị đầu lấy từ cấu hình, nên `resume` vẫn tính lịch đúng.
- Lỗi đã gặp và cách sửa:
  - **`forkserver` không có trên Windows.** `AsyncVectorEnv(context="forkserver")` báo `ValueError` khi `num_envs > 1`, kể cả trong `test_pipeline`. Sửa: dùng `spawn` khi `sys.platform == "win32"`. Đo trên máy này: 16 trình giả lập chạy song song qua `spawn` đạt khoảng 2.300 khung hình/giây.
  - **`nes-py` 8.2.1 chỉ có mã nguồn trên PyPI**, cài trên Windows đòi Visual Studio Build Tools. Sửa: biên dịch DLL trình giả lập bằng Zig (cài qua pip) rồi đóng gói thành `tools/wheels/nes_py-8.2.1-py3-none-win_amd64.whl`. DLL được nạp qua ctypes, nên dùng được cho mọi Python 3 và chỉ phụ thuộc UCRT có sẵn trong Windows. Cách dựng: `tools/build_nes_py_windows.py`.
  - **`--config` thay hẳn `default.yaml`**, không gộp. Đã ghi vào quy tắc 4 của `phan-cong.md`.
- Duyệt chéo: duyệt phần wrappers và env của TV1 (chưa tới lượt).

## Lần chạy

(chưa có; E1 γ = 0.99 bắt đầu sau buổi họp cuối tuần 1)
