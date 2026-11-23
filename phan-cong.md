# Phân công nhóm: Mario PPO

Điền tên vào bảng dưới, sau đó cả tài liệu chỉ gọi theo mã TV1–TV4. Cột sps đo ở cuối tuần 1 (mục 3).

| Mã | Họ tên | Máy (CPU / số lõi / GPU) | sps đo được | Vai trò chính |
|---|---|---|---|---|
| TV1 | Quốc Anh | | | Môi trường: wrapper, mẹo huấn luyện, `env.py`, trình chiếu |
| TV2 | Công Kiên | Ryzen 7 7435HS / 8 lõi, 16 luồng / RTX 4050 Laptop 6 GB; RAM 16 GB; Windows 11 | (đo sau tích hợp; trần giả lập ~580) | **Điều phối nhóm**. Mạng và vòng huấn luyện: `model.py`, `train.py`, tích hợp, hình vẽ, ghép báo cáo |
| TV3 | Hải Đăng | | | GAE, đánh giá nhiều seed, chọn và huấn luyện màn thưởng, phân tích thất bại |
| TV4 | Trường | | | Mất mát PPO và vòng cập nhật, đánh giá cuối, đóng gói bài nộp |

Đổi vai thì chỉ cần đổi tên trong bảng này. Gợi ý khi đổi:
- TV3 và TV4 hợp với người vững toán và lý thuyết RL (GAE, hàm mất mát cắt).
- TV1 hợp với người thích làm việc với môi trường và trình bày (giữ phần trình chiếu).
- TV2 nên là người nắm kế hoạch chung.

## Điều gì quyết định điểm cao hay thấp (đọc trước)

| Hạng mục (đề Mục 5) | Điểm | Cách tính | Chủ chính |
|---|---|---|---|
| A. Cài đặt | 30 | Bài kiểm tra công khai: wrappers 6, model 6, GAE 9, PPO 9. Tự kiểm tra trước được | TV1, TV2, TV3, TV4 |
| B. Màn 1-1 | 25 | 25 · min(1, f / 0.8), f là `flag_rate` khi chạy đúng lệnh Mục 4 | E0 (TV1), chọn checkpoint (TV3) |
| C. Màn thưởng | 15 | 15 · min(1, f / 0.8) | TV3, TV4 |
| D. Báo cáo | 25 | Đủ 6 mục của đề Mục 6 | TV2 ghép, mỗi người viết phần mình |
| E. Trình bày | 5 | 10 phút trình bày, 5 phút hỏi đáp | TV1, cả nhóm |
| Thưởng | +10 | 10 · min(1, f / 0.8) trên một màn ngoài danh sách | TV2 |

**Thứ tự ưu tiên khi thiếu thời gian**: A → B đạt 0.8 → D đủ mục → C → E → thưởng.

**Sáu điều rút ra từ cách chấm:**

1. **Điểm màn tính liên tục, không phải đạt hay trượt.**
   - f = 0.6 vẫn được 75% điểm của hạng mục, nên màn thưởng và màn ngoài danh sách đi được một phần vẫn có điểm. Không bỏ dở.
   - Mỗi 0.1 `flag_rate` đáng: B 3.1 điểm, C 1.9 điểm, thưởng 1.25 điểm.
   - Vượt 0.8 không được thêm điểm. Phần vượt chỉ là biên an toàn.
2. **Biết trước được điểm B và C.**
   - `evaluate.py` cố định seed cho NOOP, cho môi trường và cho việc lấy mẫu hành động. Cùng checkpoint, cùng máy thì cho cùng kết quả.
   - Cách chọn checkpoint nộp: chạy **đúng lệnh Mục 4** (seed 2026) trên mọi checkpoint ứng viên, đồng thời chạy `eval_seeds.py` trên các seed khác. Chọn checkpoint đạt từ 0.8 ở seed 2026 **và** cao ở các seed khác.
   - Máy giảng viên có thể khác phiên bản torch hoặc CPU, nên vài lượt có thể ra khác. Với xác suất chạm cờ thật p, khả năng đạt từ 16/20 lượt trở lên:

     | p | Khả năng đạt ≥ 16/20 |
     |---|---|
     | 0.8 | 63% |
     | 0.85 | 83% |
     | 0.9 | 96% |
     | 0.95 | 99.7% |

     Vì vậy cần p thật khoảng 0.9 trở lên, ước lượng bằng nhiều seed.
   - Chạy lệnh Mục 4 trên hai máy khác nhau trong nhóm. Nếu hai máy ra khác nhau thì càng cần biên an toàn.
3. **Checkpoint phải nạp được bằng mã nộp cuối cùng.**
   - Giảng viên dựng lại mạng từ `config["model"]` của checkpoint và chạy các wrapper của **mã nộp**.
   - Nếu sau khi huấn luyện có người đổi `GrayResize`, `FrameSkip` hay cấu trúc `ActorCritic`, checkpoint sẽ không nạp được hoặc nhìn thấy ảnh khác lúc huấn luyện. Khi đó B và C có thể về 0.
   - Vì vậy phải đóng băng phần này từ mốc `e0-start` (quy tắc 12).
4. **Vi phạm quy định 3.5 có thể mất trắng điểm màn**: sửa `evaluate.py` hoặc phần `eval_mode`, đưa thông tin khác ngoài khung hình vào mạng, dùng checkpoint từ bên ngoài, hoặc checkpoint không tái tạo được. Xem các quy tắc 8, 10 và 11.
5. **Ngân sách màn thưởng tối đa 10 triệu bước.**
   - Khi học chuyển giao, dùng `init_from=<checkpoint 1-1>`, **không dùng `resume=`**. Với `init_from`, số bước đếm lại từ 0 và lịch lr, lịch entropy chạy lại từ đầu. Với `resume`, số bước nối tiếp từ checkpoint 1-1, lr bắt đầu đã bị giảm, và ngân sách khó giải trình.
   - Đặt `algo.total_steps` ≤ 10 triệu ngay từ đầu, và ghi trong báo cáo số bước 1-1 đã dùng để khởi tạo.
   - Theo đề Mục 3.3, 10 triệu bước là ngân sách huấn luyện trên chính màn thưởng. Khởi tạo từ checkpoint 1-1 của nhóm được phép, chỉ cần nêu rõ trong báo cáo.
6. **Báo cáo (25 điểm) là phần chắc ăn nhất sau A**, chỉ cần đủ mục và đúng yêu cầu:
   - mục 1 có bảng `pytest` và các lỗi đã gặp;
   - mục 2 có bảng cấu hình của mỗi màn, phần cứng, số bước, thời gian, sps;
   - mục 3 có đủ 6 đường cong: lợi tức, x, cờ, entropy, approxKL, tỉ lệ cắt. `scripts/plot_curves.py` vẽ sẵn cả 6, nhưng **mỗi hình phải kèm một đoạn nhận xét**;
   - mục 4 nêu giả thuyết, kết quả, kết luận; chỉ đổi đúng một yếu tố; chạy 2 seed nếu kịp;
   - mục 5 có biểu đồ vị trí x cuối lượt;
   - tối đa 8 trang.

## 0. Quy tắc chung

### Không đè việc nhau

1. **Mỗi tệp có đúng một chủ** (bảng mục 1). Muốn sửa tệp của người khác thì nhắn chủ tệp sửa, hoặc thống nhất trước trong nhóm chat.
2. **`mario_ppo/ppo.py` có hai chủ, chia theo hàm**: TV3 chỉ sửa `compute_gae`, TV4 chỉ sửa các hàm còn lại. Không ai định dạng lại (reformat) cả tệp.
3. **Git**: mỗi người làm trên nhánh riêng `<mã>/<việc>`, ví dụ `tv1/wrappers`. Chỉ gộp vào `main` khi `pytest` của phần mình đã qua **và đã có người duyệt chéo** (quy tắc 9). Kéo `main` về nhánh mình trước khi gộp.
4. **Cấu hình**: không sửa `configs/default.yaml` (trừ TV2). Mỗi lần chạy có tệp riêng `configs/<màn>-<tên-thí-nghiệm>.yaml`; người chạy là chủ tệp đó. Lưu ý: `--config` **thay hẳn** `default.yaml` (không gộp), nên tệp riêng phải là bản sao đầy đủ của `default.yaml` rồi mới sửa. Khóa mẹo mới không cần có trong `default.yaml`: `make_env` đọc khóa qua `tricks.get(..., <giá trị tắt>)`, còn lần chạy nào cần thì đặt khóa trong tệp cấu hình riêng hoặc trên dòng lệnh.
5. **Lần chạy**: mỗi lần chạy có một chủ, chạy trên máy của người đó, `run_dir=runs/<màn>/<mã>-<tên-thí-nghiệm>-s<seed>`. Không ai chạy trùng thí nghiệm của người khác (xem các bảng ở mục 3).
6. **Thư mục `runs/` không đưa lên git** vì checkpoint và video nặng. Chỉ những lần chạy dùng để nộp mới được sao vào `nop-bai/` ở tuần 4. Chia sẻ checkpoint giữa các máy qua Drive.
7. **Ghi kết quả**: mỗi người ghi vào tệp riêng `ghi-chu/<mã>.md` (cấu hình, **mã commit**, số bước, thời gian, sps, flag_rate, nhận xét). Không ai sửa tệp ghi chú của người khác.
8. **Không sửa `mario_ppo/evaluate.py`, cũng không sửa phần `eval_mode` của `env.py`** (quy định 3.5). Mẹo huấn luyện chỉ được thêm vào khối `if not eval_mode:`.

### Duyệt chéo (để ai cũng hiểu cả hệ thống khi bị hỏi lúc demo)

9. Mỗi lần gộp vào `main` phải có một người khác đọc và đồng ý. Cặp duyệt cố định:

| Người viết | Người duyệt |
|---|---|
| TV1 (wrappers, env) | TV2 |
| TV2 (model, train) | TV1 |
| TV3 (GAE) | TV4 |
| TV4 (mất mát PPO) | TV3 |

Người duyệt phải giải thích lại được phần mình đã duyệt. Đây cũng là cách chuẩn bị cho 5 phút hỏi đáp.

### Tái tạo được (quy định 3.5.4)

10. **Mẹo mới chỉ được thêm bằng khóa mới, mặc định tắt.** Không đổi hành vi của khóa đã có, vì một lần chạy đang dùng khóa đó sẽ không còn tái tạo được.
11. Trước khi bắt đầu một lần chạy, commit mọi thay đổi và ghi mã commit (`git rev-parse --short HEAD`) vào `ghi-chu/<mã>.md`. Không chạy từ mã chưa commit.
12. Khi E0 bắt đầu, TV2 gắn nhãn `git tag e0-start`.
    - Từ mốc này, **đóng băng hành vi** của `FrameSkip`, `GrayResize`, `ScalarizeReward`, chuỗi wrapper trong `env.py` và cấu trúc `ActorCritic` với mỗi giá trị `arch` đã dùng.
    - Muốn thay đổi thì chỉ được thêm khóa mới, mặc định tắt (quy tắc 10).
    - Sau mỗi lần gộp vào `main`, TV4 chạy lại `evaluate.py` trên một checkpoint E0 để chắc checkpoint vẫn nạp được và ra cùng kết quả.

### Khi một lần chạy gặp sự cố

13. **Chủ lần chạy tự quyết** dừng, đổi lr hay clip, hoặc chạy lại; sau đó báo nhóm trong ngày và ghi vào `ghi-chu/`. Dấu hiệu hỏng: `approx_kl` thường xuyên trên 0.05, entropy sụp gần 0 quá sớm, hoặc x_pos không tăng sau 1 triệu bước.
14. Không tự ý chiếm máy của người khác. Cần thêm máy thì đề nghị trong buổi họp.

## 1. Chủ sở hữu tệp

| Tệp / thư mục | Chủ | Ghi chú |
|---|---|---|
| `mario_ppo/wrappers.py` | TV1 | Ba hàm TODO cùng các wrapper mẹo ở cuối tệp |
| `mario_ppo/env.py` (chỉ khối `if not eval_mode:`) | TV1 | Bật mẹo theo khóa trong `env:` |
| `mario_ppo/model.py` | TV2 | Nature-CNN trước, IMPALA-CNN sau, chọn bằng `model.arch` |
| `mario_ppo/train.py` | TV2 | Lịch giảm dần entropy, `init_from` |
| `configs/default.yaml`, `pyproject.toml` | TV2 | |
| `scripts/plot_curves.py` | TV2 | Vẽ đường cong học từ `progress.csv` |
| `tests/test_<phần>_*.py` (kiểm thử thêm của nhóm) | Chủ của phần đó | Không sửa 5 tệp kiểm thử công khai. Ví dụ: `test_model_arch.py`, `test_train_helpers.py` của TV2 |
| `bao-cao/` (bản ghép cuối) | TV2 | Mỗi người viết phần của mình trong `bao-cao/<mã>.md`, TV2 ghép lại |
| `mario_ppo/ppo.py`: hàm `compute_gae` | TV3 | |
| `scripts/eval_seeds.py`, `scripts/plot_xpos.py` | TV3 | Đánh giá nhiều seed, biểu đồ vị trí x cuối lượt |
| `mario_ppo/ppo.py`: các hàm còn lại | TV4 | `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`, `ppo_loss`, `ppo_update` |
| `nop-bai/` | TV4 | Đánh giá cuối, đóng gói zip, kiểm tra tái tạo |
| `trinh-chieu/` | TV1 | |

## 2. Hợp đồng giao diện (thống nhất trước để làm song song)

| Thỏa thuận | Người cung cấp | Người dùng |
|---|---|---|
| Khóa cấu hình mẹo, tất cả mặc định tắt: `env.flag_bonus` (số thực, đơn vị phần thưởng sau trọng số), `env.max_no_progress_steps` (số nguyên), `env.no_progress_as_terminal` (mặc định `false`: lượt bị cắt trả `truncated=True`; `true` chỉ dùng cho thí nghiệm E3), `env.sticky_prob` (0–1) | TV1 | Người chạy thí nghiệm |
| Thưởng cờ cộng vào thành phần `x_pos` của vector (chia cho `weights[0]`), vì wrapper mẹo nằm trước `ScalarizeReward` | TV1 | |
| `ppo_update(model, opt, batch, cfg)` đọc `cfg.ent_coef` mỗi lần gọi, để `train.py` có thể giảm dần hệ số này | TV4 | TV2 |
| Khóa `algo.ent_coef_end` (mặc định `null` = giữ cố định), giảm tuyến tính từ `ent_coef` theo số bước | TV2 | Người chạy thí nghiệm |
| Khóa `model.arch: nature \| impala` (mặc định `nature`); giữ đúng giao diện `forward`/`act` | TV2 | |
| Khóa `init_from` (checkpoint `.pt`): chỉ nạp trọng số; optimizer mới; số bước, lịch lr và lịch entropy bắt đầu từ 0. Dùng cho mọi lần học chuyển giao (P2, P3, B1, B2, màn ngoài danh sách) | TV2 | TV3, TV4 |
| `ppo_update` trả dict có `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clipfrac` | TV4 | TV2 (nhật ký, hình vẽ) |
| `scripts/eval_seeds.py --ckpt ... --level ... --seeds 1 7 42` gọi `evaluate_checkpoint`, in flag_rate từng seed và giá trị nhỏ nhất | TV3 | Tất cả |

## 3. Kế hoạch theo tuần

### Tuần 1: cài đặt (mốc: `pytest` qua hết, nộp mã)

| Người | Việc | Bài kiểm tra / kết quả |
|---|---|---|
| Cả nhóm | Cài môi trường trên máy mình (README); ghi cấu hình máy vào bảng đầu tệp. **Máy Windows**: không cần WSL, không cần Visual Studio. Dùng `tools/wheels/nes_py-8.2.1-py3-none-win_amd64.whl` (cách dựng: `tools/build_nes_py_windows.py`). Lệnh cài, chạy trong thư mục dự án: `python -m venv .venv`, rồi `.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`, rồi `.venv\Scripts\python -m pip install tools/wheels/nes_py-8.2.1-py3-none-win_amd64.whl`, rồi `.venv\Scripts\python -m pip install -e .`. Muốn gỡ thì chỉ cần xóa thư mục `.venv`. `train.py` tự dùng `spawn` trên Windows (nhánh `tv2/train`), vì `forkserver` không có trên Windows | `pytest -q` chạy được (lúc đầu lỗi NotImplementedError là bình thường) |
| TV1 | `FrameSkip`, `GrayResize`, `ScalarizeReward` | `tests/test_wrappers.py` (6 điểm) |
| TV1 | **Bắt buộc xong trước cuối tuần**: `FlagBonus`, `NoProgressTruncation`, `StickyAction` cùng các khóa ở mục 2 | Kiểm tra nhỏ: bật hoặc tắt khóa thì hành vi đúng như mô tả |
| TV2 | `ActorCritic.__init__`, `forward`: khởi tạo trực giao (gain √2 cho tầng ẩn, 0.01 cho đầu policy, 1 cho đầu value), chia ảnh cho 255 | `tests/test_model.py` (6 điểm). **Xong**, nhánh `tv2/model`, chờ TV1 duyệt |
| TV2 | **Bắt buộc xong trước cuối tuần**: lịch giảm entropy (`algo.ent_coef_end`) và `init_from` trong `train.py` | **Xong**, nhánh `tv2/train`, chờ TV1 duyệt |
| TV2 | `scripts/plot_curves.py` (làm được đến đầu tuần 2) | **Xong**, nhánh `tv2/train` |
| TV3 | `compute_gae`: kết thúc thì không bootstrap; bị cắt thì bootstrap từ `final_values`; cắt truy hồi ở ranh giới lượt | `tests/test_gae.py` (9 điểm) |
| TV3 | `scripts/eval_seeds.py` | |
| TV4 | Các hàm mất mát, `approx_kl = ((ratio-1) - logratio).mean()`, chuẩn hóa lợi thế theo minibatch, cắt gradient 0.5, dừng sớm theo `target_kl` | `tests/test_ppo.py` (9 điểm) |
| TV2 (tích hợp) | Gộp các nhánh, chạy `tests/test_pipeline.py`, **nộp mã mốc tuần 1** | Cả bộ `pytest` qua |
| Cả nhóm | Chạy thử 1-1 ngắn (khoảng 200k bước) trên máy mình, ghi sps vào bảng đầu tệp | |

Thứ tự gợi ý: ngày 1–3 mỗi người làm phần mình, ngày 4 tích hợp và duyệt chéo, ngày 5 chạy thử, đo sps rồi họp chốt cấu hình E0.
Gỡ lỗi PPO: ở epoch đầu, minibatch đầu thì `ratio` phải bằng 1; `approx_kl` thường dưới 0.02.

**Họp cuối tuần 1**:
- Chốt cấu hình E0.
- Xem máy nào có sps cao nhất. **E0 chạy trên máy đó.** Nếu máy mạnh nhất không phải của TV1, người có máy đó đổi toàn bộ việc tuần 2 với TV1 (đổi cả dòng trong bảng dưới).
- Nếu mẹo hoặc lịch entropy chưa xong, **vẫn khởi động E0 không có phần đó**, để không mất ngày. Thêm vào sau bằng cách chạy tiếp từ checkpoint và ghi rõ trong `ghi-chu/`.

### Tuần 2: huấn luyện 1-1, thăm dò màn thưởng

Cấu hình gốc **E0** (cấu hình đề xuất trong bản tổng hợp nghiên cứu): γ = 0.9, λ = 0.95, lr 2.5e-4 giảm dần, clip 0.2, `ent_coef` 0.01 giảm về khoảng 0.0005, trọng số `[1,1,2,0,0]`, `noop_max` 30, có thưởng cờ và cắt lượt khi không tiến (tính là cắt ngắn).
Các thí nghiệm so sánh chỉ đổi **đúng một** lựa chọn so với E0, chạy **2 triệu bước** (đủ để thấy xu hướng; nhánh đối chứng dùng 2 triệu bước đầu của E0).

| Máy | Lần chạy, theo thứ tự | Mục đích |
|---|---|---|
| TV1 (hoặc máy mạnh nhất) | **E0** 1-1, seed 1, 3–5 triệu bước | Checkpoint 1-1 để nộp, đối chứng cho mọi thí nghiệm, điểm xuất phát học chuyển giao |
| TV2 | **E1** 1-1, γ = 0.99, 2 triệu bước | Thí nghiệm so sánh 1 |
| TV3 | 1) **P1** thăm dò 4-1 từ đầu, 1 triệu bước. 2) **P2** thăm dò 1-2, học chuyển giao (`init_from=`) từ checkpoint E0 khoảng 2 triệu bước, 1 triệu bước | Chọn màn thưởng |
| TV4 | 1) **E2** 1-1, `ent_coef` cố định 0.01, 2 triệu bước (không cần lịch entropy nên chạy được ngay). 2) **P3** thăm dò 3-3, học chuyển giao (`init_from=`) từ checkpoint E0, 1 triệu bước | Thí nghiệm so sánh 2 và chọn màn thưởng |

**Họp cuối tuần 2**:
- TV3 chạy `eval_seeds.py` trên các checkpoint E0, chốt checkpoint 1-1 (cần `flag_rate` từ 0.8 trở lên ở mọi seed). Nếu E0 chưa đạt, E0 chạy tiếp; checkpoint tốt nhất của E2 là phương án dự phòng.
- Chốt màn thưởng: màn có x_pos tăng nhanh nhất ở P1–P3, và ghi rõ là học chuyển giao hay huấn luyện từ đầu.
- Nhìn kết quả E1, E2 rồi quyết định thí nghiệm nào chênh lệch chưa rõ, cần chạy seed 2.

### Tuần 3: màn thưởng, thí nghiệm, phân tích thất bại

| Máy | Lần chạy | Mục đích |
|---|---|---|
| TV3 | **B1** màn thưởng, seed 1, tối đa 10 triệu bước | Checkpoint màn thưởng để nộp |
| TV4 | **B2** màn thưởng: seed 2 với cấu hình như B1. Nếu B1 kẹt quá 3 triệu bước thì chuyển sang biến thể lr 1e-4, clip 0.1 | Dự phòng cho B1 |
| TV1 | 1) **E3** 1-1, `no_progress_as_terminal: true`, 2 triệu bước. 2) Seed 2 của thí nghiệm được chọn ở buổi họp tuần 2 | Thí nghiệm so sánh 3 (Pardo 2018), đủ 2 seed |
| TV2 | 1) Seed 2 của E0 trong 2 triệu bước đầu (đối chứng cho seed 2). 2) Khi 1-1 đã đạt và B1 đang tiến triển: thử một màn ngoài danh sách bằng `init_from=` từ checkpoint 1-1. Điểm thưởng tính liên tục nên đi được một phần vẫn có điểm. Nếu B1 kẹt thì máy này ưu tiên hỗ trợ màn thưởng, vì mỗi 0.1 cờ ở màn thưởng đáng 1.9 điểm, ở màn ngoài danh sách chỉ đáng 1.25 | Đủ 2 seed, điểm thưởng |

TV3: viết `plot_xpos.py`, xem video các lượt chết, ghi lại chỗ tác tử hay chết và lý do (cho cả 1-1 và màn thưởng).

**Ước lượng thời gian máy**: tuần 2 khoảng 12 triệu bước, tuần 3 khoảng 26 triệu bước, chia cho 4 máy.
- Ở 300 sps, 1 triệu bước mất khoảng 55 phút, nên B1 10 triệu bước mất khoảng 9 giờ.
- Ở 100 sps thì B1 mất khoảng 28 giờ, phải khởi động B1 ngay đầu tuần 3.
- Sau khi đo sps ở tuần 1, cập nhật lại phần ước lượng này.

### Tuần 4: báo cáo, trình chiếu, nộp bài

| Phần báo cáo (tối đa 8 trang) | Người viết |
|---|---|
| 1. Cài đặt | Mỗi người viết phần mình đã cài và phần mình đã duyệt (khoảng ½ trang). TV4 làm bảng kết quả `pytest` |
| 2. Thiết lập huấn luyện (bảng cấu hình, phần cứng, số bước, thời gian, sps) | TV2 (lấy số liệu từ `ghi-chu/` của mọi người) |
| 3. Đường cong học | TV2 vẽ hình; chủ mỗi lần chạy viết nhận xét cho hình của mình |
| 4. Thí nghiệm so sánh | Chủ thí nghiệm: TV2 (E1), TV4 (E2), TV1 (E3) |
| 5. Phân tích thất bại | TV3 |
| 6. Phân công | TV1 |

| Việc khác | Người làm |
|---|---|
| Ghép báo cáo thành PDF | TV2 |
| Trình chiếu 10 phút và kịch bản demo | TV1 |
| Chạy `evaluate.py` theo đúng lệnh Mục 4 của đề trên các checkpoint nộp, xuất JSON và video | TV4 |
| Đóng gói `nhom-<số>.zip` (mã, `config.yaml`, `progress.csv`, `evals.csv`, checkpoint, JSON, video, PDF, slide), kèm cả lần chạy 1-1 dùng làm `init_from` cho màn thưởng. Kiểm tra mỗi checkpoint khớp mã commit và cấu hình đã ghi | TV4 |
| **Kiểm tra cuối**: giải nén zip vào thư mục mới, cài môi trường sạch, chạy đúng lệnh Mục 4 trên từng checkpoint nộp; `flag_rate` phải khớp số trong báo cáo | TV4, TV3 làm lại trên máy khác |
| Đọc soát toàn bộ báo cáo lần cuối | TV3 |
| Tập trình bày: mỗi người tự giải thích được GAE, hàm mất mát cắt và các mẹo huấn luyện | Cả nhóm |

## 4. Họp nhóm

- Công Kiên (TV2) điều phối: hẹn lịch, giữ thời gian, ghi lại các quyết định chốt.
- Hai lần mỗi tuần, mỗi lần 15 phút. Mỗi người báo ba ý: đã xong gì, đang kẹt gì, máy đang chạy gì.
- Các buổi họp chốt bắt buộc: cuối tuần 1 (cấu hình E0, máy chạy E0), cuối tuần 2 (checkpoint 1-1, màn thưởng, thí nghiệm cần seed 2), giữa tuần 3 (B1 có tiến triển không, có cần đổi sang biến thể không).
