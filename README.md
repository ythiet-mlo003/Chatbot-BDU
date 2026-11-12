# Chatbot trợ lý sinh viên BDU

Đồ án tốt nghiệp: **Xây dựng Chatbot trợ lý sinh viên BDU ứng dụng mô hình
ngôn ngữ tiền huấn luyện kết hợp kỹ thuật RAG**.

Repo này chứa code và tài liệu để chạy chatbot (giao diện Streamlit). Hai
file dữ liệu lớn không đẩy lên đây được, lấy từ thư mục Drive của đồ án
(xem mục "Lấy 2 file dữ liệu lớn" bên dưới).

## Lấy code về máy

```text
git clone https://github.com/ythiet-mlo003/Chatbot-BDU.git
```

Nên clone vào đường dẫn ngắn, ví dụ `D:\Chatbot-BDU`.

## Lấy 2 file dữ liệu lớn

Tải từ [thư mục Chatbot-BDU trên Google Drive](https://drive.google.com/drive/folders/1hm-5YwbSie9U5AnMN1vphTBgEQjUIPn4),
đặt đúng chỗ như sau (tạo thư mục nếu chưa có):

- `chunks-v7.3.2-filtered.jsonl` (12.910 đoạn) → `chunks-v7.3.2-filtered/`
- `vectors_bge_m3_with_linkonly.npz` (13.040 vector) →
  `embedding-colab-v7.3.2-filtered/vector-result/`

Hai file này không có trong repo vì vượt giới hạn của đường đẩy lên GitHub
(file text 23 MB và file vector nhị phân 66 MB).

## Chạy chatbot trên Windows

1. Sao chép `.env.example` thành `.env` ở thư mục gốc và điền khóa Gemini
   ngay sau `LLM_API_KEY=`. Khóa lấy miễn phí tại Google AI Studio, không
   cần thẻ ngân hàng. Tệp `.env` không có trong repo và bị `.gitignore`
   loại khỏi Git.
2. Chạy các lệnh sau trong PowerShell tại thư mục gốc. Hai dòng đầu chỉ cần
   khi tạo môi trường và cài thư viện.

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r .\rag-pipeline-minimal\requirements-streamlit.txt
   .\.venv\Scripts\python.exe -m streamlit run .\rag-pipeline-minimal\app_streamlit.py --server.address 127.0.0.1
   ```

3. Mở [localhost:8501](http://localhost:8501). Lần hỏi đầu tiên có thể mất
   thời gian tải model BGE-M3 về máy. Máy dùng CPU để nhúng câu hỏi, còn LLM
   sinh câu trả lời qua API.

Chi tiết xem `rag-pipeline-minimal/README-buoc5.md`.

## Cấu trúc repo

- `rag-pipeline-minimal/app_streamlit.py` — giao diện Streamlit.
- `rag-pipeline-minimal/rag_streamlit_core.py` — truy xuất hybrid (BM25 +
  vector FAISS, kết hợp RRF, top 10 chunk) và gọi LLM.
- `rag-pipeline-minimal/requirements-streamlit.txt` — thư viện cần cài.
- `rag-pipeline-minimal/README-buoc5.md` — hướng dẫn chạy và câu hỏi thử.
- `chunks-v7.3.2-filtered/chunks-v7.3.2-link-only.jsonl` — 130 đoạn chỉ chứa
  liên kết, dùng khi câu hỏi xin danh sách cá nhân.
- `.env.example` — mẫu tệp `.env`.
- `.gitignore` — loại `.env` và môi trường ảo khỏi Git.

## Cách trả lời

- Mỗi câu hỏi được truy xuất độc lập: nhúng câu hỏi bằng BGE-M3, tìm top 30
  bằng FAISS và top 30 bằng BM25 (tiêu đề nhân 3), kết hợp RRF lấy 10 đoạn.
- Câu hỏi chứa "hiện nay"/"hiện tại" chỉ lấy đoạn của năm mới nhất.
- LLM chỉ trả lời theo tài liệu được cung cấp; thiếu căn cứ thì từ chối.
- Câu hỏi xin danh sách cá nhân chỉ trả về liên kết nguồn, không tiết lộ
  họ tên, mã sinh viên hoặc điểm của cá nhân.
