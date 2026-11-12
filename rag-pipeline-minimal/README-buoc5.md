# Chatbot trợ lý sinh viên BDU

Demo Streamlit cho bước dựng pipeline RAG. Mỗi câu hỏi được truy xuất độc lập. Lịch sử trên giao diện dùng để xem lại cuộc trò chuyện.

## Khởi động trên Windows

1. Giải nén ở đường dẫn ngắn, ví dụ `D:\Chatbot-BDU`, rồi mở thư mục `Chatbot-BDU` trong VS Code.
2. Mở tệp `.env` ở thư mục gốc và điền khóa Gemini ngay sau `LLM_API_KEY=`. Nếu cần tạo lại tệp, sao chép từ `.env.example`.
3. Chạy các lệnh sau trong PowerShell tại thư mục gốc. Hai dòng đầu chỉ cần khi tạo môi trường và cài thư viện.

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r .\rag-pipeline-minimal\requirements-streamlit.txt
   .\.venv\Scripts\python.exe -m streamlit run .\rag-pipeline-minimal\app_streamlit.py --server.address 127.0.0.1
   ```

4. Mở [localhost:8501](http://localhost:8501). Lần hỏi đầu tiên có thể mất thời gian tải BGE-M3 về máy. Máy dùng CPU để nhúng câu hỏi, còn LLM sinh câu trả lời qua API.

Biến môi trường PowerShell được ưu tiên nếu đã đặt. Có thể chỉnh `LLM_MODEL` và `LLM_BASE_URL` trong `.env` để dùng cấu hình khác. Tệp `.gitignore` loại `.env` khỏi Git.

## Dữ liệu của app

| Đường dẫn tính từ thư mục gốc | Vai trò |
|---|---|
| `chunks-v7.3.2-filtered/chunks-v7.3.2-filtered.jsonl` | 12.910 đoạn nội dung |
| `chunks-v7.3.2-filtered/chunks-v7.3.2-link-only.jsonl` | 130 đoạn dùng để tìm liên kết |
| `embedding-colab-v7.3.2-filtered/vector-result/vectors_bge_m3_with_linkonly.npz` | 13.040 vector BGE-M3 khớp với hai tệp JSONL |

## Cách truy xuất và kiểm tra

- FAISS dựng chỉ mục cosine từ tệp vector có sẵn khi cần truy xuất lần đầu. Chỉ mục được giữ trong bộ nhớ. Dùng mã chunk để kiểm tra và căn chỉnh vector với văn bản.
- BM25 và FAISS được kết hợp bằng RRF, lấy tối đa 10 đoạn đưa vào prompt. App hiển thị liên kết từ `source_url` của đoạn mà LLM chọn làm căn cứ.
- Ngưỡng cosine là tùy chọn thử nghiệm và mặc định tắt. Khi bật, app từ chối trước khi gọi LLM nếu mọi đoạn được chọn đều có cosine dưới ngưỡng. Số hiển thị là cosine cao nhất trong các đoạn được chọn. Điểm cao chỉ thể hiện độ tương đồng và vẫn cần LLM kiểm tra căn cứ.

| Câu hỏi thử | Hành vi cần kiểm tra |
|---|---|
| Xếp loại tốt nghiệp được xác định như thế nào theo điểm trung bình tích lũy thang điểm 4? | Trả lời các mốc GPA và dẫn quy chế |
| Chuẩn đầu ra của ngành Công nghệ thông tin trình độ đại học gồm những năng lực gì? | Từ chối nếu tài liệu thiếu chuẩn đầu ra đúng ngành |
| Cho tôi danh sách sinh viên đủ điều kiện công nhận tốt nghiệp đợt 2 năm 2025. | Dẫn thông báo kết quả đúng đợt, chỉ hiển thị liên kết |
