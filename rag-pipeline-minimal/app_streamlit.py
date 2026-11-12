"""Giao diện demo cục bộ cho Chatbot-BDU."""

import json
import logging
import os
import re
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import streamlit as st
from dotenv import dotenv_values
from openai import APIError

from rag_streamlit_core import (
    CHUNKS, LINK_ONLY, DEFAULT_BASE_URL, SYSTEM_PROMPT, VEC_BGE,
    Retriever, build_prompt, call_llm,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFUSAL = "Tôi không tìm thấy thông tin này trong tài liệu của trường."
WEB_PROMPT = SYSTEM_PROMPT + """

Riêng giao diện này, xuất đúng một đối tượng JSON với ba khóa:
- status: "answer", "link_only" hoặc "refuse".
- answer: câu trả lời bằng văn bản thuần. Bỏ dòng ghi nguồn, URL và Markdown vì ứng dụng sẽ hiển thị liên kết riêng.
- sources: mảng số nguyên tương ứng với các [Đoạn] thật sự làm căn cứ.
Dùng "link_only" khi câu hỏi xin danh sách cá nhân và có nguồn công bố đúng. Ở trạng thái này chỉ chọn nguồn, để answer là chuỗi rỗng.
Dùng "refuse" khi thiếu căn cứ, để sources là mảng rỗng.
"""


@st.cache_resource
def load_retriever():
    return Retriever(), Lock()


def parse_reply(raw, chunks):
    if not raw:
        raise ValueError("API trả về phản hồi trống.")
    payload = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", raw.strip(), flags=re.I)
    reply = json.loads(payload)
    if not isinstance(reply, dict):
        raise ValueError("Phản hồi không đúng định dạng.")
    status = reply.get("status")
    if status == "refuse":
        return REFUSAL, []
    ids = reply.get("sources")
    if status not in {"answer", "link_only"} or not isinstance(ids, list):
        raise ValueError("Phản hồi không đúng định dạng.")
    by_id = {chunk["n"]: chunk for chunk in chunks}
    sources = {}
    for source_id in ids:
        if type(source_id) is not int or source_id not in by_id:
            continue
        chunk = by_id[source_id]
        url = chunk["source_url"]
        parsed = urlsplit(url) if isinstance(url, str) else None
        if parsed and parsed.scheme in {"http", "https"} and parsed.hostname:
            sources.setdefault(url, chunk["title"])
    if not sources:
        return REFUSAL, []
    if status == "link_only":
        return "Bạn có thể xem danh sách tại nguồn bên dưới.", list(sources.items())
    answer = reply.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Phản hồi thiếu nội dung trả lời.")
    return answer.strip(), list(sources.items())


def render_reply(message):
    st.text(message["answer"])
    for url, title in message["sources"]:
        st.link_button(title or url, url)
    if message["score"] is not None:
        st.caption(f"Cosine cao nhất trong các đoạn đưa vào prompt · {message['score']:.3f}")


def main():
    st.set_page_config(page_title="Chatbot trợ lý sinh viên BDU", page_icon="🎓")
    st.title("Chatbot trợ lý sinh viên BDU")
    st.caption("Tra cứu tài liệu Trường Đại học Bình Dương bằng BM25, BGE-M3, FAISS và LLM.")
    file_settings = dotenv_values(PROJECT_ROOT / ".env", encoding="utf-8-sig")
    api_key = (os.environ.get("LLM_API_KEY") or file_settings.get("LLM_API_KEY") or "").strip()
    model = os.environ.get("LLM_MODEL") or file_settings.get("LLM_MODEL") or "gemini-3.5-flash-lite"
    base_url = os.environ.get("LLM_BASE_URL") or file_settings.get("LLM_BASE_URL") or DEFAULT_BASE_URL

    with st.sidebar:
        st.subheader("Cấu hình thử nghiệm")
        st.caption(f"LLM · {model}")
        use_threshold = st.checkbox("Bật ngưỡng cosine")
        threshold = st.slider("Ngưỡng từ chối", 0.0, 1.0, 0.60, 0.01, disabled=not use_threshold)
        st.caption("Ngưỡng chỉ dùng để thử nghiệm. Mô hình vẫn phải kiểm tra căn cứ trong tài liệu.")

    if not api_key:
        st.info("Điền LLM_API_KEY vào tệp .env ở thư mục gốc Chatbot-BDU rồi khởi động lại Streamlit.")
    missing = [
        str(Path(path).relative_to(PROJECT_ROOT))
        for path in (CHUNKS, LINK_ONLY, VEC_BGE)
        if not Path(path).is_file()
    ]
    if missing:
        st.warning("Thiếu dữ liệu RAG. Đặt đúng vị trí theo README-buoc5.md.")
        st.code("\n".join(missing))

    history = st.session_state.setdefault("history", [])
    for message in history:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["question"])
            else:
                render_reply(message)

    question = st.chat_input("Bạn muốn hỏi gì về BDU?", max_chars=1000, disabled=bool(missing or not api_key))
    if not question or not question.strip():
        return
    question = question.strip()
    history.append({"role": "user", "question": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"), st.spinner("Đang tìm căn cứ trong tài liệu BDU..."):
        try:
            retriever, lock = load_retriever()
            with lock:
                ranked, score = retriever.retrieve(question)
            if not ranked or (use_threshold and score < threshold):
                answer, sources = REFUSAL, []
            else:
                chunks = retriever.format_chunks(ranked)
                prompt = build_prompt(question, chunks)
                raw = call_llm(prompt, model, base_url, api_key, system_prompt=WEB_PROMPT)
                answer, sources = parse_reply(raw, chunks)
        except APIError as error:
            logging.warning("Lỗi LLM: %s", type(error).__name__)
            st.error("Gọi LLM thất bại. Kiểm tra API key, tên mô hình, địa chỉ API hoặc hạn mức.")
            return
        except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
            logging.warning("Lỗi xử lý câu hỏi: %s", type(error).__name__)
            st.error("Chưa xử lý được câu hỏi. Kiểm tra dữ liệu, kết nối tải BGE-M3 hoặc thử lại.")
            return
        message = {"role": "assistant", "answer": answer, "sources": sources, "score": score}
        history.append(message)
        render_reply(message)


if __name__ == "__main__":
    main()
