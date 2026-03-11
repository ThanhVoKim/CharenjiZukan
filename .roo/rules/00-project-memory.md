# Quy tắc ghi nhớ dự án (Project Memory Rule)

Để đảm bảo các Agent khác có thể tiếp nối công việc, Agent hiện tại PHẢI tuân thủ:

1. **Duy trì file `JOURNAL.md`**: Mọi thay đổi lớn về kiến trúc, các quyết định quan trọng hoặc thay đổi trong luồng (flow) phải được ghi lại vào file `JOURNAL.md` ở thư mục `logs`.
2. **Cung cấp ngữ cảnh hiện tại**: Trước khi kết thúc nhiệm vụ, Agent phải tóm tắt trạng thái hiện tại của dự án bao gồm:
   - Những gì đã hoàn thành.
   - Những vấn đề còn tồn đọng.
   - Các bước tiếp theo được đề xuất.
3. **Luồng dữ liệu (Flow)**: Luôn tham chiếu đến sơ đồ hoặc tài liệu flow trong thư mục `docs/` để đảm bảo code mới không phá vỡ logic chung.
