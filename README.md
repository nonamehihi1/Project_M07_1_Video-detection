# Project_M07_1_Video-detection
Trong dự án này sử dụng đầu vào là video ngắn và đầu ra là dự đoán hành động. Sử dụng LS_ViT

Sử dụng bộ dữ liệu:
https://huggingface.co/datasets/jili5044/hmdb51

Trong bài này sẽ sử dụng mô hình(LS-ViT), kiến trúc sẽ gồm các thành phần sau:
+ Các module ViT cơ bản
+ SMIFModule: nắm bắt các chuyển động ngắn hạn bằng cách tính toán sự khác biệt giữa các khung hình lân cận để tạo ra bản đồ chuyển động
<img width="851" height="462" alt="{39ED7E02-56F0-462F-9302-39EACD8104A2}" src="https://github.com/user-attachments/assets/e45947bb-ed6f-4ce2-aa11-2897f53cb6ec" />

Module này hoạt động bằng cách làm nổi bật những hành động hơn so với những điểm ảnh còn lại

+ LMIModule: Giúp mô hình nắm bắt sự phụ thuộc dài hạn giữa các token trong chuỗi thời gian thông qua cơ chế attention
<img width="900" height="365" alt="{AD752D85-12CD-4C82-8E51-B3313B34CF19}" src="https://github.com/user-attachments/assets/4ea174e1-403f-403e-9dbf-5cb4e471534c" />

Module này hoạt động như sau:
1. Giảm chiều dữ liệu C xuống C/r
2. Tính toán chuyển động bằng cách lấy frame ((t-1) - t) + (t - (t+1)). Việc này giúp nắm bắt sự thay đổi theo chiều thời gian
3. Gộp thông tin, Tensor được tính trung bình theo chiều N để thu được Tensor có kích thước mới, giúp tập trung vào sự thay đổi theo thời gian
4. Sau đó mạng đưa qua lớp MLP + sigmoid để xem khung hình nào chứa chuyển động quan trọng
5. Cuối cùng là cộng đầu ra với ảnh gốc để thu được ảnh chứa đồng thời thông tin gốc bức ảnh và các 'gợi ý' chuyển động

+ Kiến trúc Backbone và Head: Mô hình được xây dựng bằng cách xếp chồng các khối LSViT, mỗi khối bao gồm lớp Attention chuẩn, lớp MLP và LMIModule

<img width="282" height="128" alt="{C0908FB5-87B4-4588-9452-131D5D20E927}" src="https://github.com/user-attachments/assets/be5770df-3edb-41e2-95d1-fba86675761e" />

