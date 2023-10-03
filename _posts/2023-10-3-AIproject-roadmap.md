---
title: Mỗi khi làm một project AI tôi sẽ vào đây đọc
date: 2023-09-27 15:30:00 +/-TTTT
categories: [Machine learning]
tags: [overview, basic, deep learning, machine learning]
toc: true
math: true
comments: true
published: true
img_path: /pic/DL1/
---



# 8 bước chuẩn bị cho một dự án liên quan đến lĩnh vực AI (Machine learning)

Bày viết này là tổng hợp những kiến thức và kinh nghiệm của tôi nghiên cứu được về các bước trong việc triển khai một dự án liên quan đến lĩnh vực AI, và có tham khảo từ nhiều nguồn.

## 1. Xác định vấn đề (Define the problem)

Bước đầu tiên quan trọng trong quá trình thực hiện một dự án Machine Learning là xác định vấn đề cụ thể mà chúng ta muốn giải quyết. Trong quá trình này, chúng ta cần xác định rõ mục tiêu của dự án, không nhất thiết phải giới hạn trong khía cạnh kinh doanh, nhưng việc hiểu rõ cách giải quyết vấn đề cuối cùng bằng Machine Learning là rất quan trọng. Bước này cũng là thời điểm để xem xét và so sánh các kịch bản và phương pháp giải quyết vấn đề, cũng như thảo luận về các giả định và kiến thức chuyên môn cần thiết. Trong quá trình này, chúng ta cũng cần xác định loại vấn đề Machine Learning cụ thể (có giám sát, không giám sát, v.v.) được áp dụng và đặt ra các chỉ tiêu hiệu suất mà chúng ta mong muốn đạt được.

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- Xác định vấn đề cần giải quyết
- Mục tiêu của dự án
- Các kiến thức chuyên môn cần có

## 2. Thu thập dữ liệu (Collect data)

Trong bước thu thập dữ liệu, chúng ta tập trung vào việc quản lý dữ liệu một cách chi tiết và tổ chức. Đầu tiên, chúng ta cần xác định số lượng dữ liệu cần thiết và loại dữ liệu mà chúng ta cần thu thập. Điều này đòi hỏi chúng ta phải biết rõ ràng về loại dữ liệu mà dự án yêu cầu, có thể là dữ liệu thời gian, quan sát, hình ảnh, và nhiều loại dữ liệu khác.

Tiếp theo, chúng ta phải quan tâm đến việc thu thập dữ liệu. Điều này bao gồm xác định nguồn dữ liệu cụ thể, đảm bảo tuân thủ các quy tắc và quy định pháp lý liên quan đến việc thu thập dữ liệu.

Khi đã có dữ liệu, việc đảm bảo sự ẩn danh của nó là quan trọng. Chúng ta cần bảo đảm rằng thông tin cá nhân không được tiết lộ và tạo các biện pháp bảo vệ dữ liệu đối với các tập dữ liệu nhạy cảm.

Cuối cùng, chúng ta phải chuyển đổi và chuẩn bị dữ liệu để phù hợp với quy trình đào tạo mô hình AI. Điều này bao gồm việc tạo các tập dữ liệu đào tạo, xác nhận và kiểm tra, đảm bảo rằng dữ liệu sẵn sàng cho việc xây dựng và đào tạo mô hình.

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- Xác định độ lớn của dữ liệu, và loại dữ liệu
- Thu thập dữ liệu: nguồn dữ liệu, phương pháp thu thập, cần phải tuân thủ các quy tắc và quy định pháp lý
- Đảm bảo tính an toàn thông tin dữ liệu, bảo vệ dữ liệu
- Quản lý dữ liệu 

## 3. Khám phá dữ liệu (Data exploration)

Bước này trong quy trình kiểm tra dữ liệu có sự tương đồng với phân tích dữ liệu khám phá (Exploratory Data Analysis - EDA) trong lĩnh vực thống kê và khoa học dữ liệu. Mục tiêu chính ở đây là khám phá và hiểu sâu hơn về dữ liệu trước khi bắt đầu xây dựng mô hình. Trong giai đoạn này, chúng ta cần kiểm tra lại các giả định ban đầu về dữ liệu và tìm hiểu về chúng.

Ở bước này đôi khi chúng ta cần sự trợ giúp của chuyên gia để giải quyết các câu hỏi phức tạp liên quan đến mối tương quan trong dữ liệu, đặc biệt đối với những người mới bắt đầu trong lĩnh vực Machine Learning. Trong quá trình này, ta cần nghiên cứu các thuộc tính và đặc điểm của dữ liệu, cũng như biểu đồ hóa một cách tổng quan để trực quan hóa các thuộc tính và giá trị tương ứng (có thể sử dụng biểu đồ hơn là tính toán số học để đơn giản hóa việc hiểu rõ vấn đề). Hãy lưu trữ các phát hiện và nhận thức của bạn trong tài liệu để sử dụng sau này trong quá trình phân tích và xây dựng mô hình. Ví dụ như bạn đang EDA một bộ dữ liệu y khoa thì bạn cần phải có sự tư vấn từ các bác sĩ, chứ bạn đâu đủ kiến thức y học để phân tích những bộ dữ liệu liên quan đến y học này đúng không? hoặc là khám phá xem dữ liệu của bạn có bị mất cân bằng (Imbalance) hay không?, dữ liệu của bạn có phù hợp với model bạn sử dụng hay không?...

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- EDA dữ liệu
- Xác định các vấn đề cụ thể liên quan đến dữ liệu

## Bình luận & thảo luận

Cảm ơn bạn đã dành thời gian để đọc, hãy trò chuyện và góp ý với mình ở dưới hoặc vào bằng <a href = "https://forms.gle/ZUrzUFKadCJBAEzaA"> link </a>.

<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSdYX6124QWR49d27Gu08whQH9MhDvXeW9o4KkA-kblLt4URwA/viewform?embedded=true" width="640" height="686" frameborder="0" marginheight="0" marginwidth="0">Đang tải…</iframe>
