---
title: Mỗi khi làm một project AI tôi sẽ vào đây đọc
date: 2023-10-3 15:30:00 +/-TTTT
categories: [Machine Learning]
tags: [overview, basic, deep learning, machine learning]
toc: true
math: true
comments: true
published: true
img_path: /pic/AI_project_map/
---



# 8 bước chuẩn bị cho một dự án liên quan đến lĩnh vực AI (Machine learning)

Bày viết này là tổng hợp những kiến thức và kinh nghiệm của tôi nghiên cứu được về các bước trong việc triển khai một dự án liên quan đến lĩnh vực AI, và có tham khảo từ nhiều nguồn.

![Map](map.png)

## 1. Xác định vấn đề (Define The Problem)

Bước đầu tiên quan trọng trong quá trình thực hiện một dự án Machine Learning là xác định vấn đề cụ thể mà chúng ta muốn giải quyết. Trong quá trình này, chúng ta cần xác định rõ mục tiêu của dự án, không nhất thiết phải giới hạn trong khía cạnh kinh doanh, nhưng việc hiểu rõ cách giải quyết vấn đề cuối cùng bằng Machine Learning là rất quan trọng. Bước này cũng là thời điểm để xem xét và so sánh các kịch bản và phương pháp giải quyết vấn đề, cũng như thảo luận về các giả định và kiến thức chuyên môn cần thiết. Trong quá trình này, chúng ta cũng cần xác định loại vấn đề Machine Learning cụ thể (có giám sát, không giám sát, v.v.) đều này là quan trọng để bạn chuẩn bị loại dữ liệu cần thiết, và đặt ra các chỉ tiêu hiệu suất mà chúng ta mong muốn đạt được.

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- [x] Xác định vấn đề cần giải quyết.
- [x] Mục tiêu của dự án.
- [x] Các kiến thức chuyên môn cần có.

## 2. Thu thập dữ liệu (Collect Data)

Trong bước thu thập dữ liệu, chúng ta tập trung vào việc quản lý dữ liệu một cách chi tiết và tổ chức. Đầu tiên, chúng ta cần xác định số lượng dữ liệu cần thiết và loại dữ liệu mà chúng ta cần thu thập. Điều này đòi hỏi chúng ta phải biết rõ ràng về loại dữ liệu mà dự án yêu cầu, có thể là dữ liệu thời gian, quan sát, hình ảnh, và nhiều loại dữ liệu khác.

Tiếp theo, chúng ta phải quan tâm đến việc thu thập dữ liệu. Điều này bao gồm xác định nguồn dữ liệu cụ thể, đảm bảo tuân thủ các quy tắc và quy định pháp lý liên quan đến việc thu thập dữ liệu.

Khi đã có dữ liệu, việc đảm bảo sự ẩn danh của nó là quan trọng. Chúng ta cần bảo đảm rằng thông tin cá nhân không được tiết lộ và tạo các biện pháp bảo vệ dữ liệu đối với các tập dữ liệu nhạy cảm.

Cuối cùng, chúng ta phải chuyển đổi và chuẩn bị dữ liệu để phù hợp với quy trình đào tạo mô hình AI. Điều này bao gồm việc tạo các tập dữ liệu đào tạo, xác nhận và kiểm tra, đảm bảo rằng dữ liệu sẵn sàng cho việc xây dựng và đào tạo mô hình.

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- [x] Xác định độ lớn của dữ liệu, và loại dữ liệu.
- [x] Thu thập dữ liệu: nguồn dữ liệu, phương pháp thu thập, cần phải tuân thủ các quy tắc và quy định pháp lý.
- [x] Đảm bảo tính an toàn thông tin dữ liệu, bảo vệ dữ liệu.
- [x] Quản lý dữ liệu.

## 3. Khám phá dữ liệu (Data Exploration)

Bước này trong quy trình kiểm tra dữ liệu có sự tương đồng với phân tích dữ liệu khám phá (Exploratory Data Analysis - EDA) trong lĩnh vực thống kê và khoa học dữ liệu. Mục tiêu chính ở đây là khám phá và hiểu sâu hơn về dữ liệu trước khi bắt đầu xây dựng mô hình. Trong giai đoạn này, chúng ta cần kiểm tra lại các giả định ban đầu về dữ liệu và tìm hiểu về chúng.

Ở bước này đôi khi chúng ta cần sự trợ giúp của chuyên gia để giải quyết các câu hỏi phức tạp liên quan đến mối tương quan trong dữ liệu, đặc biệt đối với những người mới bắt đầu trong lĩnh vực Machine Learning. Trong quá trình này, ta cần nghiên cứu các thuộc tính và đặc điểm của dữ liệu, cũng như biểu đồ hóa một cách tổng quan để trực quan hóa các thuộc tính và giá trị tương ứng (có thể sử dụng biểu đồ hơn là tính toán số học để đơn giản hóa việc hiểu rõ vấn đề). Hãy lưu trữ các phát hiện và nhận thức của bạn trong tài liệu để sử dụng sau này trong quá trình phân tích và xây dựng mô hình. Ví dụ như bạn đang EDA một bộ dữ liệu y khoa thì bạn cần phải có sự tư vấn từ các bác sĩ, chứ bạn đâu đủ kiến thức y học để phân tích những bộ dữ liệu liên quan đến y học này đúng không? hoặc là khám phá xem dữ liệu của bạn có bị mất cân bằng (Imbalance) hay không?, dữ liệu của bạn có phù hợp với model bạn sử dụng hay không?...

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- [x] EDA dữ liệu.
- [x] Xác định các vấn đề cụ thể liên quan đến dữ liệu.


## 4. Chuẩn bị dữ liệu (Data Preparation)

Chuẩn bị dữ liệu hoặc tiền xử lý dữ liệu là bước này là thời điểm để thực hiện các biến đổi dữ liệu mà bạn đã xác định là hữu ích trong các bước trước. Nó bao gồm việc làm sạch dữ liệu, lựa chọn các thuộc tính quan trọng, và áp dụng các phương pháp kỹ thuật. Trong quy trình này, cũng sẽ tiến hành chuẩn hóa dữ liệu để đảm bảo tính thống nhất. Mô hình học từ dữ liệu nên chất lượng dữ liệu kém có thể khiến mô hình không hiệu quả sau khi được triển khai. Dữ liệu phải được preprocessing (tiền xử lý thật tốt) theo các bước ở dưới:

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- [x] Clean data (Làm sạch dữ liệu).
- [x] Outlier Removal (Loại bỏ giá trị ngoại lệ, ngoại lai, nhiễu).
- [x] Missing Data Handling (Xử lý dữ liệu thiếu).
- [x] Feature extraction (Trích xuất đặc trưng).
- [x] Áp dụng các phương pháp kỹ thuật xử lý dữ liệu mô hình đạt kết quả tốt hơn.


## 5. Xây dựng mô hình (Modeling)

Ở trong bước này chúng ta sẽ lựa chọn các mô hình Machine learning, Deep learning phù hợp với vấn đề và dữ liệu của chúng ta, tùy thuộc vào từng tình huống mà chúng ta sẽ chọn những model khác nhau, và thử trên nhiều model khác nhau để được kết quả tốt nhất. Sau đó bạn sẽ huấn luyện các mô hình được chọn bằng cách điều chỉnh các tham số và học cách biểu diễn quan hệ giữa đầu vào và đầu ra. Quá trình này có thể đòi hỏi nhiều lần thử nghiệm và điều chỉnh.

Trong quá trình huấn luyện, bạn sẽ chuyển dữ liệu đã chuẩn bị trước đó đến mô hình học máy của mình, để hệ thống tìm các mẫu hữu ích và đưa ra dự đoán. Qua đó, hệ thống có thể học hỏi từ dữ liệu để hoàn thành những nhiệm vụ đã được đặt ra. Theo thời gian, cùng với quá trình huấn luyện và đào tạo, mô hình Machine Learning sẽ hoạt động và dự đoán tốt hơn.

Sau khi đã huấn luyện mô hình, bạn cần kiểm tra lại xem chúng đã hoạt động chính xác như mong muốn chưa. Bạn có thể đánh giá bằng cách kiểm tra kết quả phân tích của mô hình trên một dữ liệu chưa từng thấy trước đó, xem thử hệ thống có nhận dạng được đúng và dự đoán đúng hay không. Khi đánh giá như vậy, bạn sẽ có được một thước đo đúng về cách mô hình của bạn hoạt động, cũng như tốc độ xử lý thông tin và dự đoán của nó.

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- [x] Lựa chọn mô hình phù hợp (Model Selection).
- [x] Huấn luyện mô hình (Train model).
- [x] Đánh giá mô hình bằng các phương pháp đo lường phù hợp (Model Evaluation).

## 6. Tinh chỉnh mô hình (Model Fine-Tuning)

Tinh chỉnh mô hình là quá trình quan trọng sau khi bạn đã hoàn thành quá trình huấn luyện và đánh giá mô hình Machine Learning của mình. Trong giai đoạn này, bạn tối ưu hóa các tham số của mô hình để cải thiện hiệu suất hoặc hiệu quả của nó trong hệ thống của bạn.

Tham số trong mô hình Machine Learning là các biến được sử dụng để mô tả và ảnh hưởng đến quá trình học của mô hình. Đây là các giá trị mà lập trình viên hoặc người huấn luyện mô hình thiết lập ban đầu. Mục tiêu của việc tinh chỉnh tham số là tìm ra giá trị tốt nhất cho các tham số này để tối ưu hóa độ chính xác hoặc hiệu suất của mô hình.

Quá trình này thường đòi hỏi sự thử nghiệm và đánh giá liên tục của mô hình trên tập dữ liệu kiểm tra hoặc trong môi trường thực tế. Bằng cách điều chỉnh các tham số như tỷ lệ học (learning rate), kiến trúc mạng neural, kích thước batch, số lượng epoch, và các tham số khác, bạn có thể tìm ra giá trị tối ưu để đạt được độ chính xác cao nhất hoặc hiệu suất tốt nhất cho mô hình của mình.

Ngắn gọn lại ở bước này chúng ta cần giải quyết rõ những vấn đề sau:

- [x] Điều chỉnh các tham số để model đạt được hiệu quả tốt nhất

Ngoài ra tinh chỉnh mô hình còn có một loại khác đó là một kỹ thuật học sâu giúp cải thiện hiệu suất của mô hình cho một nhiệm vụ cụ thể. Kỹ thuật này hoạt động bằng cách sử dụng một mô hình đã được đào tạo trước trên một tập dữ liệu lớn và sau đó điều chỉnh các tham số của mô hình để phù hợp với một tập dữ liệu nhỏ hơn, cụ thể hơn. nếu bạn không xây model từ đầu.

Quy trình tinh chỉnh mô hình kiểu này thường bao gồm các bước sau:

- [x] Chọn một mô hình đã được đào tạo trước trên một tập dữ liệu lớn.
- [x] Tạo một mô hình mới dựa trên mô hình đã được đào tạo trước.
- [x] Đóng băng các lớp đầu tiên của mô hình mới.
- [x] Huấn luyện các lớp cuối của mô hình mới trên tập dữ liệu nhỏ.

Ví dụ trực quan: Giả sử bạn có một tập dữ liệu hình ảnh của chó và mèo. Tập dữ liệu này khá nhỏ, chỉ có 100 hình ảnh. Bạn muốn xây dựng một mô hình để phân loại các hình ảnh này, bạn có thể sử dụng một mô hình đã được đào tạo trước trên một tập dữ liệu hình ảnh lớn. Mô hình này có thể học hỏi các đặc điểm chung của hình ảnh, chẳng hạn như hình dạng, màu sắc và kết cấu.

Sau đó, bạn có thể tinh chỉnh mô hình này trên tập dữ liệu 100 hình ảnh của chó và mèo. Bạn sẽ đóng băng các lớp đầu tiên của mô hình và chỉ huấn luyện các lớp cuối.

Kết quả là, mô hình của bạn sẽ học hỏi các đặc điểm cụ thể của chó và mèo. Điều này sẽ giúp mô hình phân loại các hình ảnh trong tập dữ liệu 100 hình ảnh chính xác hơn.

## 7. Đưa ra dự đoán (Generate Predictions)

Cuối cùng, chúng ta sẽ sử dụng mô hình Machine Learning mà mình đã tạo để phân tích thông tin và đưa ra dự đoán một cách chính xác.

- [x] Dự đoán được kết quả tốt với những vấn đề đặt ra.

## 8. Trình bày giải pháp & Tạo ra sản phẩm (Presenting Solutions & Building Products)

Bước này thì thường rất ít dự án AI có thể chạm tới, bởi vì đa số các dự án AI đều mang tính học thuật là chính, nhưng nếu project của bạn đủ tốt bạn có thể viết paper (bài báo) nộp các hội nghị khoa học hoặc các tạp chí khoa học nổi tiếng trên thế giới, thậm chí bạn có thể ứng dụng các giải pháp AI này vào sản phẩm của mình.

Dưới đây là một số ví dụ về "Trình bày giải pháp & Tạo ra sản phẩm" trong lĩnh vực AI:

1. Một bạn sinh viên có thể nghiên cứu về AI để giải quyết một vấn đề trong cuộc sống, sau đó lên ý tưởng, tìm ra các giải pháp, tạo ra những giải pháp mới, thực hiện nó, sau đó viết một bài báo nghiên cứu khoa học để nộp các hội nghị, tạp chí uy tín, hoặc tự làm ra sản phẩm để bán.
2. Một công ty sử dụng AI để phát triển một hệ thống tự lái mới. Quá trình này bao gồm việc xác định vấn đề của lái xe tự lái, tạo ra các ý tưởng giải pháp, phát triển các mô hình AI và tạo ra một hệ thống tự lái hoạt động.
3. Một ngân hàng sử dụng AI để phát triển một hệ thống phát hiện gian lận mới. Quá trình này bao gồm việc xác định vấn đề của gian lận, tạo ra các ý tưởng giải pháp, phát triển các mô hình AI và tạo ra một hệ thống phát hiện gian lận có thể xác định các giao dịch gian lận.
4. Một công ty sử dụng AI để phát triển một hệ thống hỗ trợ khách hàng mới. Quá trình này bao gồm việc xác định vấn đề của dịch vụ khách hàng, tạo ra các ý tưởng giải pháp, phát triển các mô hình AI và tạo ra một hệ thống hỗ trợ khách hàng có thể trả lời các câu hỏi của khách hàng một cách chính xác và nhanh chóng.

- [x] Paper
- [x] Product


Lưu ý là những bước trên đây mình đặt ra cần có sự cố gắng và lặp đi lặp lại nhiều lần để cho ra được kết quả tuyệt vời.

## Tham khảo

[1] [https://www.seldon.io/how-to-build-a-machine-learning-model](https://www.seldon.io/how-to-build-a-machine-learning-model)

[2] [https://ohstem.vn/7-buoc-xay-dung-mo-hinh-machine-learning/](https://ohstem.vn/7-buoc-xay-dung-mo-hinh-machine-learning/)

[3] [https://www.thegioimaychu.vn/blog/ai-hpc/checklist-8-buoc-chuan-bi-cho-mot-du-an-machine-learning-p552/](https://www.thegioimaychu.vn/blog/ai-hpc/checklist-8-buoc-chuan-bi-cho-mot-du-an-machine-learning-p552/)


## Bình luận & thảo luận

Cảm ơn bạn đã dành thời gian để đọc, hãy trò chuyện và góp ý với mình ở dưới hoặc vào bằng <a href = "https://forms.gle/ZUrzUFKadCJBAEzaA"> link </a>.

<iframe src="https://docs.google.com/forms/d/e/1FAIpQLSdYX6124QWR49d27Gu08whQH9MhDvXeW9o4KkA-kblLt4URwA/viewform?embedded=true" width="640" height="686" frameborder="0" marginheight="0" marginwidth="0">Đang tải…</iframe>
