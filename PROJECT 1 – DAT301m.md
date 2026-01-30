# **PROJECT 1 – SLIDE CONTENT SKELETON**

*(Image/Audio-based-related: CNN vs ViT-Transformer)*

*LINK KAGGLE: https://www.kaggle.com/competitions/beyond-visible-spectrum-ai-for-agriculture-2026*

---

## **0\. COVER SLIDE**

* Project title: **Multimodal Crop Disease Classification**  
* Course: AI Development with TensorFlow(DAT301m)  
* Semester: Spring 2026  
* Team name: Team 2  
* Members & roles:   
  * Nguyễn Thị Minh Thư \- Problem Definition  
  * Triệu Quốc Hào \- CNN Modeling  
  * Nguyễn Lê Gia Bảo \- ViT Modeling  
* Instructor: Huỳnh Văn Thống

---

## **1\. INTRODUCTION & PROBLEM DEFINITION**

### **1.1 Problem Statement**

Vấn đề: Xây dựng model deep learning multimodal để tự động phân loại tình trạng bệnh trên cây lúa mì từ dữ liệu hình ảnh remote sensing được thu thập bằng UAV. Cụ thể là bệnh Rust (1 trong những bệnh phổ biến nhất ở lúa mì) gây thiệt hại về năng xuất

Input: Mỗi sample gồm 3 modalities đồng bộ cho cùng 1 vùng:

- RGB (.png): ảnh màu tự nhiên (gồm 3 kênh)  
- MS (multispectral, .tif): 5 băng tần quan trọng cho phân tích sức khỏe thực vật (Blue \~480nm, Green \~550nm, Red \~650nm, Red Edge \~740nm, NIR \~833nm)  
- HS (hyperspectral, .tif): 125 băng tần liên tục từ 450-950nm, độ phân giải phổ \~4nm, nhưng nhiễu ở 2 đầu phổ(khoảng 10 băng đầu và 14 băng cuối)

Output: Phân loại mỗi patch ảnh vào 1 trong 3 lớp:

- Health: khỏe mạnh  
- Rust: bệnh Rust  
- Other: tình trạng khác

Mục tiêu chính: 

- [ ] Dựa trên thông tin quang phổ từ các modality khác nhau để xử lí sự khác biệt về độ phân giải quang phổ giữa các modality  
- [ ] Đạt độ chính xác cao trên validation set (dữ liệu bị ẩn nhãn)  
- [ ] Đảm bảo model có khả năng thích ứng với dữ liệu remote sensing có độ phân giải và phổ khác nhau

### **1.2 Motivation**

Kinh tế \- xã hội: bệnh Rust là 1 trong những bệnh nghiêm trọng nhất ở lúa mì, mỗi năm trên toàn cầu bệnh Rust gây thiệt hại hàng tỉ USD. Nếu phát hiện sớm và chính xác sẽ giúp giảm thiểu lây lan, hạn chế sử dụng thuốc bảo vệ thực vật, nhờ đấy giảm chi phí và bảo vệ môi trường

Hạn chế của các phương pháp truyền thống: Giám sát thủ công/ảnh RGB thông thường thiếu độ nhạy, đặc biệt ở giai đoạn nhiễm bệnh sớm (khi triệu chứng chưa rõ ràng/chưa nhìn thấy rõ bằng mắt thường). Hyperspectral và multispectral cung cấp thông tin spectral signature chi tiết (ví dụ: thay đổi ở Red Edge và NIR khi cây bị stress), giúp phát hiện bệnh trước khi biểu hiện hình ảnh

Ưu điểm của remote sensing \+ AI: 

- UAV \+ hyperspectral cho độ phân giải cao (\~4nm/pixel) phù hợp quan sát chi tiết  
- Multimodal giúp tăng robustness và accuracy hơn là single modality  
- Giải quyết vấn đề thiếu dữ liệu labelled (đây đc xem là thách thức lớn trong nông nghiệp)

### **1.3 Scope & Constraints**

Scope: 

- Tập trung vào Multimodal Classification trên dữ liệu UAV wheat (RGB, MS, HS)  
- Phát triển end-to-end pipeline: preprocessing (đồng bộ modalities, loại bỏ noise spectral, augmentation), feature fusion, backbone (CNN, ViT), classification head  
- Có thể sử dụng pre-trained weights (ImageNet, hoặc các mô hình remote sensing nếu public), không sử dụng external labelled data  
- Đầu ra: Jupiter notebook reproducible chứa toàn bộ code, từ data loading đến inference trên validation set \-\> xuất file CSV submission (Id, Category)

Constraints: 

- Dữ liệu: Chỉ được dùng dataset cung cấp (train \+ val). Không được phép dùng bất kì external dataset nào (kể cả public hyperspectral datasets khác)  
- Pre-trained models: Được phép dùng ImageNet weights hoặc tương tự, phải công khai và reproduce kết quả. Nếu không reproduce được \-\> mất quyền nhận giải  
- Submission rules:   
+ Chỉ submit mỗi 12h/lần  
+ Deadline: March 1/2026  
+ Không private sharing code giữa các team (vi phạm \-\> disqualify)  
- Evaluation:  
+ Public leaderboard: quantitative metric  
+ Private leaderboard quyết định xếp hạng cuối  
+ Quantitative check code để đảm bảo không cheat

***\*\* Khác:*** Tuân thủ Kaggle Foundational Rules

### **1.4 Practical Applicability**

Precision Agriculture: Triển khai hệ thống giám sát bệnh cây trồng tự động trên diện rộng cho phép nông dân/phân xưởng phát hiện sớm ổ bệnh rust \-\> can thiệp cục bộ (phun thuốc chính xác, loại bỏ cây nhiễm), giảm 30–50% lượng thuốc bảo vệ thực vật và tăng năng suất

Scalability:

- Task 1 (UAV hyperspectral) phù hợp cho farm quy mô vừa và lớn, giám sát chi tiết  
- Task 2 (Sentinel \+ SSL) mở rộng lên cấp quốc gia/toàn cầu, sử dụng dữ liệu vệ tinh miễn phí, giải quyết vấn đề thiếu labelled data \-\> khả thi triển khai thực tế ở các nước đang phát triển

Tích hợp vào hệ thống lớn hơn:

- Kết hợp với IoT, drone tự động, GIS để tạo bản đồ bệnh hại real-time  
- Hỗ trợ nông nghiệp bền vững: giảm tác động môi trường, tuân thủ các tiêu chuẩn xuất khẩu

Mở rộng:

- Áp dụng cho các cây trồng khác (ngô, đậu tương, lúa…) chỉ cần fine-tune  
- Kết hợp thêm thời gian (time-series) để phát hiện stress sớm hơn  
- Triển khai edge computing trên drone để inference real-time, giảm độ trễ

---

## **2\. DATASET ANALYSIS**

### **2.1 Dataset Overview**

* Modalities used: RGB / MS / HS  
* Number of samples:   
- Train set: 600 samples cho mỗi modality (RGB, MS, HS) \-\> tổng 1800 file train. Phân bổ đều: 200 Health, 200 Rust, 200 Other (tổng 600).  
- Validation set: 300 samples cho mỗi modality (RGB, MS, HS) \-\> tổng 900 file val. Labels ẩn (random filenames), ground truth ở result.csv (dùng cho submission và public leaderboard). 

\=\> Tổng dataset: 900 samples labelled (train) \+ 300 samples unlabelled (val) \= 1200 samples.

* Number of classes: 3 lớp  
- Health  
- Rust  
- Other

### **2.2 Data Characteristics**

* Resolution:  
- Độ phân giải không gian \~4 cm/pixel (từ UAV bay ở độ cao 60 mét với S185 snapshot hyperspectral sensor)  
- Độ phân giải phổ: 4 nm/band cho HS, cho phép capture chi tiết spectral signatures của bệnh (ví dụ: thay đổi reflectance ở Red Edge/NIR khi cây bị stress)  
* Channel information:  
- RGB: 3 channels (true-color)  
- MS: 5 bands (vegetation indices-friendly)  
- HS: 125 bands (high-dim spectral, noise ở \~10 bands đầu \+ \~14 bands cuối)  
* Class distribution:  
- Train: Balanced hoàn hảo – 200/600 (33.33%) cho mỗi lớp (Health, Rust, Other).

\=\> Các modalities paired theo prefix "hyper\_" \+ số (ví dụ: health\_hyper\_1.png cho RGB, health\_hyper\_1.tif cho MS/HS) \-\> dễ matching khi load data

- Validation: 300 samples cho mỗi, modality class distribution unknown (labels ẩn)

### **2.3 Data Partitioning Strategy**

* Train / Validation / Test split:   
* Seed strategy:

### **2.4 Challenges in Data**

- Dữ liệu HS có quá nhiều kênh (125 bands): Mô hình dễ bị quá tải, tốn nhiều thời gian tính toán và dễ bị overfit  
- Nhiễu ở đầu và cuối phổ:\~10 đầu và \~14 cuối   
- Multimodal heterogeneity (RGB, MS, HS): Mỗi ảnh có độ phân giải phổ khác nhau nên việc kết hợp khá khó  
- Limited labelled data: Train set nhỏ (đặc trưng hyperspectral agriculture, thu thập đắt đỏ) \-\> dễ overfit   
- Alignment & preprocessing phức tạp: Modalities phải align chính xác (đã cung cấp aligned, nhưng vẫn cần check registration error), format .tif GeoTIFF cần rasterio/geopandas để load đúng geo-info  
- Variability: Dữ liệu từ 2 ngày (May 3 & May 8 2019, pre-grouting & middle grouting) \-\> có thể có temporal variation, nhưng dataset không time-series explicit  
- Reproducibility & submission: Toàn bộ pipeline phải chạy trong Jupyter Notebook. Code cần rõ ràng, đọc dữ liệu đúng cách (rasterio cho .tif, PIL cho .png), và khi chạy inference trên tập validation phải xuất ra file CSV đúng định dạng yêu cầu

---

## **3\. EVALUATION METRICS**

### **3.1 Metrics Used**

* Accuracy  
* Macro-F1

### **3.2 Reason for Choosing Metrics**

* Accuracy:  
- Đơn giản, dễ hiểu, trực quang ranking tổng thể  
- Phù hợp với dữ liệu cân bằng như Wheat Disease Multimodal Classification Dataset   
- Accuracy cao thì khả năng mô hình phát hiện đúng trong nhiều trường hợp hơn \-\> hỗ trợ trực tiếp cho giám sát diện rộng  
- Là metric đánh giá được dùng nhiều trong các cuộc thi nông nghiệp, dễ triển khai, thuận tiện so sánh kết quả  
* Macro-F1:  
- Xử lí tốt class imbalence tiềm ẩn ở val set (vì đây là tập không được label nên chưa biết dữ liệu có cân bằng hay không)  
- Đảm bảo hiệu suất model tốt đều trên các classes, quan trọng là class Rust(minority class \- bệnh gây hại lớn nếu miss detection có thể dẫn đến thiệt hại kinh tế cao)  
- F1 cân bằng precision và recall \-\> tránh model bias về majority class (Health), recall Rust thấp (miss bệnh \-\> không can thiệp kịp thời)  
- Trong remote sensing & hyperspectral classification, Macro-F1 (hoặc weighted variants) thường dùng để đánh giá feature extraction spectral-spatial hiệu quả trên classes khác nhau

---

## **4\. MODEL OVERVIEW (HIGH-LEVEL)**

### **4.1 Overall Pipeline**

**Input \-\> Backbone \-\> Feature \-\> Fusion \-\> Head \-\> Output**

\[PLACEHOLDER: Pipeline Diagram\]

![][image1]

Fig1.Pipeline Diagram

---

## **5\. RGB BACKBONE MODELS**

### **5.1 ResNet-based RGB Encoder**

* Architecture summary:

\[PLACEHOLDER: ResNet Architecture Diagram\]

\[PLACEHOLDER: Code Snippet – ResNet Encoder\]

        \# \===== RGB ResNet \=====

        if self.use\_rgb:

            self.rgb\_enc \= timm.create\_model(

                "resnet18",

                pretrained=True,

                num\_classes=0,

                global\_pool="avg"

            )

            for p in self.rgb\_enc.parameters():

                p.requires\_grad \= False

            for p in self.rgb\_enc.layer4.parameters():

                p.requires\_grad \= True

            rgb\_dim \= self.rgb\_enc.num\_features

            self.rgb\_norm \= nn.LayerNorm(rgb\_dim)

            feat\_dims.append(rgb\_dim)

---

### **5.2 ViT-based RGB Encoder**

* Architecture summary:

\[PLACEHOLDER: ViT Architecture Diagram\]

\[PLACEHOLDER: Code Snippet – ViT Encoder\]

        \# \===== RGB ViT \=====

        if self.use\_rgb:

            self.rgb\_enc \= timm.create\_model(

                "vit\_base\_patch16\_224",

                pretrained=True,

                num\_classes=0

            )

            for p in self.rgb\_enc.parameters():

                p.requires\_grad \= False

            for blk in self.rgb\_enc.blocks\[\-2:\]:

                for p in blk.parameters():

                    p.requires\_grad \= True

            for p in self.rgb\_enc.norm.parameters():

                p.requires\_grad \= True

            self.rgb\_norm \= nn.LayerNorm(self.rgb\_enc.num\_features)

            feat\_dims.append(self.rgb\_enc.num\_features)

---

## **6\. MULTISPECTRAL & HYPERSPECTRAL BRANCHES**

### **6.1 MS Feature Extraction (CNN Small)**

* Architecture description:

\[PLACEHOLDER: MS CNN Block Diagram\]

\[PLACEHOLDER: Code Snippet – MS Encoder\]

        \# \===== MS \=====

        if self.use\_ms:

            self.ms\_enc \= SmallSpectralEncoder(5, 256)

            self.ms\_norm \= nn.LayerNorm(256)

            feat\_dims.append(256)

---

### **6.2 HS Feature Extraction (CNN Small)**

* Architecture description:

\[PLACEHOLDER: HS CNN Block Diagram\]

\[PLACEHOLDER: Code Snippet – HS Encoder\]

        if self.use\_hs:

            self.hs\_enc \= SmallSpectralEncoder(hs\_in\_ch, 256)

            self.hs\_norm \= nn.LayerNorm(256)

            feat\_dims.append(256)

---

## **7\. FEATURE FUSION MECHANISM**

### **7.1 Fusion Strategy**

* Type: Late Fusion / Feature-level Fusion

\[PLACEHOLDER: Fusion Diagram\]

### **7.2 Fusion Implementation**

\[PLACEHOLDER: Code Snippet – Fusion Module\]

        fusion\_dim \= sum(feat\_dims)

        self.gate \= nn.Sequential(

            nn.Linear(fusion\_dim, fusion\_dim),

            nn.Sigmoid()

        )

---

## **8\. CLASSIFICATION HEAD**

### **8.1 Head Architecture**

* Layers:  
* Activation:  
* Regularization:

\[PLACEHOLDER: Code Snippet – Classification Head\]

        self.classifier \= nn.Sequential(

            nn.Linear(fusion\_dim, 512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(512, n\_classes),

        )

---

## **9\. TRAINING STRATEGY**

### **9.1 Training Setup**

* Optimizer:  
* Learning rate: 3e-4  
* Batch size: 32  
* Epochs: 50

### **9.2 Freezing & Fine-tuning Strategy**

* Frozen layers: toàn bộ các layer đầu trừ 2 layer cuối (ViT \+ ResNet)  
* Unfrozen layers:

\[PLACEHOLDER: Code Snippet – Freeze / Unfreeze\]

---

## **10\. EXPERIMENTAL RESULTS**

### **10.1 CNN (ResNet) Results**

* Metrics summary:

\[PLACEHOLDER: Result Table – ResNet\]

---

### **10.2 ViT Results**

* Metrics summary:

\[PLACEHOLDER: Result Table – ViT\]

---

### **10.3 Model Comparison**

* CNN vs ViT

\[PLACEHOLDER: Comparison Table\]

---

## **11\. WANDB LOG ANALYSIS (MANDATORY)**

### **11.1 Training Curves**

* Loss  
* Accuracy  
* Macro-F1

\[PLACEHOLDER: WANDB Screenshots\]

### **11.2 Gradient & Stability Analysis**

\[PLACEHOLDER: WANDB Gradient Charts\]

---

## **12\. DEMO**

### **12.1 Demo Description**

* Input format:  
* Output format:

### **12.2 Demo Results**

\[PLACEHOLDER: Demo Screenshots / GIFs\]

---

## **13\. DISCUSSION**

### **13.1 Observations**

* \[Fill here\]

### **13.2 Strengths & Limitations**

* \[Fill here\]

---

## **14\. CONCLUSION**

* Key takeaways:

---

## **15\. FUTURE WORK**

* Possible improvements:

---

## **16\. REFERENCES**

* Papers:  
* Libraries:

---

## **17\. APPENDIX**

### **A. Full Model Code**

\[PLACEHOLDER\]

### **B. Additional WANDB Logs**

\[PLACEHOLDER\]

[image1]: <data:image/png;base64,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>