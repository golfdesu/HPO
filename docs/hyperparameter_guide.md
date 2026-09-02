# 📘 คู่มือการใช้และการหา Hyperparameters ในงานวิจัย Time-Series Forecasting

---

## 1. ในงานวิจัยเดียวกัน ควรใช้ Hyperparameter เดียวกัน หรือใช้อันที่ดีที่สุดของแต่ละโมเดล?

### 💡 คำตอบ: **ต้องใช้อันที่ดีที่สุดของแต่ละโมเดล (Optimal Hyperparameters per Model)**
การใช้ Hyperparameter ชุดเดียวกันกับทุกโมเดล ถือว่าเป็น **Unfair Comparison (การเปรียบเทียบที่ไม่เป็นธรรม)** ในทางวิชาการ เนื่องจากสถาปัตยกรรมแต่ละแบบมีธรรมชาติและความต้องการทรัพยากรที่แตกต่างกัน

### ⚖️ กฎการแบ่งประเภทตัวแปรในการทดลองวิจัย (Experimental Protocol)

#### 1.1 ตัวแปรที่ต้อง "ล็อกให้เหมือนกัน 100%" (Fixed Conditions)
* **Dataset & Train/Val/Test Split**: สัดส่วนการแบ่งข้อมูลต้องเหมือนกันทุกโมเดล (เช่น Train 60%, Val 20%, Test 20%)
* **Lookback Window ($L$) & Forecast Horizon ($H$)**: ระยะเวลาย้อนหลังและระยะพยากรณ์ล่วงหน้าต้องเท่ากัน (เช่น Lookback=96, Horizon=48)
* **Random Seeds**: ชุด Seed ที่ใช้รันซ้ำต้องเป็นชุดเดียวกัน (เช่น Seeds 164, 256, 355, 1234, 2026)
* **Evaluation Metrics**: สูตรการวัดผลเดียวกัน (MAE, RMSE, WAPE, Peak Zone MAE)

#### 1.2 ตัวแปรที่ต้อง "จูนหาค่าที่ดีที่สุดแยกตามโมเดล" (Model-Specific Tuned Parameters)
* **Learning Rate**: เช่น `1e-3`, `5e-4`, `1e-4`
* **Network Capacity ($d_{model}$, Hidden Units, Num Layers)**: เช่น 2, 3, 4 Layers หรือ $d_{model} \in \{64, 128, 256\}$
* **Dropout / Regularization**: ปรับให้เหมาะสมตามระดับ Overfit ของโมเดลนั้นๆ
* **Model-Specific Parameters**: เช่น Patch Length / Stride (สำหรับ PCD / PatchTST), Factor (สำหรับ Informer), Quantile Head (สำหรับ TFT)

---

### 📝 ข้อความระบุมาตรฐานในบทความวิจัย (Methodology Section)

> *"To ensure a fair comparison, all baseline models and the proposed model were individually tuned on the validation set using grid search (or Bayesian optimization) to find their respective optimal hyperparameter configurations prior to final test set evaluation."*

---

## 2. วิธีการหา Hyperparameter (Hyperparameter Optimization - HPO)

### 🛠️ 2.1 3 วิธีหลักในการหา

1. **Optuna (Bayesian Optimization)** ⭐ *[แนะนำที่สุด - เร็วและฉลาด]*
   * **หลักการ**: ใช้ระบบสถิติเดาว่าตัวแปรไหนน่าจะดีขึ้น โดยวิเคราะห์จากผลรันในรอบก่อนหน้า
   * **ข้อดี**: เจอจุดที่ดีที่สุดเร็วกว่า Grid Search 3–5 เท่า (รัน 15–20 รอบก็เห็นผลชัดเจน)
   * **เครื่องมือ**: ไลบรารี `optuna` ใน Python

2. **Grid Search** *(แบบครอบคลุม - มาตรฐานงานวิจัย)*
   * **หลักการ**: กำหนดรายการค่าตัวแปร แล้วทดสอบจับคู่ทุกความเป็นไปได้ (Exhaustive Search)
   * **ข้อดี**: เป็นระบบ ครอบคลุม ชัดเจน ตรวจสอบง่าย
   * **ข้อเสีย**: ช้ามากหากมีตัวแปรหลายตัว

3. **Random Search** *(แบบสุ่มสโคป)*
   * **หลักการ**: สุ่มค่าตัวแปรตามจำนวนรอบที่กำหนด (เช่น สุ่ม 20 รอบ)
   * **ข้อดี**: รวดเร็ว และมักได้ค่าที่ดีพอสมควร

---

### 📋 2.2 ขั้นตอนการหาที่ถูกต้อง (Search Workflow)

```
1. แบ่งข้อมูล Train / Validation / Test
   ↓
2. เทรนโมเดลบน Train Set ด้วย Hyperparameters ชุดหนึ่ง
   ↓
3. ประเมินผลบน Validation Set (ดู Validation Loss หรือ Validation MAE)
   ↓
4. เลือกชุด Hyperparameters ที่ให้ Validation Loss ต่ำที่สุด
   ↓
5. นำชุด Hyperparameters นั้นไปรันประเมินผลจริงบน Test Set (ด้วย 5 Seeds)
```

⚠️ **ข้อควรระวัง**: ห้ามใช้ Test Set ในการเลือก Hyperparameters เด็ดขาด! ต้องใช้ **Validation Set** เท่านั้น เพื่อป้องกันปัญหา Data Leakage

---

### 📊 2.3 ช่วงค่าตัวแปรแนะนำสำหรับ Time-Series Transformer

| Hyperparameter | ช่วงค่าที่แนะนำให้ลองจูน |
| :--- | :--- |
| **Learning Rate** | `1e-3`, `5e-4`, `1e-4` |
| **d_model (Hidden Dim)** | `64`, `128`, `256` |
| **Num Layers** | `2`, `3`, `4` |
| **Dropout Rate** | `0.05`, `0.1`, `0.2` |
| **Patch Length** (สำหรับ PCD / PatchTST) | `8`, `16`, `24` |
| **Stride** (สำหรับ PCD / PatchTST) | `4`, `8` |

---

### 💻 2.4 ตัวอย่างโค้ด Python สรุปการใช้ Optuna

```python
import optuna
import tensorflow as tf

def objective(trial):
    # 1. นิยามช่วงตัวแปรที่อยากให้ Optuna ค้นหา
    d_model = trial.suggest_categorical('d_model', [64, 128, 256])
    num_layers = trial.suggest_int('num_layers', 2, 4)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-3, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.2)
    patch_len = trial.suggest_categorical('patch_len', [8, 16, 24])
    stride = trial.suggest_categorical('stride', [4, 8, 12])

    # 2. สร้างโมเดลด้วยค่าตัวแปรในรอบนี้
    model = build_model(
        d_model=d_model, 
        num_layers=num_layers, 
        dropout_rate=dropout_rate,
        patch_len=patch_len,
        stride=stride
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    # 3. เทรนโมเดลด้วย Train Set และดูผลบน Validation Set
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=30, 
        verbose=0
    )

    # 4. ส่งคืนค่า Validation Loss ที่ต่ำที่สุด
    best_val_loss = min(history.history['val_loss'])
    return best_val_loss

# 5. สั่งให้ Optuna ค้นหา 20 รอบ
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("🏆 Best Hyperparameters Found:")
print(study.best_params)
```
