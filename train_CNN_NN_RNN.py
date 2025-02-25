import os
import cv2
import numpy as np
import glob
import time
import math
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt

# Định nghĩa thư mục chứa tập dữ liệu hình ảnh
THU_MUC = r"D:/KLTN/data_cam_xuc100"
LOAI = ["binh_thuong", "buon", "cuoi", "ngac_nhien", "so_hai", "tuc_gian"]

# Hàm load dữ liệu chung cho CNN và MLP
def load_data(data_path, img_size=(48, 48), is_mlp=False):
    images, labels = [], []
    supported_formats = ("*.jpg", "*.png", "*.jpeg")
    
    for label in LOAI:
        label_path = os.path.join(data_path, label)
        for file_format in supported_formats:
            for file in glob.glob(os.path.join(label_path, file_format)):
                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                
                if is_mlp:
                    images.append(img.flatten())  # Flatten for MLP
                else:
                    images.append(img)  # Keep 2D for CNN
                
                labels.append(label)
    
    return np.array(images), np.array(labels)

# Load dữ liệu cho cả CNN và MLP
du_lieu, nhan = load_data(THU_MUC, is_mlp=False)
du_lieu = du_lieu.reshape(len(du_lieu), 48, 48, 1)  # Đổi hình dạng cho CNN

# Chuyển đổi nhãn sang giá trị số học
lb = LabelBinarizer()
nhan = lb.fit_transform(nhan)

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
(tap_huan_luyen_X, tap_kiem_tra_X, tap_huan_luyen_Y, tap_kiem_tra_Y) = train_test_split(du_lieu, nhan, test_size=0.20, stratify=nhan, random_state=math.floor(time.time()))

# ---- Huấn luyện mô hình CNN ----
cnn_model = Sequential()
cnn_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(64, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Conv2D(128, (3, 3), activation='relu'))
cnn_model.add(MaxPooling2D((2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(128, activation='relu'))
cnn_model.add(Dropout(0.5))  # Lớp Dropout
cnn_model.add(Dense(len(LOAI), activation='softmax'))

cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

checkpoint = ModelCheckpoint('cnn_best_model.h5', save_best_only=True)
cnn_history = cnn_model.fit(datagen.flow(tap_huan_luyen_X, tap_huan_luyen_Y, batch_size=32),
                            epochs=50, validation_data=(tap_kiem_tra_X, tap_kiem_tra_Y), callbacks=[checkpoint])

# ---- Huấn luyện mô hình MLP ----
X, y = load_data(THU_MUC, is_mlp=True)

# Chuẩn hóa dữ liệu cho MLP
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=math.floor(time.time()))

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), 
                    activation='relu', 
                    solver='adam', 
                    max_iter=50, 
                    random_state=math.floor(time.time()), 
                    verbose=True, 
                    learning_rate_init=1e-5)

train_accuracies, test_accuracies = [], []
train_losses, test_losses = [], []
print("Đánh giá mô hình...")

# Dự đoán cho CNN trên tập huấn luyện
du_doan_train = cnn_model.predict(tap_huan_luyen_X)
y_true_train = np.argmax(tap_huan_luyen_Y, axis=1)
y_pred_train = np.argmax(du_doan_train, axis=1)
do_chinh_xac_huan_luyen = accuracy_score(y_true_train, y_pred_train)
phan_tram_du_doan_dung_huan_luyen = do_chinh_xac_huan_luyen * 100

# Dự đoán cho CNN trên tập kiểm tra
du_doan_test = cnn_model.predict(tap_kiem_tra_X)
cnn_pred_classes = np.argmax(du_doan_test, axis=1)
y_true_cnn = np.argmax(tap_kiem_tra_Y, axis=1)


# In ra báo cáo phân loại cho CNN
print("Báo cáo phân loại cho mô hình huấn luyện CNN:")
print(classification_report(y_true_train, y_pred_train, target_names=LOAI))
print("Báo cáo phân loại cho mô hình CNN:")
print(classification_report(y_true_cnn, cnn_pred_classes, target_names=LOAI))


# Huấn luyện mô hình MLP (từng epoch một)
for epoch in range(1, 51):  # Loop for each epoch
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_encoded))  # Manual training step
    
    # Dự đoán cho tập huấn luyện và tập kiểm tra
    train_accuracy = mlp.score(X_train, y_train)
    y_pred = mlp.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Dự đoán cho tập huấn luyện và tập kiểm tra
    train_accuracy = mlp.score(X_train, y_train)
    y_train_pred = mlp.predict(X_train)  # Dự đoán cho tập huấn luyện
    y_test_pred = mlp.predict(X_test)  # Dự đoán cho tập kiểm tra
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Tính toán loss
    loss = log_loss(y_test, mlp.predict_proba(X_test))

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    train_losses.append(loss)
    test_losses.append(loss)
    
    # In thông số chi tiết
    print(f"Epoch {epoch}/50 - Loss: {loss:.4f} - Train Accuracy: {train_accuracy * 100:.2f}% - Test Accuracy: {test_accuracy * 100:.2f}%")


print("Báo cáo phân loại cuối cùng cho tập huấn luyện:")
print(classification_report(y_train, y_train_pred, target_names=LOAI))
# In ra báo cáo phân loại cho MLP
print("Báo cáo phân loại cho mô hình MLP:")
print(classification_report(y_test, y_pred, target_names=LOAI))

# Load dữ liệu cho RNN
du_lieu, nhan = load_data(THU_MUC, is_mlp=True)
scaler = StandardScaler()
du_lieu = scaler.fit_transform(du_lieu)  # Chuẩn hóa dữ liệu

# Mã hóa nhãn bằng LabelEncoder cho RNN
label_encoder_RNN = LabelEncoder()
nhan = label_encoder_RNN.fit_transform(nhan)

# Chia tập dữ liệu thành tập huấn luyện và kiểm tra
X_train_RNN, X_test_RNN, y_train_RNN, y_test_RNN = train_test_split(
    du_lieu, nhan, test_size=0.2, stratify=nhan, random_state=math.floor(time.time())
)

# Chuyển đổi thành chuỗi (sequence) cho RNN
X_train_RNN = X_train_RNN.reshape((X_train_RNN.shape[0], 1, X_train_RNN.shape[1]))  # 1 timestep
X_test_RNN = X_test_RNN.reshape((X_test_RNN.shape[0], 1, X_test_RNN.shape[1]))

# Xây dựng mô hình RNN
RNN_model = Sequential()
RNN_model.add(LSTM(128, input_shape=(X_train_RNN.shape[1], X_train_RNN.shape[2]), return_sequences=False))
RNN_model.add(Dropout(0.5))
RNN_model.add(Dense(64, activation='relu'))
RNN_model.add(Dropout(0.5))
RNN_model.add(Dense(len(LOAI), activation='softmax'))  # Sử dụng softmax cho bài toán phân loại nhiều lớp
RNN_model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Huấn luyện mô hình RNN
print("Đang huấn luyện mô hình...")
RNN_history = RNN_model.fit(
    X_train_RNN, y_train_RNN, 
    epochs=50, batch_size=32, 
    validation_data=(X_test_RNN, y_test_RNN)
)

# Đánh giá trên tập huấn luyện
y_train_pred_RNN = np.argmax(RNN_model.predict(X_train_RNN), axis=1)
train_accuracy = accuracy_score(y_train_RNN, y_train_pred_RNN)
print(f"Độ chính xác trên tập huấn luyện: {train_accuracy * 100:.2f}%")

# Báo cáo phân loại trên tập huấn luyện
print("Báo cáo phân loại trên tập huấn luyện:\n")
print(classification_report(y_train_RNN, y_train_pred_RNN, target_names=label_encoder_RNN.classes_))

# Đánh giá trên tập kiểm tra
y_test_pred_RNN = np.argmax(RNN_model.predict(X_test_RNN), axis=1)
test_accuracy = accuracy_score(y_test_RNN, y_test_pred_RNN)
print(f"Độ chính xác trên tập kiểm tra: {test_accuracy * 100:.2f}%")

# Báo cáo phân loại trên tập kiểm tra
print("Báo cáo phân loại trên tập kiểm tra:\n")
print(classification_report(y_test_RNN, y_test_pred_RNN, target_names=label_encoder_RNN.classes_))

# So sánh mô hình trên tập huấn luyện và kiểm tra
print("\nSo sánh mô hình trên tập huấn luyện và kiểm tra:")
print(f"- Độ chính xác trên tập huấn luyện: {train_accuracy * 100:.2f}%")
print(f"- Độ chính xác trên tập kiểm tra: {test_accuracy * 100:.2f}%")

# Đồ thị Accuracy chung cho CNN, MLP và RNN
plt.figure(figsize=(10, 6))

# Đào tạo CNN, MLP và RNN
plt.plot(range(1, 51), cnn_history.history['accuracy'], label="CNN đào tạo", marker='o')
plt.plot(range(1, 51), train_accuracies, label="NN đào tạo", marker='x')
plt.plot(range(1, 51), RNN_history.history['accuracy'], label="RNN đào tạo", marker='^')

# Kiểm tra CNN, MLP và RNN
plt.plot(range(1, 51), cnn_history.history['val_accuracy'], label="CNN kiểm tra", marker='s')
plt.plot(range(1, 51), test_accuracies, label="NN kiểm tra", marker='D')
plt.plot(range(1, 51), RNN_history.history['val_accuracy'], label="RNN kiểm tra", marker='p')

# Thêm tiêu đề, nhãn và chú thích
plt.title("Độ chính xác qua các bước đào tạo: CNN, NN và RNN")
plt.xlabel("Số bước đào tạo (epochs)")
plt.ylabel("Độ chính xác (accuracy)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()