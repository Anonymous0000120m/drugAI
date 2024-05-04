import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential

class LymphomaDataGenerator:
    def __init__(self, num_samples=1000, num_features=10, data_path='lymphoma_data.csv'):
        self.num_samples = num_samples
        self.num_features = num_features
        self.data_path = data_path

    def generate_real_data(self):
        # 生成真实淋巴瘤数据
        features = np.random.rand(self.num_samples, self.num_features)
        labels = np.ones((self.num_samples, 1))  # 表示正例
        real_data = np.concatenate((features, labels), axis=1)
        return real_data

    def generate_fake_data(self):
        # 使用 GAN 模型生成假的淋巴瘤数据
        model = self.build_generator()
        noise = np.random.rand(self.num_samples, self.num_features)
        fake_data = model.predict(noise)
        labels = np.zeros((self.num_samples, 1))  # 表示负例
        fake_data_with_labels = np.concatenate((fake_data, labels), axis=1)
        return fake_data_with_labels

    def build_generator(self):
        # 构建生成器模型
        model = Sequential([
            Dense(128, input_dim=self.num_features),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(64),
            LeakyReLU(alpha=0.2),
            BatchNormalization(),
            Dense(self.num_features, activation='tanh')
        ])
        return model

    def save_to_csv(self, data):
        # 将数据保存为 CSV 文件
        column_names = [f"feature_{i}" for i in range(self.num_features)] + ['label']
        df = pd.DataFrame(data, columns=column_names)
        df.to_csv(self.data_path, index=False)
        print(f"Data saved to {self.data_path}")

# 创建淋巴瘤数据生成器实例
generator = LymphomaDataGenerator(num_samples=1500, num_features=8, data_path='lymphoma_data.csv')

# 生成真实和假的淋巴瘤数据，并保存为 CSV 文件
real_data = generator.generate_real_data()
fake_data = generator.generate_fake_data()
all_data = np.concatenate((real_data, fake_data), axis=0)
generator.save_to_csv(all_data)
