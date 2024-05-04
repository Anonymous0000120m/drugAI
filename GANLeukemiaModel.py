import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

class GANLeukemiaModel:
    def __init__(self, num_features, num_samples, noise_dim):
        self.num_features = num_features  # 特征数
        self.num_samples = num_samples    # 样本数
        self.noise_dim = noise_dim        # 噪声维度

        # 生成器模型
        self.generator = self.build_generator()

        # 判别器模型
        self.discriminator = self.build_discriminator()

        # 优化器
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def build_generator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.noise_dim,)),
            tf.keras.layers.Dense(self.num_features, activation='tanh')
        ])
        return model

    def build_discriminator(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.num_features,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    def generate_fake_samples(self, batch_size):
        noise = tf.random.normal([batch_size, self.noise_dim])
        return self.generator(noise)

    def train_step(self, real_samples):
        batch_size = tf.shape(real_samples)[0]
        
        # 生成假样本
        fake_samples = self.generate_fake_samples(batch_size)

        # 计算判别器的损失
        with tf.GradientTape() as disc_tape:
            real_output = self.discriminator(real_samples)
            fake_output = self.discriminator(fake_samples)
            disc_loss = -tf.reduce_mean(tf.math.log(real_output) + tf.math.log(1. - fake_output))

        # 更新判别器的权重
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # 计算生成器的损失
        with tf.GradientTape() as gen_tape:
            fake_samples = self.generate_fake_samples(batch_size)
            fake_output = self.discriminator(fake_samples)
            gen_loss = -tf.reduce_mean(tf.math.log(fake_output))

        # 更新生成器的权重
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))

        return disc_loss, gen_loss

    def train(self, epochs):
        for epoch in range(epochs):
            for _ in range(self.num_samples // 128):  # 使用小批量训练
                real_samples = np.random.randn(128, self.num_features)  # 使用随机生成的真实样本
                disc_loss, gen_loss = self.train_step(real_samples)

            if epoch % 50 == 0:
                print(f"Epoch {epoch}, Discriminator Loss: {disc_loss}, Generator Loss: {gen_loss}")

    def generate_samples(self, num_samples):
        noise = tf.random.normal([num_samples, self.noise_dim])
        fake_samples = self.generator(noise).numpy()
        return fake_samples

    def save_samples_to_csv(self, samples, output_file):
        df = pd.DataFrame(samples, columns=[f'feature_{i}' for i in range(self.num_features)])
        df.to_csv(output_file, index=False)

# 使用示例
num_features = 10
num_samples = 500
noise_dim = 100

# 创建 GANLeukemiaModel 实例
gan_leukemia_model = GANLeukemiaModel(num_features, num_samples, noise_dim)

# 训练 GAN 模型
gan_leukemia_model.train(epochs=1000)

# 生成样本并保存到 CSV 文件
samples = gan_leukemia_model.generate_samples(num_samples)
gan_leukemia_model.save_samples_to_csv(samples, 'leukemia_samples.csv')

# 绘制曲线图
time = np.linspace(0, 10, num_samples)  # 时间
damage_level = np.random.uniform(low=0, high=1, size=num_samples)  # 破坏程度等级
plt.plot(time, damage_level)
plt.xlabel('Time')
plt.ylabel('Damage Level')
plt.title('Leukemia Impact on Immune System')
plt.show()
