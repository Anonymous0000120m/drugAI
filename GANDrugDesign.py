from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import re

class GANDrugDesign:
    def __init__(self, num_features, noise_dim):
        self.num_features = num_features
        self.noise_dim = noise_dim
        self.generator = self.build_generator()

    def build_generator(self):
        input_layer = Input(shape=(self.noise_dim,))
        x = Dense(128, activation='relu')(input_layer)
        output_layer = Dense(self.num_features, activation='tanh')(x)
        return Model(input_layer, output_layer)

    def generate_samples(self, num_samples):
        noise = np.random.normal(0, 1, size=(num_samples, self.noise_dim))
        return self.generator.predict(noise)

class ChemicalStructureDesign:
    def __init__(self, num_samples, num_features, noise_dim):
        self.num_samples = num_samples
        self.num_features = num_features
        self.noise_dim = noise_dim
        self.gan_model = GANDrugDesign(num_features, noise_dim)

    def design_chemical_structures(self, samples):
        designed_structures = []

        for sample in samples:
            for s in sample:
                # 使用正则表达式将小数转换为整数，并将其作为氢原子数添加到原子标记的后面
                atom_label = 'C' + str(int(s * 1000000))  # 将小数扩大到整数级别
                mol = Chem.MolFromSmiles(atom_label)
                if mol is not None:
                    # 添加羟基官能团到SMILES字符串中
                    modified_smiles = atom_label + 'O'
                
                    # 添加一个环到SMILES字符串中（这里添加一个五元环）
                    modified_smiles += 'C1CCCC1'

                    # 添加一个双键到SMILES字符串中（这里添加一个双键到第一个碳原子）
                    modified_smiles = modified_smiles[:1] + '=C' + modified_smiles[1:]

                    # 将SMILES字符串中的环闭合索引更新为唯一的值
                    modified_smiles = re.sub(r'(?<!\d)([0-9]+)(?!\d)', lambda x: str(int(x.group(0)) + 1000), modified_smiles)

                    designed_structures.append(modified_smiles)

        return designed_structures



    def extract_features(self, structures):
        # 提取化学结构的特征
        features = []

        for structure in structures:
            mol = Chem.MolFromSmiles(structure)
            if mol is not None:
                # 提取分子量等化学特征
                molecular_weight = Descriptors.MolWt(mol)
                features.append(molecular_weight)

        return features

    def save_structure_data_to_csv(self, structures, features, output_file):
        # 保存化学结构数据到CSV文件
        df = pd.DataFrame({'SMILES': structures, 'MolecularWeight': features})
        df.to_csv(output_file, index=False)

    def train_gan(self, epochs):
        # 训练GAN模型
        # 这里需要根据实际情况调用GAN模型的训练方法
        pass

    def train_and_generate_structures(self, epochs, output_file):
        # 训练 GAN 模型
        self.train_gan(epochs)

        # 生成样本并设计化学结构
        samples = self.gan_model.generate_samples(self.num_samples)
        designed_structures = self.design_chemical_structures(samples)

        # 提取化学结构特征
        structure_features = self.extract_features(designed_structures)

        # 保存化学结构数据到CSV文件
        self.save_structure_data_to_csv(designed_structures, structure_features, output_file)

# 使用示例
num_samples = 500
num_features = 10
noise_dim = 100

# 创建 ChemicalStructureDesign 实例
chemical_structure_design = ChemicalStructureDesign(num_samples, num_features, noise_dim)

# 训练并生成化学结构数据
chemical_structure_design.train_and_generate_structures(epochs=1000, output_file='designed_structures.csv')
