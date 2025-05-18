# 🧠 Colorização de Rostos com GANs (UTKFace)

Projeto final da disciplina de **Computação Gráfica**, cujo objetivo é aplicar **Redes Geradoras Adversariais (GANs)** para colorir automaticamente imagens em preto e branco de rostos humanos, utilizando o dataset **UTKFace**.

---

## 🎯 Objetivo

Transformar imagens P&B (preto e branco) de rostos humanos em imagens coloridas realistas utilizando aprendizado profundo com **PyTorch**.

---

## 📂 Estrutura do Projeto

📦 gan-colorizacao-utkface
┣ 📁 resultados_gan_color/
┃ ┣ gan_colorida.png
┃ ┣ gan_filtrada.png
┃ ┗ loss_gan.png
┣ utk.zip (dataset)
┣ colorizacao_gan.py
┗ README.md


---

## 🧠 Técnicas Utilizadas

- **GAN (Generative Adversarial Network)**
- **Generator baseado em UNet**
- **Discriminador CNN**
- **Funções de perda**:
  - `BCEWithLogitsLoss` (perda adversarial)
  - `L1Loss` (reconstrução da imagem)
- **Filtros adicionais com OpenCV**:
  - `cv2.detailEnhance`
  - `cv2.bilateralFilter`

---

## 📊 Hiperparâmetros

| Parâmetro       | Valor         |
|------------------|---------------|
| Épocas           | 120           |
| Tamanho da imagem| 128 x 128     |
| Batch size       | 16            |
| Otimizador       | Adam          |
| Dataset          | UTKFace (10k imagens) |
| Dispositivo      | Google Colab (GPU T4) |

---

## 📦 Dataset

- **Nome**: [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- Contém rostos com diversidade de etnia, idade e sexo
- Utilizado para treinar a GAN
- Extraído para: `/content/UTKFace/crop_part1`

---

## 🚀 Como Executar

1. Clone este repositório:
   
git clone https://github.com/seu-usuario/gan-colorizacao-utkface.git
cd gan-colorizacao-utkface

2. Instale as dependências:
pip install torch torchvision opencv-python matplotlib pillow

3. Coloque o arquivo utk.zip na raiz ou baixe do Kaggle.
   
4. Execute o script principal:
python colorizacao_gan.py

📉 Curva de Perda

O gerador aprende gradualmente a colorir de forma mais realista
O discriminador estabiliza ao longo do treinamento

⚠️ Limitações

- Imagens com resolução baixa (128x128)
- Modelo não usa Perceptual Loss (VGG)
- Não possui Data Augmentation
- Focado exclusivamente em rostos humanos (não generaliza para outros domínios)

💡 Melhorias Futuras

- Substituir L1Loss por Perceptual Loss (VGG16)
- Aumentar resolução de saída (ex: 256x256)
- Incorporar técnicas de style transfer
- Aplicar o modelo em vídeos ou fotos históricas

👨‍💻 Autor
Lucca Carnaúba Peixoto Rosário
Disciplina: Computação Gráfica
Curso: Sistemas de Informação
Ano: 2025
