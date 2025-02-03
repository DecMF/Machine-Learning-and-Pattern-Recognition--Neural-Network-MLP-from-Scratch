# Implementação de MLP From Scratch

Este projeto foi desenvolvido como parte da disciplina **MO444**, ministrada pela professora **Sandra**, e contou com a colaboração de **Rafael Simionato(IC/Unicamp)**. O objetivo é explorar redes neurais e desenvolver um modelo eficiente para a tarefa de classificação, sem o uso direto de pacotes de machine learning como TensorFlow e Scikit-Learn.

## Objetivo do Projeto
Construir uma **rede neural multicamada (MLP - Multi-Layer Perceptron)** para classificar imagens do conjunto de dados **OrganCMNIST**, implementando do zero os principais componentes necessários para o treinamento da rede. O objetivo é evitar *overfitting* e comparar diferentes arquiteturas e técnicas de otimização.

## 🗂 Base de Dados
O **OrganCMNIST** é um dos conjuntos de dados disponíveis no [MedMNIST](https://medmnist.com/). Ele contém imagens 2D extraídas de tomografias computadorizadas 3D, focando na classificação de **11 órgãos diferentes**. As imagens são em escala de cinza e foram padronizadas para **1x28x28 pixels**.

O conjunto de dados é dividido da seguinte forma:
- **12.975** exemplos de treinamento;
- **2.392** exemplos de validação;
- **8.216** exemplos de teste.

Cada imagem pertence a uma das 11 classes de órgãos:

| ID | Órgão | # Imagens |
|:---:|:---|---:|
| 0 | Bexiga | 2.167 |
| 1 | Fêmur Esquerdo | 1.152 |
| 2 | Fêmur Direito | 1.112 |
| 3 | Coração | 1.223 |
| 4 | Rim Esquerdo | 1.947 |
| 5 | Rim Direito | 2.062 |
| 6 | Fígado | 5.250 |
| 7 | Pulmão Esquerdo | 1.898 |
| 8 | Pulmão Direito | 1.931 |
| 9 | Pâncreas | 2.102 |
| 10 | Baço | 2.739 |

A base de dados foi utilizada conforme a divisão padrão de treino, validação e teste, respeitando a avaliação via **balanced accuracy**.

## Implementação da Rede Neural

### **Forward Propagation**
Foi implementada iterando pelas camadas da rede e computando as ativações de cada neurônio com base na seguinte equação:

\[
Z^{(l)} = W^{(l)} A^{(l-1)} + b^{(l)}
\]
\[
A^{(l)} = f(Z^{(l)})
\]

Onde:
- \( W^{(l)} \) são os pesos da camada \( l \);
- \( b^{(l)} \) é o viés da camada \( l \);
- \( A^{(l-1)} \) são as ativações da camada anterior;
- \( f \) é a função de ativação (ReLU nas camadas ocultas e Softmax na camada de saída).

### **Backward Propagation**
O backrpopagation foi implementada para calcular os gradientes dos pesos e vieses a fim de minimizar a função de perda **Cross-Entropy**. O gradiente da função de perda em relação à saída da última camada foi calculado como:

\[
\frac{\partial J}{\partial Z^{(L)}} = A^{(L)} - Y
\]

Para as camadas intermediárias, utilizamos a regra da cadeia para calcular os gradientes:

\[
\frac{\partial J}{\partial W^{(l)}} = \frac{\partial J}{\partial Z^{(l)}} A^{(l-1)T}
\]
\[
\frac{\partial J}{\partial b^{(l)}} = \sum \frac{\partial J}{\partial Z^{(l)}}
\]
\[
\frac{\partial J}{\partial A^{(l-1)}} = W^{(l)T} \frac{\partial J}{\partial Z^{(l)}}
\]

Dessa forma, iteramos de trás para frente ajustando os pesos de acordo com a regra do gradiente descendente:

\[
W^{(l)} := W^{(l)} - \alpha \frac{\partial J}{\partial W^{(l)}}
\]
\[
b^{(l)} := b^{(l)} - \alpha \frac{\partial J}{\partial b^{(l)}}
\]

Onde \( \alpha \) é a taxa de aprendizado.

### **Otimizadores Implementados**
Foram desenvolvidos e testados diferentes métodos de otimização para ajustar os pesos da rede:
- **Stochastic Gradient Descent (SGD)**;
- **Adam**.

## Experimentos e Resultados

### **Arquitetura Inicial e Impacto da Regularização**
Uma rede neural inicial foi projetada com **duas camadas ocultas** contendo **512 e 256 neurônios**, respectivamente. Algumas observações:
- Sem *batch normalization*, houve problemas de instabilidade do gradiente.
- O *early-stopping* interrompeu o treinamento após **199 épocas**, evitando *overfitting*.
- **Balanced accuracy final:** 66,9% (treino) e 65,1% (validação).

### **Comparação de Inicializações: He vs. Xavier**
Testamos as inicializações de pesos **He** e **Glorot/Xavier**:
- **He:** Resultou em convergência muito lenta, obtendo 55,5% de acurácia balanceada.
- **Glorot/Xavier:** Melhor desempenho, alcançando **68,5% (treino) e 67,4% (validação)**.

### **Impacto da Função de Ativação**
Foi testada a substituição da ativação **ReLU** por **Sigmoid**:
- A ativação **Sigmoid** reduziu significativamente a performance, com acurácias próximas de **51%**.

## Conclusões
Os principais aprendizados deste projeto incluem:
- **A inicialização dos pesos** impacta significativamente a convergência da rede.
- **ReLU** foi a melhor função de ativação para camadas ocultas.
- **Glorot/Xavier** teve melhor performance que **He**.
- **SGD** superou **Adam** para essa configuração.
- **A profundidade da rede influencia a performance**, mas muitas camadas podem prejudicar o desempenho.

## Próximos Passos
- Explorar arquiteturas **mais profundas** com normalização de batch;
- Testar outras funções de ativação, como **Leaky ReLU**;
- Implementar **dropout** para regularização;
- Experimentar outros otimizadores como **RMSProp**.



