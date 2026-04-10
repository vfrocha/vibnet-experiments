# VibNet Experiments

Repositório contendo os experimentos de validação cruzada e Transfer Learning (Domain Adaptation) para o projeto VibNet, utilizando arquiteturas DenseNet-121 e ResNet em dados de vibração de rolamentos.

## 🗂️ Estrutura do Repositório
* `experiments/`: Scripts de treinamento e avaliação separados por arquitetura.
* `results/`: Relatórios de métricas (Accuracy, F1-Score, AUC) gerados pelos experimentos.
* `weights/`: Pesos pré-treinados (`.pth`) gerados na Etapa 1 (Source Domain).

## ⚙️ Como Instalar
Clone o repositório e instale as dependências:
```bash
git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO
pip install -r requirements.txt
```

## ⚙️ Instalação e Requisitos
Clone o repositório:

```Bash
git clone [https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git](https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git)
cd SEU_REPOSITORIO
```
Instale as dependências:
Recomenda-se o uso de um ambiente virtual (venv ou conda).

```Bash
pip install -r requirements.txt
Prepare os dados:
Garanta que a pasta dataset_final/ esteja na raiz com a estrutura esperada pelos carregadores de dados (VibDataset).
```

##  Como Executar os Experimentos
Os scripts executam automaticamente duas etapas fundamentais:

* Etapa Fonte (Source): Pré-treinamento do backbone em domínios correlatos.
* Etapa Alvo (Target): Fine-tuning e teste no domínio alvo usando a técnica Leave-One-Condition-Out (LODO-CV).
* Para rodar o experimento principal com DenseNet-121 na base CWRU Unificada:

```Bash
python experiments/densenet/cwru_unified_exp_v2.py
```

Estratégias Comparadas:
O script avalia e compara quatro tipos de inicialização na mesma rodada:
* Scratch: Pesos aleatórios.
* ImageNet: Pesos pré-treinados do dataset ImageNet.
* VibNet\_from\_Scratch: Pré-treino em vibração iniciando de pesos aleatórios.
* VibNet\_from\_ImageNet: Pré-treino em vibração iniciando de pesos da ImageNet.

## 📈 Métricas de Avaliação
Para garantir robustez estatística, o sistema reporta a média e o desvio padrão de:

* Accuracy (Acurácia simples).
* Balanced Accuracy (Acurácia balanceada para classes desiguais).
* Macro F1-Score (Média harmônica entre precisão e recall).
* Macro AUC (Área sob a curva ROC calculada via Softmax).
