# VibNet Experiments

Repositório contendo os experimentos de validação cruzada e Transfer Learning (Domain Adaptation) para o projeto VibNet, utilizando arquiteturas como DenseNet-121 em dados de vibração de rolamentos.

## 🗂️ Estrutura do Repositório

O projeto adota uma arquitetura modular para facilitar a reprodutibilidade e a adição de novos datasets:

* `core/`: Núcleo do sistema. Contém abstrações de dados, modelos, métricas, transformações de sinal e o motor de treinamento principal.
* `scripts/`: Scripts utilitários de infraestrutura (ex: download e pré-processamento inicial dos sinais).
* `experiments/`: Scripts e Jupyter Notebooks que disparam os testes de validação cruzada (LODO-CV) para cada domínio.
* `dataset_final/`: (Gerado localmente) Diretório de armazenamento das imagens transformadas.
* `results/`: Relatórios de métricas (Accuracy, F1-Score, AUC) gerados em `.txt` ou `.csv` a cada execução.
* `weights/`: Pesos pré-treinados (`.pth`) do Source Domain, armazenados para evitar re-treinos desnecessários.

## ⚙️ Instalação e Requisitos

Clone o repositório para sua máquina local ou servidor:

```bash
git clone [https://github.com/vfrocha/vibnet-experiments.git](https://github.com/vfrocha/vibnet-experiments.git)
cd vibnet-experiments
```

Recomenda-se o uso de um ambiente virtual (venv ou conda). Instale as dependências:

```bash
pip install -r requirements.txt
```
🛠️ Pré-processamento (Geração de Imagens)
Antes de iniciar os experimentos de Deep Learning, é necessário converter os sinais brutos em espectrogramas. Este projeto utiliza uma Transformada de Fourier de Curto Prazo (STFT) e operações de detrend adaptadas para gerar imagens RGB padronizadas.

Para processar as bases de dados (CWRU, HUST, PU, UORED) automaticamente a partir do repositório em nuvem, execute:

```bash
python scripts/prepare_datasets.py
```
Nota: Este comando criará as pastas raw_data/ e dataset_final/ automaticamente.

🚀 Como Executar os Experimentos
Os scripts de experimentos executam automaticamente duas etapas fundamentais:

Etapa Fonte (Source): Pré-treinamento do backbone excluindo a base alvo.

Etapa Alvo (Target): Fine-tuning e teste no domínio alvo usando a técnica Leave-One-Condition-Out Cross-Validation (LODO-CV).

Para garantir que o Python reconheça a estrutura modular do projeto, execute os experimentos como módulos a partir da raiz do repositório. Por exemplo, para rodar o experimento no dataset PU Unificado:

```bash
python -m experiments.run_pu_unified_exp
```
(O mesmo padrão se aplica aos outros scripts dentro da pasta experiments/, como CWRU, HUST e UORED).

Estratégias Comparadas
A cada execução, o script avalia quatro tipos de inicialização de pesos:

Scratch: Pesos aleatórios iniciais.
ImageNet: Pesos clássicos pré-treinados no dataset ImageNet.
VibNet_from_Scratch: Pré-treino em vibração (Source Domains) iniciando de pesos aleatórios.
VibNet_from_ImageNet: Pré-treino em vibração (Source Domains) iniciando de pesos da ImageNet.

📈 Métricas de Avaliação
Para garantir robustez estatística em distribuições desbalanceadas, o motor reporta:
Accuracy: Acurácia global simples.
Balanced Accuracy: Acurácia balanceada (dá pesos iguais para as classes).

Macro F1-Score: Média harmônica entre precisão e recall, calculada macroscopicamente.

Macro AUC: Área sob a curva ROC calculada via probabilidades originadas por Softmax.
