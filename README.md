# 🚛 Monitoramento do Estado de Alerta do Motorista V1

### Sistema baseado em YOLOv8 e Google MediaPipe para monitoramento de distrações ao volante em tempo real.



## 📖 Sobre o Projeto

A distração ao volante está entre as principais causas de acidentes de trânsito, especialmente no transporte de cargas, onde longas jornadas aumentam a ocorrência de fadiga e sonolência.

Este projeto apresenta uma solução baseada em **Aprendizado de Máquina** e **Visão Computacional** para monitorar, em tempo real, distrações relacionadas ao uso do telefone celular e sinais de sonolência durante a condução de veículos.

A proposta integra os modelos **YOLOv8** e **Google MediaPipe Face Mesh** em uma única aplicação, permitindo detectar diferentes tipos de distração simultaneamente e emitir alertas sonoros e visuais sempre que situações de risco forem identificadas.

---

# 🎯 Objetivo

Desenvolver um sistema capaz de monitorar o estado de alerta do motorista por meio de técnicas de Visão Computacional, identificando automaticamente distrações relacionadas ao uso do celular e à sonolência, contribuindo para aumentar a segurança durante a condução.

---

# 🧠 Metodologia

A abordagem metodológica inclui o uso de modelos pré-treinados, que são redes neurais já treinadas em grandes conjuntos de dados.

A solução foi desenvolvida integrando duas tecnologias de Inteligência Artificial:

- **YOLOv8** para detecção do telefone celular;
- **MediaPipe** para monitoramento da região dos olhos e identificação de sinais de sonolência.


---

# ✨ Diferenciais da Abordagem

Diferentemente de abordagens baseadas no treinamento de modelos do zero, este trabalho adotou a utilização de **modelos pré-treinados**, estratégia amplamente utilizada em aplicações de Visão Computacional devido às seguintes vantagens:

- redução significativa do tempo de desenvolvimento;
- menor necessidade de recursos computacionais para treinamento;
- utilização de modelos previamente treinados e amplamente validados pela comunidade científica;
- maior facilidade de implementação e integração entre diferentes tecnologias;
- possibilidade de concentrar os esforços no desenvolvimento da solução e na integração dos módulos de detecção.

Essa abordagem tornou possível construir uma aplicação funcional capaz de realizar monitoramento em tempo real utilizando tecnologias consolidadas, reduzindo a complexidade do desenvolvimento sem comprometer a eficiência do sistema.

Outro diferencial deste trabalho foi a **integração do YOLOv8 e do Google MediaPipe em uma única solução**, permitindo detectar simultaneamente duas das principais causas de distração ao volante.

Além disso, enquanto diversos trabalhos relacionados concentram-se apenas na identificação das distrações, esta proposta incorpora **alertas sonoros e visuais em tempo real**, permitindo que o sistema atue de forma preventiva ao alertar imediatamente o motorista quando um comportamento de risco é identificado.

---

# 📊 Avaliação

Foram avaliadas diferentes combinações entre os modelos do YOLOv8 (Nano, Small, Medium, Large e Extra Large) juntamente com o MediaPipe Face Mesh.

A avaliação considerou:

- Precisão média (mAP);
- Taxa de processamento (FPS);
- Execução em CPU;
- Execução em GPU.

Os resultados experimentais demonstraram que o modelo YOLOv8m alcançou um mAP@50 de 0,50, atingindo 20,5 FPS em GPU e 5 FPS em CPU.

---

# 🚗 Funcionamento

O sistema executa continuamente o seguinte fluxo de processamento:

```text
                                 Captura de frames de vídeo
                                              │
                                              ▼
                                      pré-processamento
                                              │
    Detecção do telefone celular ────► processamento ◄──── Monitoramento da sonolência
                                              │
                                              ▼
                                      Módulo de decisão
                                              │
                                              ▼
                                         resultados
                                              │
                                              ▼
                              Emissão de alertas sonoros e visuais
```

---

# 📊 Principais Contribuições

- Integração entre YOLOv8 e Google MediaPipe Face Mesh.
- Monitoramento simultâneo do uso do telefone celular e da sonolência.
- Implementação de alertas sonoros e visuais em tempo real.
- Avaliação de desempenho em CPU e GPU.
- Validação da solução em ambiente veicular real.
- Estrutura modular para integração de novos recursos.

---

# 💻 Tecnologias Utilizadas

- Python
- OpenCV
- YOLOv8
- Google MediaPipe Face Mesh
- PyTorch
- PyCharm

---

# 🚛 Aplicações

A solução pode ser utilizada em diferentes cenários, tais como:

- Monitoramento de motoristas profissionais;
- Transporte de cargas;
- Sistemas de apoio à condução;
- Pesquisa em Visão Computacional;
- Estudos relacionados à prevenção de acidentes de trânsito.

---

# 📁 Estrutura do Projeto

```text
Driver-Alert-Monitoring/

├── images/
├── src/
├── weights/
├── README.md
└── requirements.txt
```

