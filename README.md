# Projeto de Detecção Facial com YOLO e Mediapipe

Este projeto utiliza YOLO e Mediapipe para realizar a detecção facial em tempo real através da webcam.

## Estrutura do Projeto

- `main.py`: Arquivo principal que executa a detecção facial.
- `utils.py`: Contém funções auxiliares para configuração do vídeo, face mesh e YOLO.
- `test_utils.py`: Testes unitários para as funções em `utils.py`.
- `.env`: Arquivo para variáveis de ambiente.
- `requirements.txt`: Lista de dependências do projeto.
- `.github/workflows/python-app.yml`: Configuração do workflow do GitHub Actions para pipeline de teste unitário.

## Instalação

1. Clone o repositório:
    ```sh
    $ git clone https://github.com/lsmgoes/driving_distraction_detector.git
    $ cd driving_distraction_detector
    ```

2. Crie um ambiente virtual e ative-o:
    ```sh
    $ python -m venv venv
    $ venv\Scripts\activate  # No Linux use `source venv/bin/activate`
    ```

3. Instale as dependências:
    ```sh
    $ pip install -r requirements.txt
    ```

4. Configure as variáveis de ambiente no arquivo `.env` (se necessário).

## Uso

Para executar o projeto, basta rodar o arquivo `main.py`:
```sh
$ python main.py
