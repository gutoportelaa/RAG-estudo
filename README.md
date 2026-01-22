# RAG com Gemini e LangChain (Free Tier Friendly)

Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation)** utilizando a API do Google Gemini e a biblioteca LangChain. O objetivo é permitir que uma IA responda perguntas baseadas em documentos de texto privados (PDFs ou TXTs), superando as limitações de conhecimento de treino do modelo.

<img width="2816" height="1536" alt="Fluxograma-RAG" src="https://github.com/user-attachments/assets/bce94b74-6e7f-4e95-8751-32956043c91d" />

##  Diferenciais deste Projeto
- **Custo Zero:** Configurado para rodar inteiramente no plano gratuito da API do Google Gemini.
- **Robustez:** Implementação de lógica de **Retry** (tentativa) e **Batching** (lotes) na criação do banco vetorial para evitar erros de *Rate Limit* (Erro 429) comuns em APIs gratuitas.
- **Verificação de Relevância:** O bot só responde se encontrar contexto suficiente no banco de dados (score de similaridade).

##  Tecnologias Utilizadas
- **Python 3.12**
- **LangChain:** Framework para orquestração de LLMs.
- **Google Generative AI (Gemini):**
  - `models/text-embedding-004`: Para vetorização do texto.
  - `gemini-1.5-flash` (ou 2.5): Para geração de respostas (Chat).
- **ChromaDB:** Banco de dados vetorial local.
- **Dotenv:** Gerenciamento de variáveis de ambiente.

##  Como Rodar

### 1. Pré-requisitos
Tenha o Python instalado e uma API Key do Google AI Studio.

### 2. Instalação
Clone o repositório e instale as dependências:
```bash
pip install langchain langchain-community langchain-google-genai langchain-chroma python-dotenv
```
### 3. Configuração
Para garantir que o projeto rode com segurança e conecte-se aos serviços do Google, siga estes passos:

1.  **Obtenha a API Key:**
    * Acesse o [Google AI Studio](https://aistudio.google.com/).
    * Crie uma nova chave de API (gratuita).

2.  **Configure o Ambiente:**
    * Crie um arquivo chamado `.env` na raiz do projeto (onde estão os arquivos `.py`).
    * Adicione a seguinte linha dentro dele:
    ```env
    GOOGLE_API_KEY="sua_chave_secreta_aqui"
    ```
    * *Nota:* O arquivo `.env` nunca deve ser compartilhado ou subido para o GitHub.

### 4. Estrutura e Funcionamento do Código
O projeto opera em dois estágios distintos: **Ingestão** (preparação dos dados) e **Inferência** (uso do chat).

#### I. Criação do Banco de Dados (`criar_db.py`)
Este script transforma documentos de texto em uma base de dados pesquisável.

* **`carregar_documentos()`**: Utiliza o `DirectoryLoader` para ler todos os arquivos `.txt` da pasta `base`. É o primeiro passo de entrada de dados.
* **`dividir_documentos()`**: Aplica o `RecursiveCharacterTextSplitter`. Divide textos longos em pedaços (*chunks*) de 1000 caracteres. Isso é crucial para respeitar o limite de contexto da IA e melhorar a precisão da busca.
* **`vetorizar_chunks()`**:
    * Converte os textos em vetores numéricos usando `GoogleGenerativeAIEmbeddings`.
    * **Estratégia Anti-Bloqueio:** Implementa um loop que envia dados em **lotes de 5 chunks** e realiza uma pausa (`time.sleep`) de 2 segundos entre eles. Isso contorna o erro `429 Resource Exhausted` comum no plano gratuito.
    * Salva os dados na pasta `destino` usando o banco vetorial **ChromaDB**.



#### II. O Chatbot (`main.py`)
Este script realiza a mágica do RAG (Retrieval-Augmented Generation).

* **`similarity_search_with_relevance_scores()`**: Ao receber uma pergunta, o sistema busca no ChromaDB os 3 trechos de texto mais similares semanticamente.
* **Filtro de Relevância**: O código verifica a pontuação (score) da busca. Se a similaridade for muito baixa (ex: < 0.2), o bot responde que "não sabe", evitando invenções (*alucinações*).
* **`ChatPromptTemplate`**: Monta um comando para a IA contendo a pergunta do usuário + os textos recuperados do banco.
* **`ChatGoogleGenerativeAI`**: Envia o prompt montado para o modelo **Gemini Flash**, que gera a resposta final baseada estritamente nos documentos fornecidos.



### 5. Execução
Como rodar o projeto na sua máquina:

1.  **Prepare os Dados:**
    Coloque seus arquivos de texto (ex: `noticia.txt`) dentro da pasta `base`.

2.  **Gere o Banco de Dados:**
    Execute o comando abaixo. Aguarde o processamento dos lotes.
    ```bash
    python criar_db.py
    ```
    *Saída esperada:* "Banco de dados criado!"

3.  **Inicie o Chat:**
    Execute o script principal e faça sua pergunta.
    ```bash
    python main.py
    ```

### 6. Conceitos Chave Aprendidos
* **Embeddings:** Representação matemática (vetorial) de textos. Textos com significados parecidos geram números parecidos.
* **Vector Store (Chroma):** Banco de dados especializado em armazenar e buscar esses vetores rapidamente.
* **Rate Limiting:** A importância de controlar a velocidade das requisições (batching/sleep) ao trabalhar com APIs públicas e gratuitas.
* **Context Window:** A necessidade de dividir textos grandes (chunking) para caber na memória de curto prazo da IA.
