from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings # modelos para comparação U transformação entre textos e vetores
from dotenv import load_dotenv

# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate          #outras versões utilizam apenas langchaini.prompts, mas serve para declarar templates e variáveis ao prompt
from langchain_google_genai import ChatGoogleGenerativeAI    #Os modelos diferem entre embeddings e chat de perguntas e respostas


load_dotenv()
CAMINHO_DB = "destino"

prompt_template = """
Responda a pergunta do usuário:
{pergunta}

com base nessas informações:
{base_de_conhecimento}

Se você não encontrar a resposta para a pergunta nessas informações, responda: 'Não sei te dizer isso, preciso de mais conhecimento :( '
"""

def perguntar():
    pergunta = input("Faça uma pergunta para nosso bot:")

    funcao_embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    db = Chroma(persist_directory=CAMINHO_DB, embedding_function=funcao_embedding)

    resultados = db.similarity_search_with_relevance_scores(pergunta, k=3)
    if len(resultados) == 0 or resultados[0][1] < 0.2:
        print("Não conseguiu encontrar nada relevante o suficiente na base")
        return
    textos_resultado = []
    for resultado in resultados:
        texto = resultado[0].page_content
        textos_resultado.append(texto)
    
    
    base_de_conhecimento = "\n\n-----\n\n".join(textos_resultado)
    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt = prompt.invoke({"pergunta" : pergunta, "base_de_conhecimento": base_de_conhecimento})
    # print(prompt)
    
    modelo = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    texto_resposta = modelo.invoke(prompt)
    print("Resposta da IA:    \n", texto_resposta)
    
perguntar()