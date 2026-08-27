# Otimização de ROI e Vendas com PySpark & Gemini LLM

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Carrega a chave do .env
load_dotenv()

# Inicializa o cliente do Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Inicializando sessão Spark
spark = SparkSession.builder.appName("ProjetoMarketingLLM").getOrCreate()

# Caminho do arquivo CSV
csv_file_path = "/opt/spark/dados/ecommerce_marketing-llm-analise/dataset_ecommerce_marketing.csv"


# Função para carregar os dados
def carrega_dados(spark_session, file_path):
  # inferSchema=True garante que números e datas sejam tipados corretamente
  dados = spark_session.read.csv(file_path, header=True, inferSchema=True)
  print(f"\nTotal de registros carregados: {dados.count()}\n")
  return dados


# Carrega os dados
df_spark = carrega_dados(spark, csv_file_path)

# Total de investimento por categoria
df_spark.groupBy("categoria").agg(
    F.sum("investimento_mkt").alias("total_investimento")
).show()

# Média de investimento por canal
df_spark.groupBy("canal_marketing").agg(
    F.avg("investimento_mkt").alias("media_investimento")
).show()


# Função para coletar estatísticas e preparar o prompt para o mês de março
def prepara_dados_prompt(df):
  # Extrai o mês da coluna de data e filtra para o mês de março (3)
  df_marco = df.withColumn("mes", F.month("data")).filter(F.col("mes") == 3)

  # Calcula a média de Investimento, Desconto e Receita por categoria e canal
  stats_df = df_marco.groupBy("categoria", "canal_marketing").agg(
      F.avg("investimento_mkt").alias("media_investimento"),
      F.avg("desconto_medio_pct").alias("media_desconto"),
      F.avg("receita_vendas").alias("media_receita"),
  ).collect()

  # Formatando os dados coletados em uma string para o prompt
  dados_formatados = "Estatísticas de vendas para o mês de Março:\n"
  for row in stats_df:
    dados_formatados += (
        f"Categoria: {row['categoria']}, Canal Marketing:"
        f" {row['canal_marketing']}, Média Investimento:"
        f" R$ {row['media_investimento']:.2f}, Média Desconto:"
        f" {row['media_desconto']:.2f}%, Média Receita: R$"
        f" {row['media_receita']:.2f}\n"
    )

  
  return dados_formatados


# Extraindo e formatando dados para o prompt
dados_para_prompt = prepara_dados_prompt(df_spark)


# Função para gerar uma pergunta sobre os dados e obter insights do LLM
def analisa_dados_com_llm(pergunta_usuario, contexto_dados):
  prompt = f"""
    Você é um especialista sênior em Marketing Analytics, E-commerce e Otimização de Performance (Growth/ROAS).
    
    Analise atentamente o resumo estatístico consolidado a seguir:
    ----------------------------------------
    {contexto_dados}
    ----------------------------------------
    
    Com base nesses dados, responda à pergunta do usuário:
    "{pergunta_usuario}"
    
    Diretrizes para a resposta:
    1. Seja analítico, objetivo e direto ao ponto.
    2. Identifique os canais mais e menos eficientes (relação investimento vs. receita gerada e elasticidade de desconto).
    3. Forneça recomendações práticas para realocação de orçamento e ajustes estratégicos de precificação/promoção.
    """

  # Chamada usando o SDK oficial do Gemini
  config = types.GenerateContentConfig(
    temperature=0.2,
    function_calling_config=types.FunctionCallingConfig(
        mode=types.FunctionCallingMode.NONE
    ),
)

response = client.models.generate_content(
    model="gemini-3.5-flash-lite", contents=prompt, config=config
)

  return response.text


# Pergunta sobre os dados
pergunta = (
    "Considerando os dados consolidados, quais canais de marketing e faixas de"
    " desconto apresentaram o melhor retorno sobre investimento (ROI/ROAS) para"
    " as categorias principais, e onde devemos cortar ou aumentar verba no"
    " próximo ciclo?"
)

# Obtém a resposta a partir do LLM
resposta = analisa_dados_com_llm(pergunta, dados_para_prompt)
print("\n--- Relatório Gerado pelo Gemini ---")
print(resposta)

# Finaliza a sessão Spark
spark.stop()

#Fim