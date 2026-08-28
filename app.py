import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 1. CONFIGURAÇÃO DA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MarketIntel: Analytics de Mkt & Gemini LLM",
    page_icon="📊",
    layout="wide",
)

# Estilização CSS leve
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #1e88e5;
        padding: 15px;
        border-radius: 5px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Carrega variáveis locais ou do Streamlit Cloud Secrets
load_dotenv()
gemini_key = os.getenv("GEMINI_API_KEY") or st.secrets.get(
    "GEMINI_API_KEY", None
)


@st.cache_resource
def get_gemini_client(api_key):
  if not api_key:
    return None
  return genai.Client(api_key=api_key)


client = get_gemini_client(gemini_key)


# -----------------------------------------------------------------------------
# 2. CARREGAMENTO DOS DADOS
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
  df = pd.read_csv("dataset_ecommerce_marketing.csv")
  df["data"] = pd.to_datetime(df["data"])
  df["mes_ano"] = df["data"].dt.strftime("%m/%Y")
  df["roas"] = np.where(
      df["investimento_mkt"] > 0,
      df["receita_vendas"] / df["investimento_mkt"],
      0,
  )
  return df


try:
  df = load_data()
except Exception as e:
  st.error(
      "Não foi possível carregar os dados. Verifique se"
      " 'dataset_ecommerce_marketing.csv' está no diretório correto."
  )
  st.stop()

# -----------------------------------------------------------------------------
# 3. BARRA LATERAL (SIDEBAR) - MODELO PADRONIZADO
# -----------------------------------------------------------------------------
with st.sidebar:
  st.title("⚙️ Painel de Controle")
  st.markdown("---")

  # Filtros de Dados
  st.subheader("📅 Período & Filtros")

  meses_disponiveis = sorted(df["data"].dt.to_period("M").unique().astype(str))
  mes_selecionado = st.selectbox(
      "Mês de Referência:",
      options=meses_disponiveis,
      index=len(meses_disponiveis) - 1,
  )

  todas_categorias = sorted(df["categoria"].unique().tolist())
  categorias_selecionadas = st.multiselect(
      "Categorias:",
      options=todas_categorias,
      default=todas_categorias,
  )

  todos_canais = sorted(df["canal_marketing"].unique().tolist())
  canais_selecionados = st.multiselect(
      "Canais de Mídia:",
      options=todos_canais,
      default=todos_canais,
  )

  st.markdown("---")

  # Seção Prática do Projeto
  st.subheader("📌 Sobre o Projeto")
  st.markdown("""
    **Objetivo:** Consolidar métricas de receita, investimento e elasticidade de descontos com PySpark, integrado ao Google Gemini para recomendações táticas de mídia.
    
    **Impacto de Negócio:** Otimização do ROAS, eliminação de desperdício em canais pagos e proteção da margem bruta contra promoções excessivas.
    """)

  st.markdown("---")

  # Seção do Desenvolvedor
  st.subheader("🛠️ Desenvolvedor")
  st.markdown("**Douglas Vittori**")
  st.caption("Cientista de Dados em Formação")

  st.markdown("🔗 **Acesse meus projetos:**")
  st.markdown(
      "[🚀 Meu Portfólio de"
      " Dados](https://douglasvittori-portfolio.lovable.app/)"
  )

# Aplicar filtros
df_filtrado = df[
    (df["data"].dt.to_period("M").astype(str) == mes_selecionado)
    & (df["categoria"].isin(categorias_selecionadas))
    & (df["canal_marketing"].isin(canais_selecionados))
].copy()

if df_filtrado.empty:
  st.warning(
      "Nenhum registro encontrado para os filtros selecionados na barra lateral."
  )
  st.stop()

# -----------------------------------------------------------------------------
# 4. CABEÇALHO EXECUTIVO
# -----------------------------------------------------------------------------
st.title("📊 MarketIntel: Performance & Marketing Analytics")
st.caption(
    "Diagnóstico automatizado de retorno sobre investimento publicitário (ROAS)"
    " e eficiência de campanhas com IA Generativa."
)
st.markdown("---")

# -----------------------------------------------------------------------------
# 5. INDICADORES CHAVE DE DESEMPENHO (KPIs)
# -----------------------------------------------------------------------------
st.subheader(f"📊 Indicadores de Desempenho ({mes_selecionado})")

total_receita = df_filtrado["receita_vendas"].sum()
total_investimento = df_filtrado["investimento_mkt"].sum()
roas_consolidado = (
    total_receita / total_investimento if total_investimento > 0 else 0
)
desconto_medio = df_filtrado["desconto_medio_pct"].mean()

col1, col2, col3, col4 = st.columns(4)

with col1:
  st.metric(
      label="Faturamento Total",
      value=f"R$ {total_receita:,.2f}",
      help="Receita bruta total gerada no período selecionado.",
  )

with col2:
  st.metric(
      label="Investimento em Mídia",
      value=f"R$ {total_investimento:,.2f}",
      help="Total investido em canais de mídia paga.",
  )

with col3:
  st.metric(
      label="ROAS Consolidado",
      value=f"{roas_consolidado:.2f}x",
      delta=(
          f"{(roas_consolidado - 1) * 100:.1f}% Margem"
          if total_investimento > 0
          else None
      ),
      help="Retorno sobre investimento em anúncios (Receita / Investimento).",
  )

with col4:
  st.metric(
      label="Desconto Médio",
      value=f"{desconto_medio:.2f}%",
      help="Taxa média de desconto promocional concedida nas vendas.",
  )

st.markdown("---")

# -----------------------------------------------------------------------------
# 6. VISÃO GRÁFICA EXECUTIVA (PLOTLY)
# -----------------------------------------------------------------------------
st.subheader("Visão Gráfica da Eficiência Comercial")

col_g1, col_g2 = st.columns(2)

# Gráfico 1: Receita por Canal
df_canal = (
    df_filtrado.groupby("canal_marketing", as_index=False)["receita_vendas"]
    .sum()
    .sort_values(by="receita_vendas", ascending=True)
)

fig_canal = go.Figure()
fig_canal.add_trace(
    go.Bar(
        x=df_canal["receita_vendas"],
        y=df_canal["canal_marketing"],
        orientation="h",
        marker=dict(color="#00D4FF"),
        hovertemplate=(
            "<b>Canal:</b> %{y}<br><b>Receita:</b> R$ %{x:,.2f}<extra></extra>"
        ),
    )
)
fig_canal.update_layout(
    title="Receita Gerada por Canal de Aquisição",
    xaxis_title="Faturamento (R$)",
    yaxis_title="Canal",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20),
    height=360,
)

with col_g1:
  st.plotly_chart(fig_canal, use_container_width=True)

# Gráfico 2: ROAS Médio por Categoria (Apenas canais com investimento)
df_roas_cat = (
    df_filtrado[df_filtrado["investimento_mkt"] > 0]
    .groupby("categoria", as_index=False)["roas"]
    .mean()
    .sort_values(by="roas", ascending=False)
)

fig_roas = go.Figure()
fig_roas.add_trace(
    go.Bar(
        x=df_roas_cat["categoria"],
        y=df_roas_cat["roas"],
        marker=dict(color="#FFA500"),
        hovertemplate=(
            "<b>Categoria:</b> %{x}<br><b>ROAS Médio:</b> %{y:.2f}x<extra></extra>"
        ),
    )
)
fig_roas.update_layout(
    title="Eficiência de Mídia Paga (ROAS Médio por Categoria)",
    xaxis_title="Categoria",
    yaxis_title="ROAS Médio (x)",
    template="plotly_dark",
    margin=dict(l=20, r=20, t=40, b=20),
    height=360,
)

with col_g2:
  st.plotly_chart(fig_roas, use_container_width=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# 7. CAMADA DE IA GENERATIVA (GOOGLE GEMINI)
# -----------------------------------------------------------------------------
st.subheader("🤖 Consultor Executivo com Gemini AI")
st.write(
    "Selecione uma análise estratégica sugerida ou digite sua própria pergunta"
    " de negócio baseada nos filtros selecionados."
)


def prepara_prompt_dados(df_contexto):
  stats = df_contexto.groupby(
      ["categoria", "canal_marketing"], as_index=False
  ).agg({
      "investimento_mkt": "mean",
      "desconto_medio_pct": "mean",
      "receita_vendas": "mean",
  })
  resumo = (
      f"Estatísticas de Marketing consolidada para o período"
      f" ({mes_selecionado}):\n"
  )
  for _, row in stats.iterrows():
    resumo += (
        f"- Categoria: {row['categoria']}, Canal: {row['canal_marketing']}, "
        f"Média Investimento: R$ {row['investimento_mkt']:.2f}, "
        f"Média Desconto: {row['desconto_medio_pct']:.2f}%, "
        f"Média Receita: R$ {row['receita_vendas']:.2f}\n"
    )
  return resumo


# Lista de perguntas estratégicas predefinidas
sugestoes_perguntas = [
    (
        "🎯 Diagnóstico Geral: Quais canais e categorias devem ter o orçamento"
        " ampliado e onde cortar verbas?"
    ),
    (
        "✂️ Cenário de Corte: Se precisarmos cortar 20% do orçamento total,"
        " quais canais e categorias devem sofrer cortes primeiro?"
    ),
    (
        "🏷️ Elasticidade de Preço: Quais categorias apresentam risco de queima"
        " de margem por excesso de desconto?"
    ),
    (
        "⚔️ Comparativo Direto: Compare a eficiência e ROAS do TikTok Ads"
        " contra o Meta Ads neste período."
    ),
    (
        "💼 Resumo Executivo: Crie 3 tópicos diretos para apresentar na reunião"
        " de diretoria com os principais destaques."
    ),
    "✏️ Outra pergunta (Digitar manualmente...)",
]

opcao_selecionada = st.selectbox(
    "Selecione uma pergunta de negócio:", sugestoes_perguntas
)

if opcao_selecionada == "✏️ Outra pergunta (Digitar manualmente...)":
  pergunta_executiva = st.text_area(
      "Digite sua pergunta personalizada sobre os dados:",
      value="Quais ações rápidas podemos tomar para aumentar o ROAS geral no próximo mês?",
      height=90,
  )
else:
  pergunta_executiva = opcao_selecionada

if st.button("🚀 Gerar Insights e Recomendações com Gemini"):
  if not client:
    st.error(
        "Chave GEMINI_API_KEY não localizada. Configure seu arquivo .env ou os"
        " Secrets do Streamlit."
    )
  else:
    with st.spinner("O Gemini está analisando as métricas consolidadas..."):
      contexto = prepara_prompt_dados(df_filtrado)
      prompt_final = f"""
            Você é um especialista sênior em Marketing Analytics, E-commerce e Otimização de Performance (Growth/ROAS).
            
            Analise atentamente o resumo estatístico consolidado a seguir:
            ----------------------------------------
            {contexto}
            ----------------------------------------
            
            Responda à pergunta do gestor:
            "{pergunta_executiva}"
            
            Diretrizes para o relatório:
            1. Seja analítico, objetivo e direto ao ponto.
            2. Calcule e aponte os canais de melhor e pior ROAS.
            3. Alerte sobre categorias queimando margem com excesso de desconto.
            4. Conclua com 3 a 4 recomendações práticas de realocação orçamentária.
            5. Formatação: Nunca use sintaxe matemática/LaTeX para valores monetários. Escreva os valores sempre como texto simples no padrão brasileiro (ex: R$ 1.500,00).
            """
      try:
        config = types.GenerateContentConfig(
            temperature=0.2,
        )
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt_final,
            config=config,
        )

        texto_formatado = response.text.replace("R$", "R\\$").replace(
            " $", " \\$"
        )
        st.markdown("### 📋 Diagnóstico Executivo")
        st.markdown(texto_formatado)

      except Exception as ex:
        if "429" in str(ex) or "RESOURCE_EXHAUSTED" in str(ex):
          st.warning(
              "⏳ **Limite temporário de requisições atingido (Cota do Gemini"
              " Free).** Por favor, aguarde cerca de 1 minuto antes de gerar"
              " uma nova análise."
          )
        else:
          st.error(f"Erro ao processar chamada com a API do Gemini: {ex}")

st.markdown("---")

# -----------------------------------------------------------------------------
# 8. TABELA DE DADOS E EXPORTAÇÃO
# -----------------------------------------------------------------------------
st.subheader("Detalhamento Transacional & Exportação")
st.write(
    "A tabela abaixo exibe a relação consolidada por categoria e canal no"
    " período filtrado."
)

df_tabela = df_filtrado.groupby(
    ["categoria", "canal_marketing"], as_index=False
).agg({
    "investimento_mkt": "sum",
    "receita_vendas": "sum",
    "desconto_medio_pct": "mean",
    "roas": "mean",
})

df_tabela.columns = [
    "Categoria",
    "Canal de Mídia",
    "Investimento Total (R$)",
    "Receita Total (R$)",
    "Desconto Médio (%)",
    "ROAS Médio (x)",
]

st.dataframe(
    df_tabela.style.format({
        "Investimento Total (R$)": "R$ {:,.2f}",
        "Receita Total (R$)": "R$ {:,.2f}",
        "Desconto Médio (%)": "{:.2f}%",
        "ROAS Médio (x)": "{:.2f}x",
    }),
    use_container_width=True,
    hide_index=True,
)

csv_data = df_tabela.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Baixar Tabela Consolidada em CSV",
    data=csv_data,
    file_name=f"analise_marketing_{mes_selecionado.replace('/', '_')}.csv",
    mime="text/csv",
    help="Clique para baixar a visão consolidada em formato CSV.",
)