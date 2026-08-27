# 📊 MarketIntel: Otimização de ROI e Performance em E-Commerce com PySpark e Google Gemini LLM

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat&logo=python)
![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.2-orange?style=flat&logo=apachespark)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue?style=flat&logo=docker)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-3.5%20Flash-8E75B2?style=flat&logo=google)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

Pipeline analítico de ponta a ponta desenvolvido para processamento distribuído com **Apache Spark (PySpark)** integrado à inteligência artificial generativa via SDK oficial do **Google Gemini (SDK `google-genai`)**.

O projeto processa mais de 3.600 registros de transações de e-commerce multicanal, consolidando métricas de investimento em tráfego pago, elasticidade de descontos e receita de vendas. Os resumos estatísticos são estruturados e direcionados ao LLM para geração automatizada de relatórios executivos com diagnósticos de ROAS e recomendações práticas de realocação orçamentária.

---

## 💡 Motivação & Perguntas de Negócio

Em operações multicanais de e-commerce (Google Ads, Meta Ads, TikTok Ads, Email Marketing e Orgânico), a tomada de decisão sobre alocação de orçamento e precificação frequentemente enfrenta gargalos: relatórios estáticos que demoram para ser consolidados e falta de interpretação ágil sobre a correlação entre descontos promocionais e eficiência de mídia paga.

Este pipeline foi projetado para responder às seguintes questões de negócio:
1. **Eficiência de Canais & ROAS:** Quais canais de aquisição entregam o melhor retorno sobre o investimento publicitário (ROAS) por categoria de produto?
2. **Elasticidade de Descontos:** Em quais categorias o percentual médio de desconto está comprometendo a margem de contribuição sem gerar retorno proporcional de vendas?
3. **Tomada de Decisão Automatizada:** Como traduzir agregações de Big Data em planos de ação claros e acionáveis para times de Growth e Marketing usando LLMs?

---

## 🛠️ Arquitetura e Tecnologias

- **Apache Spark / PySpark 3.5:** Leitura distribuída, inferência de schema, transformações temporais e agregações analíticas de alto desempenho.
- **Google GenAI SDK (`google-genai`):** Integração com modelos da família Gemini (Gemini 3.5 Flash Lite) com controle de temperatura (`temperature=0.2`) e prompt engineering estruturado para análise de dados.
- **Docker & Docker Compose:** Containerização do cluster Spark (Master e Workers) garantindo reprodutibilidade do ambiente.
- **Python-Dotenv:** Gerenciamento seguro de credenciais e chaves de API via variáveis de ambiente (`.env`).
- **Streamlit:** Interface web interativa para exploração dinâmica dos dados e geração de insights em tempo real.

---

## 🔬 Metodologia e Pipeline de Dados

```text
[Dataset CSV (+3.6k registros)] 
        │
        ▼
[Cluster PySpark] ──► Agregações por Categoria, Canal, Mês, Médias de Investimento e Desconto
        │
        ▼
[Prompt Engineering] ──► Formatação de Sumário Estatístico + Diretrizes de Negócio
        │
        ▼
[Google Gemini LLM] ──► Diagnóstico Executivo, Cálculo de ROAS e Plano de Ação
```

### 1. Processamento Distribuído
- Ingestão com `inferSchema=True` e particionamento dos dados.
- Agregações multivariadas (`groupBy` por categoria e canal) calculando médias de investimento (`investimento_mkt`), taxas de desconto (`desconto_medio_pct`) e receita líquida (`receita_vendas`).

### 2. Camada de Inteligência Analítica (LLM)
- O modelo atua como um especialista sênior em Marketing Analytics e Growth.
- Configuração determinística para focar em dados numéricos, evitando alucinações e priorizando cálculos reais de ROAS (Receita / Investimento).
- Geração de recomendações táticas: canais prioritários para escala (scale-up), canais para auditoria de lances e ajustes de política comercial.

---

## 🗂️ Estrutura do Repositório

```text
├── dados/
│   └── dataset_ecommerce_marketing.csv     # Conjunto de dados histórico (3.600+ registros)
├── jobs/
│   └── script.py                           # Pipeline principal em PySpark integrado ao Gemini
├── app.py                                  # Aplicação web interativa em Streamlit
├── docker-compose.yml                      # Orquestração do cluster Spark no Docker
├── .env.example                            # Modelo de configuração de variáveis de ambiente
├── .gitignore                              # Proteção para exclusão de chaves (.env) e arquivos temporários
├── README.md                               # Documentação técnica do projeto
└── requirements.txt                        # Dependências do ecossistema Python/Spark/Gemini
```

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- Python 3.10+ (ou ambiente configurado com Apache Spark / PySpark).
- Chave de API do Google Gemini (gratuita via [Google AI Studio](https://aistudio.google.com/)).

### Passo a Passo

1. **Clonar o Repositório:**
```bash
git clone [https://github.com/seu-usuario/marketing-analytics-spark-llm.git](https://github.com/seu-usuario/marketing-analytics-spark-llm.git)
cd marketintel-pyspark-gemini-llm
```

2. **Configurar o Ambiente Virtual e Variáveis:**
```bash
python -m venv .venv
# Linux/Mac: source .venv/bin/activate | Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```
*Edite o arquivo `.env` e insira sua `GEMINI_API_KEY`.*

3. **Execução:**

- **Opção A: Execução Local Direta**
```bash
python script.py
```

- **Opção B: Submissão via Spark-Submit (Cluster ou Container Spark)**
```bash
spark-submit --master local[*] script.py
```

---

## 💻 Desenvolvedor

**Douglas Vittori** - Cientista de Dados em Formação

🔗 Conecte-se comigo no LinkedIn: [https://www.linkedin.com/in/douglasvittori/](https://www.linkedin.com/in/douglasvittori/)

🚀 Conheça meu Portfólio de Dados: [https://douglasvittori-portfolio.lovable.app/](https://douglasvittori-portfolio.lovable.app/)