import streamlit as st
import os
import tempfile
from cv_analyser_openrouter import CVAnalyserRAG
import json
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Carrega as variáveis do .env
load_dotenv()

# Pega a chave da API do ambiente
openrouter_key = os.getenv("OPENROUTER_API_KEY")

# Configuração da página
st.set_page_config(
    page_title="Analisador de Currículos",
    page_icon="📄",
    layout="wide"
)

# Decorator para tratamento de rate limit
def handle_rate_limit(func):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception),
    )
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "Rate limit exceeded" in str(e):
                reset_time = None
                try:
                    error_data = eval(str(e).split("Error code:")[-1])
                    reset_timestamp = int(error_data.get('error', {}).get('metadata', {}).get('headers', {}).get('X-RateLimit-Reset', 0))/1000
                    reset_time = datetime.fromtimestamp(reset_timestamp).strftime('%H:%M:%S')
                except:
                    pass
                
                if reset_time:
                    st.warning(f"Limite de requisições atingido. O limite será resetado às {reset_time}.")
                else:
                    st.warning("Limite de requisições atingido. Por favor, aguarde alguns minutos.")
                
                time.sleep(60)  # Espera 1 minuto antes de tentar novamente
                raise
            raise
    return wrapper

# Sidebar com configurações
with st.sidebar:
    st.title("Configurações")
    
    if not openrouter_key:
        st.error("Configure sua OPENROUTER_API_KEY no arquivo .env")
        st.stop()
    else:
        st.success("API Key configurada no .env")
    
    st.markdown("---")
    st.markdown("### Modelo LLM")
    model_option = st.selectbox(
        "Selecione o modelo",
        ("meta-llama/llama-3.3-8b-instruct:free", "deepseek/deepseek-prover-v2:free"),
        key="model_selector"
    )
    
    st.markdown("---")
    st.markdown("### Limite de Análise")
    max_files = st.slider(
        "Número máximo de currículos para analisar de uma vez",
        min_value=1,
        max_value=10,
        value=3,
        help="Ajuste para evitar atingir limites de requisição"
    )
    
    st.markdown("---")
    st.markdown("Desenvolvido por [Seu Nome]")

# Título principal
st.title("📄 Analisador de Currículos Inteligente")
st.markdown("""
Carregue os currículos em PDF e a descrição da vaga para receber:
- 📊 Pontuação automática
- 📝 Resumo estruturado
- 🔍 Análise detalhada
""")

# Área de upload
with st.expander("🔽 Upload de Currículos", expanded=True):
    uploaded_files = st.file_uploader(
        "Selecione os currículos em PDF (máximo {} por análise)".format(max_files),
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )

# Input da descrição da vaga
job_desc = st.text_area(
    "✍️ Descrição da Vaga",
    height=200,
    placeholder="""Exemplo:
Procuramos desenvolvedor Python com:
- 3+ anos de experiência com Django
- Conhecimento em bancos de dados SQL
- Inglês intermediário""",
    key="job_description"
)

# Botão de análise
if st.button("🔍 Analisar Currículos", use_container_width=True, key="analyze_button"):
    if not uploaded_files:
        st.warning("Por favor, carregue pelo menos um currículo.", icon="⚠️")
    elif not job_desc.strip():
        st.warning("Por favor, insira a descrição da vaga.", icon="⚠️")
    elif not openrouter_key:
        st.warning("Chave da API OpenRouter não configurada.", icon="⚠️")
    else:
        # Limita o número de arquivos para análise
        files_to_process = uploaded_files[:max_files]
        if len(uploaded_files) > max_files:
            st.warning(f"Analisando apenas os primeiros {max_files} currículos de {len(uploaded_files)} carregados.", icon="⚠️")
        
        with st.spinner("Processando currículos..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_paths = []
                for uploaded_file in files_to_process:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_paths.append(file_path)
                
                # Instancia o analisador
                analyser = CVAnalyserRAG()
                
                # Configura API key e modelo
                if hasattr(analyser, "set_api_key"):
                    analyser.set_api_key(openrouter_key)
                if hasattr(analyser, "set_model"):
                    analyser.set_model(model_option)
                
                # Carrega PDFs
                analyser.load_pdfs(temp_dir)
                
                # Analisa candidatos com tratamento de rate limit
                try:
                    @handle_rate_limit
                    def analyze_with_retry():
                        try:
                            return analyser.analyse_candidates(job_desc, api_key=openrouter_key, model=model_option)
                        except TypeError:
                            return analyser.analyse_candidates(job_desc)
                    
                    results = analyze_with_retry()
                    
                except Exception as e:
                    st.error(f"Erro na análise: {str(e)}")
                    st.stop()
                
                if not results:
                    st.warning("Nenhum resultado retornado da análise.", icon="⚠️")
                else:
                    st.success(f"Análise concluída para {len(results)} currículos!")
                    st.markdown("---")
                    
                    # Cria lista de labels para as tabs
                    tab_labels = [f"{res.get('file', 'Desconhecido')} ({res.get('score', 'N/A')}/10)" for res in results]
                    
                    tabs = st.tabs(tab_labels)
                    
                    for i, (tab, result) in enumerate(zip(tabs, results), 1):
                        with tab:
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.markdown("### 📝 Resumo")
                                st.markdown(result.get("summary", "Sem resumo disponível."))
                                
                                st.download_button(
                                    label="⬇️ Baixar Resumo",
                                    data=result.get("summary", ""),
                                    file_name=f"resumo_{result.get('file', 'curriculo').replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.md",
                                    mime="text/markdown",
                                    key=f"download_summary_{i}"
                                )
                            
                            with col2:
                                st.markdown("### 🔍 Análise Detalhada")
                                st.markdown(result.get("opinion", "Sem análise detalhada disponível."))
                                
                                st.download_button(
                                    label="⬇️ Baixar Análise Completa",
                                    data=result.get("opinion", ""),
                                    file_name=f"analise_{result.get('file', 'curriculo').replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d')}.txt",
                                    mime="text/plain",
                                    key=f"download_analysis_{i}"
                                )
                    
                    # Relatório Consolidado
                    st.markdown("---")
                    st.markdown("### 📊 Relatório Consolidado")
                    
                    st.markdown("#### 🏆 Ranking dos Candidatos")
                    for i, result in enumerate(sorted(results, key=lambda x: x.get('score', 0), reverse=True), 1):
                        st.markdown(f"{i}. **{result.get('file', 'Desconhecido')}** - Pontuação: **{result.get('score', 'N/A')}/10**")
                    
                    st.markdown("#### 📈 Visão Geral")
                    df = pd.DataFrame({
                        "Candidato": [res.get("file", "Desconhecido") for res in results],
                        "Pontuação": [res.get("score", 0) for res in results],
                        "Resumo": [res.get("summary", "")[:100] + "..." for res in results]
                    })
                    st.dataframe(df.sort_values(by="Pontuação", ascending=False), use_container_width=True, hide_index=True)
                    
                    # Download JSON consolidado
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    json_results = json.dumps(results, ensure_ascii=False, indent=2)
                    st.download_button(
                        label="⬇️ Baixar Relatório Completo (JSON)",
                        data=json_results,
                        file_name=f"relatorio_consolidado_{timestamp}.json",
                        mime="application/json",
                        key="download_json_full"
                    )

# Rodapé
st.markdown("---")
st.markdown("""
🔍 **Dicas para melhor análise:**
- Limite o número de currículos por análise para evitar limites de requisição
- Para muitos currículos, analise em lotes menores
- Considere atualizar para um plano pago se precisar de mais requisições
""")