import os
import re
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Atualizado
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from typing import List, Dict, Optional
load_dotenv()
openrouter_key = os.getenv("OPENROUTER_API_KEY")


class OpenRouterClient:
    def __init__(self, model_id: str = "meta-llama/llama-3.3-8b-instruct:free"):
        """
        Cliente para interação com a API OpenRouter
        
        Args:
            model_id: ID do modelo a ser usado (padrão: Llama 3 8B gratuito)
        """
        self.model_id = model_id
        self.client = self._initialize_client()
        
    def _initialize_client(self):
        """Configura o cliente OpenAI compatível com OpenRouter"""
        return OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
    def generate_response(self, prompt: str, max_retries: int = 3) -> str:
        """
        Gera resposta para um prompt usando a API OpenRouter
        
        Args:
            prompt: Texto de entrada para o modelo
            max_retries: Número máximo de tentativas em caso de falha
            
        Returns:
            Resposta textual do modelo
        """
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": "http://localhost:3000",
                        "X-Title": "CV Analyzer",
                    },
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=2000
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Falha após {max_retries} tentativas: {str(e)}")
                continue

    def resume_cv(self, cv_text: str) -> str:
        """
        Gera um resumo estruturado do currículo
        
        Args:
            cv_text: Texto extraído do currículo
            
        Returns:
            Resumo formatado em Markdown
        """
        prompt = f'''
            Resuma este currículo em markdown com as seções:
            - Nome Completo
            - Experiência (lista concisa)
            - Habilidades Técnicas (destaque as principais)
            - Educação (formação acadêmica)
            - Idiomas (com níveis de proficiência)
            
            **Currículo:**
            {cv_text}
            
            **Formato de Saída (use markdown):**
            ```markdown
            ## Nome Completo
            [nome aqui]
            
            ## Experiência
            - [Cargo] na [Empresa] ([Período])
            ...
            
            ## Habilidades
            - [Habilidade 1]
            ...
            ```
        '''
        return self._clean_response(self.generate_response(prompt))

    def generate_score(self, cv_text: str, job_description: str) -> float:
        """
        Avalia a adequação do currículo à vaga (0-10)
        
        Args:
            cv_text: Texto do currículo
            job_description: Descrição da vaga
            
        Returns:
            Pontuação numérica (0.0 a 10.0)
        """
        prompt = f'''
            Avalie este currículo para a vaga abaixo e atribua uma pontuação de 0 a 10.
            Retorne APENAS o número, sem explicações ou formatação.
            
            **Critérios:**
            - Relevância da experiência (40%)
            - Adequação das habilidades (30%)
            - Formação acadêmica (15%)
            - Idiomas (15%)
            
            **Vaga:**
            {job_description}
            
            **Currículo:**
            {cv_text}
        '''
        response = self.generate_response(prompt)
        try:
            score = float(re.search(r"\d+\.?\d*", response).group())
            return min(max(score, 0.0), 10.0)  # Garante que está entre 0 e 10
        except:
            return 0.0

    def generate_opinion(self, cv_text: str, job_description: str) -> str:
        """
        Gera uma análise crítica detalhada do currículo
        
        Args:
            cv_text: Texto do currículo
            job_description: Descrição da vaga
            
        Returns:
            Análise formatada em Markdown
        """
        prompt = f'''
            Analise este currículo em relação à vaga e produza um relatório detalhado
            com os seguintes tópicos (formate como Markdown com subtítulos):
            
            ### 🔍 Pontos Fortes
            - [Liste 3-5 pontos fortes relevantes para a vaga]
            
            ### ⚠️ Pontos de Atenção
            - [Mencione 2-3 áreas que poderiam ser melhoradas]
            
            ### 💡 Recomendações
            - [Sugira melhorias específicas para o candidato]
            
            **Dados para Análise:**
            === VAGA ===
            {job_description}
            
            === CURRÍCULO ===
            {cv_text}
        '''
        return self._clean_response(self.generate_response(prompt))

    def _clean_response(self, text: str) -> str:
        """Remove formatação desnecessária da resposta do modelo"""
        if '```markdown' in text:
            return text.split('```markdown')[1].split('```')[0].strip()
        return text.strip()


class CVAnalyserRAG:
    def __init__(self):
        """
        Sistema RAG para análise de currículos com:
        - Processamento de PDFs
        - Embeddings e busca semântica
        - Integração com LLM via OpenRouter
        """
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'}
        )
        self.vectorstore = None
        self.llm = OpenRouterClient()

    def load_pdfs(self, pdf_folder: str) -> None:
        """
        Carrega e processa currículos em PDF de uma pasta
        
        Args:
            pdf_folder: Caminho para a pasta com os PDFs
        """
        if not os.path.exists(pdf_folder):
            raise ValueError(f"Pasta não encontrada: {pdf_folder}")
            
        documents = []
        for filename in os.listdir(pdf_folder):
            if filename.endswith(".pdf"):
                path = os.path.join(pdf_folder, filename)
                try:
                    loader = PyPDFLoader(path)
                    pages = loader.load()
                    for page in pages:
                        page.metadata["source"] = filename
                    documents.extend(pages)
                except Exception as e:
                    print(f"Erro ao processar {filename}: {str(e)}")
        
        if not documents:
            raise ValueError("Nenhum PDF válido encontrado na pasta")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

    def analyse_candidates(self, job_description: str, top_k: int = 5) -> List[Dict]:
        """
        Analisa os currículos mais relevantes para uma vaga
        
        Args:
            job_description: Descrição textual da vaga
            top_k: Número de currículos para retornar
            
        Returns:
            Lista de dicionários com resultados da análise
        """
        if not self.vectorstore:
            raise ValueError("Nenhum currículo carregado. Use load_pdfs() primeiro.")
            
        if not job_description.strip():
            raise ValueError("A descrição da vaga não pode estar vazia")
            
        docs = self.vectorstore.similarity_search(job_description, k=top_k)
        
        results = []
        for doc in docs:
            try:
                cv_content = doc.page_content
                results.append({
                    "file": doc.metadata["source"],
                    "score": self.llm.generate_score(cv_content, job_description),
                    "summary": self.llm.resume_cv(cv_content),
                    "opinion": self.llm.generate_opinion(cv_content, job_description),
                    "content": cv_content[:500] + "..."
                })
            except Exception as e:
                print(f"Erro ao analisar {doc.metadata['source']}: {str(e)}")
        
        return sorted(results, key=lambda x: x["score"], reverse=True)


# Exemplo de uso (para testes)
if __name__ == "__main__":
    analyser = CVAnalyserRAG()
    
    # Exemplo com PDFs locais
    pdf_folder = "cv_pdfs"  # Pasta com currículos
    if os.path.exists(pdf_folder):
        analyser.load_pdfs(pdf_folder)
        
        job_desc = """
        Desenvolvedor Python Sênior
        Requisitos:
        - 5+ anos com Python
        - Experiência com Django/Flask
        - Conhecimento em bancos de dados SQL/NoSQL
        - Inglês técnico
        """
        
        results = analyser.analyse_candidates(job_desc)
        print(f"Melhor candidato: {results[0]['file']} ({results[0]['score']}/10)")