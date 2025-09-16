import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

# Configuração da página
st.set_page_config(
    page_title="Pesquisa PrEP/HIV - São Paulo",
    page_icon="🏳️‍🌈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado com melhor contraste
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E90FF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #4682B4;
        border-bottom: 2px solid #1E90FF;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
    }
    .stButton>button:hover {
        background-color: #4682B4;
    }
    .success-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #E6F7FF;
        border-left: 5px solid #1E90FF;
        margin-bottom: 1.5rem;
        color: #000000 !important;
    }
    .info-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #F0F8FF;
        border-left: 5px solid #4682B4;
        margin: 1rem 0;
        color: #000000 !important;
    }
    .tech-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #F8F9FA;
        border-left: 5px solid #28A745;
        margin: 0.5rem 0;
        color: #000000 !important;
    }
    .ml-explanation {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #FFFFFF;
        border: 2px solid #6F42C1;
        margin: 1rem 0;
        color: #000000 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Melhorar contraste para todo o texto */
    .stMarkdown, .stText, .stAlert, .stInfo {
        color: #000000 !important;
    }
    /* Garantir que todos os textos sejam visíveis */
    div[data-testid="stMarkdownContainer"] {
        color: #000000 !important;
    }
    /* Remover fundos brancos sobrepostos */
    .stApp {
        background-color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Inicialização de dados
if 'dados' not in st.session_state:
    if os.path.exists("respostas_prep.csv"):
        st.session_state.dados = pd.read_csv("respostas_prep.csv")
    else:
        st.session_state.dados = pd.DataFrame()

# Função para salvar dados
def salvar_dados(resposta):
    arquivo_csv = "respostas_prep.csv"
    
    # Adicionar timestamp
    resposta['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if os.path.exists(arquivo_csv):
        df_existente = pd.read_csv(arquivo_csv)
        df_novo = pd.DataFrame([resposta])
        df_final = pd.concat([df_existente, df_novo], ignore_index=True)
    else:
        df_final = pd.DataFrame([resposta])
    
    df_final.to_csv(arquivo_csv, index=False)
    st.session_state.dados = df_final
    return True

# Barra lateral com informações do projeto
with st.sidebar:
    st.title("Sobre o Projeto")
    st.markdown("""
    <div class="info-box">
        <h4 style="color: #000000;">Pesquisa sobre Conhecimento de PrEP/PEP</h4>
        <p style="color: #000000;">Este projeto visa mapear o conhecimento sobre métodos de prevenção ao HIV na população de São Paulo.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Explicação sobre PrEP e PEP
    with st.expander("ℹ️ O que é PrEP e PEP?"):
        st.markdown("""
        <div style="color: #000000;">
            <h4 style="color: #000000;">PrEP (Profilaxia Pré-Exposição)</h4>
            <p style="color: #000000;">Uso diário de medicamentos por pessoas HIV-negativas para reduzir o risco de infecção pelo HIV.</p>
            
            <h4 style="color: #000000;">PEP (Profilaxia Pós-Exposição)</h4>
            <p style="color: #000000;">Uso emergencial de medicamentos por pessoas que possam ter sido expostas ao HIV, iniciado em até 72 horas após a exposição.</p>
        </div>
        """, unsafe_allow_html=True)

# Cabeçalho
st.markdown('<h1 class="main-header">Pesquisa sobre PrEP e Prevenção ao HIV em São Paulo</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem; color: #000000;">
    <p>Esta pesquisa tem como objetivo mapear o conhecimento sobre PrEP e PEP na população de São Paulo, 
    identificando lacunas de informação e barreiras de acesso.</p>
    <p><strong>Todas as informações são anônimas e confidenciais.</strong></p>
</div>
""", unsafe_allow_html=True)

# Formulário de pesquisa
with st.form("pesquisa_form"):
    st.markdown('<h2 class="section-header">Parte 1: Conhecimento sobre PrEP/PEP</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        q1 = st.radio("**Você conhece a PrEP (Profilaxia Pré-Exposição)?**", [
            "Sim, conheço bem", 
            "Conheço parcialmente", 
            "Já ouvi falar mas não sei detalhes", 
            "Não conheço"
        ])
        
        q2 = st.radio("**E a PEP (Profilaxia Pós-Exposição)?**", [
            "Sim, conheço bem", 
            "Conheço parcialmente", 
            "Já ouvi falar mas não sei detalhes", 
            "Não conheço"
        ])
    
    with col2:
        q3 = st.radio("**Você sabe onde conseguir PrEP/PEP em São Paulo?**", [
            "Sim, conheço vários serviços",
            "Conheço apenas um local",
            "Não sei mas gostaria de saber",
            "Não sei e não tenho interesse"
        ])
        
        q4 = st.radio("**Como você ficou sabendo sobre PrEP/PEP?**", [
            "Profissional de saúde",
            "Amigos/conhecidos",
            "Internet/redes sociais",
            "Material informativo (folhetos, cartazes)",
            "Nunca ouvi falar",
            "Outra fonte"
        ])
    
    st.markdown('<h2 class="section-header">Parte 2: Experiência Pessoal</h2>', unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        q5 = st.radio("**Você já usou ou usa PrEP/PEP?**", [
            "Sim, uso atualmente",
            "Sim, já usei no passado",
            "Não, mas pretendo usar",
            "Não uso e não tenho interesse",
            "Prefiro não responder"
        ])
        
        q6 = st.radio("**Conhece alguém que usa ou já usou PrEP/PEP?**", [
            "Sim, vários conhecidos",
            "Sim, algumas pessoas",
            "Não conheço ninguém",
            "Prefiro não responder"
        ])
    
    with col4:
        q7 = st.radio("**Com que frequência você faz teste de HIV?**", [
            "A cada 3 meses",
            "A cada 6 meses",
            "Uma vez por ano",
            "Raramente faço",
            "Nunca fiz",
            "Prefiro não responder"
        ])
        
        q8 = st.multiselect("**Quais métodos de prevenção ao HIV você utiliza?**", [
            "PrEP",
            "PEP",
            "Camisinha masculina",
            "Camisinha feminina",
            "Testagem regular",
            "Não utilizo métodos de prevenção",
            "Outro"
        ])
    
    st.markdown('<h2 class="section-header">Parte 3: Perfil Demográfico</h2>', unsafe_allow_html=True)
    
    col5, col6 = st.columns(2)
    
    with col5:
        genero = st.selectbox("**Identidade de gênero:**", [
            "Mulher cisgênero",
            "Homem cisgênero",
            "Mulher trans/transgênero",
            "Homem trans/transgênero",
            "Pessoa não-binária",
            "Travesti",
            "Agênero",
            "Gênero fluido",
            "Outro",
            "Prefiro não responder"
        ])
        
        orientacao = st.selectbox("**Orientação sexual:**", [
            "Assexual",
            "Bissexual",
            "Gay",
            "Lésbica",
            "Pansexual",
            "Heterossexual",
            "Queer",
            "Outra",
            "Prefiro não responder"
        ])
        
        raca = st.radio("**Raça/Cor:**", [
            "Amarela (origem asiática)",
            "Branca",
            "Indígena",
            "Parda",
            "Preta",
            "Prefiro não responder"
        ])
    
    with col6:
        idade = st.radio("**Faixa etária:**", [
            "13-17", "18-24", "25-29", "30-39",
            "40-49", "50-59", "60+", "Prefiro não responder"
        ])
        
        renda = st.radio("**Renda mensal individual:**", [
            "Até 1 salário mínimo", 
            "1-2 salários mínimos",
            "2-3 salários mínimos", 
            "3-5 salários mínimos",
            "Mais de 5 salários mínimos",
            "Prefiro não responder"
        ])
        
        regiao = st.selectbox("**Região de São Paulo onde mora:**", [
            "Centro expandido",
            "Zona Norte",
            "Zona Sul",
            "Zona Leste",
            "Zona Oeste",
            "Região Metropolitana",
            "Não moro em São Paulo",
            "Prefiro não responder"
        ])
    
    # Termos de consentimento
    st.markdown("""
    <div style="background-color: #F0F8FF; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; color: #000000;">
        <p><strong>Termo de Consentimento:</strong> Ao enviar este formulário, você concorda em participar desta pesquisa 
        e que seus dados anônimos sejam utilizados para fins de estudo estatístico. Todas as informações são confidenciais 
        e não serão compartilhadas de forma individual.</p>
    </div>
    """, unsafe_allow_html=True)
    
    consentimento = st.checkbox("**Eu concordo em participar da pesquisa**", value=False)
    
    # Botão de envio
    enviado = st.form_submit_button("Enviar Respostas")
    
    if enviado and consentimento:
        resposta = {
            "Conhecimento_PrEP": q1,
            "Conhecimento_PEP": q2,
            "Acesso_servicos": q3,
            "Fonte_informacao": q4,
            "Uso_PrepPEP": q5,
            "Conhece_usuarios": q6,
            "Teste_HIV_frequencia": q7,
            "Metodos_prevencao": ", ".join(q8),
            "Genero": genero,
            "Orientacao_sexual": orientacao,
            "Raca": raca,
            "Faixa_etaria": idade,
            "Renda": renda,
            "Regiao": regiao
        }
        
        if salvar_dados(resposta):
            st.markdown("""
            <div class="success-box">
                <h3 style="color: #000000;">✅ Obrigado por participar da pesquisa!</h3>
                <p style="color: #000000;">Sua contribuição é muito importante para entendermos melhor o conhecimento sobre 
                prevenção ao HIV em nossa comunidade.</p>
            </div>
            """, unsafe_allow_html=True)
    elif enviado and not consentimento:
        st.error("Você precisa concordar com os termos de consentimento para enviar o formulário.")

# Seção de visualizações (apenas se houver dados)
if not st.session_state.dados.empty:
    st.markdown("---")
    st.markdown('<h2 class="section-header">Análise dos Dados Coletados</h2>', unsafe_allow_html=True)
    
    # Estatísticas rápidas
    total_respostas = len(st.session_state.dados)
    st.markdown(f"""
    <div style="background-color: #E6F7FF; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1.5rem; color: #000000;">
        <h3 style="text-align: center; color: #000000;">📊 Total de Respostas: {total_respostas}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Filtros para os gráficos
    st.sidebar.header("Filtros para Análise")
    
    # Seleção de variáveis para análise
    variavel_x = st.sidebar.selectbox(
        "Variável para análise (eixo X):",
        options=[col for col in st.session_state.dados.columns if col != 'timestamp'],
        index=0
    )
    
    # Gráfico de barras da variável selecionada
    try:
        st.subheader(f"Distribuição de {variavel_x}")
        contagem = st.session_state.dados[variavel_x].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=contagem.values, y=contagem.index, ax=ax, palette="viridis")
        ax.set_xlabel("Número de respostas")
        ax.set_ylabel("")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro ao criar gráfico: {str(e)}")
    
    # Gráficos de comparação
    st.subheader("Relação entre Conhecimento e Demografia")
    
    col7, col8 = st.columns(2)
    
    with col7:
        try:
            # Conhecimento de PrEP por gênero
            conhecimento_genero = pd.crosstab(
                st.session_state.dados['Genero'], 
                st.session_state.dados['Conhecimento_PrEP']
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            conhecimento_genero.plot(kind='bar', ax=ax, colormap='Set3')
            ax.set_title("Conhecimento de PrEP por Identidade de Gênero")
            ax.legend(title="Conhecimento", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao criar gráfico de gênero: {str(e)}")
    
    with col8:
        try:
            # Conhecimento de PrEP por faixa etária
            conhecimento_idade = pd.crosstab(
                st.session_state.dados['Faixa_etaria'], 
                st.session_state.dados['Conhecimento_PrEP']
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            conhecimento_idade.plot(kind='bar', ax=ax, colormap='Set2')
            ax.set_title("Conhecimento de PrEP por Faixa Etária")
            ax.legend(title="Conhecimento", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erro ao criar gráfico de idade: {str(e)}")
    
    # Análise de Machine Learning (Agrupamento) - SOMENTE SE HOUVER DADOS SUFICIENTES
    if len(st.session_state.dados) >= 3:  # Pelo menos 3 respostas para 3 clusters
        st.markdown("---")
        st.markdown('<h2 class="section-header">Análise com Inteligência Artificial</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #FFF0F5; padding: 1.5rem; border-radius: 0.5rem; border: 2px solid #FF69B4; margin: 1rem 0; color: #000000;">
            <h4 style="color: #000000;">🤖 Como a Inteligência Artificial está analisando estas respostas</h4>
            <p style="color: #000000;">Usamos um algoritmo de <strong>aprendizado não supervisionado</strong> chamado K-Means para agrupar 
            automaticamente os respondentes com base em padrões similares em suas respostas.</p>
            
            <h5 style="color: #000000;">O processo:</h5>
            <ol style="color: #000000;">
                <li>Convertemos todas as respostas em valores numéricos</li>
                <li>Normalizamos os dados para que todas as variáveis tenham o mesmo peso</li>
                <li>Reduzimos a dimensionalidade para visualização em 2D (PCA)</li>
                <li>Agrupamos os respondentes em clusters com características similares</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
        try:
            # Preparar dados para clustering
            dados_ml = st.session_state.dados.copy()
            
            # Codificar variáveis categóricas
            le = LabelEncoder()
            for col in dados_ml.select_dtypes(include=['object']).columns:
                if col != 'timestamp' and col != 'Metodos_prevencao':
                    dados_ml[col] = le.fit_transform(dados_ml[col].astype(str))
            
            # Remover colunas com muitos valores únicos ou não numéricos
            dados_ml = dados_ml.drop(['timestamp', 'Metodos_prevencao'], axis=1, errors='ignore')
            
            # Verificar se ainda temos dados suficientes após o pré-processamento
            if len(dados_ml) >= 3:
                # Padronizar os dados
                scaler = StandardScaler()
                dados_scaled = scaler.fit_transform(dados_ml)
                
                # Aplicar K-Means (usando no máximo n-1 clusters onde n é o número de amostras)
                n_clusters = min(3, len(dados_scaled) - 1)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(dados_scaled)
                
                # Reduzir dimensionalidade para visualização
                pca = PCA(n_components=2)
                componentes = pca.fit_transform(dados_scaled)
                
                # Criar DataFrame para visualização
                df_viz = pd.DataFrame({
                    'PC1': componentes[:, 0],
                    'PC2': componentes[:, 1],
                    'Cluster': clusters
                })
                
                # Visualizar clusters
                fig, ax = plt.subplots(figsize=(10, 8))
                scatter = ax.scatter(df_viz['PC1'], df_viz['PC2'], c=df_viz['Cluster'], cmap='viridis', alpha=0.7)
                ax.set_xlabel('Componente Principal 1')
                ax.set_ylabel('Componente Principal 2')
                ax.set_title('Agrupamento de Respostas usando Machine Learning (K-Means)')
                legend = ax.legend(*scatter.legend_elements(), title="Clusters")
                ax.add_artist(legend)
                st.pyplot(fig)
                
                # Interpretação dos clusters
                st.info("""
                **Interpretação dos Clusters:** Esta análise agrupa os respondentes com base em padrões similares em suas respostas.
                - **Cluster 0**: Possivelmente representa pessoas com menor conhecimento sobre PrEP/PEP
                - **Cluster 1**: Possivelmente representa pessoas com conhecimento intermediário
                - **Cluster 2**: Possivelmente representa pessoas com maior conhecimento e experiência com PrEP/PEP
                """)
                
                # Estatísticas por cluster
                st.subheader("Características dos Grupos Identificados")
                
                # Adicionar cluster aos dados originais para análise
                dados_com_cluster = st.session_state.dados.copy()
                dados_com_cluster['Cluster'] = clusters
                
                # Mostrar características de cada cluster
                for cluster_id in range(n_clusters):
                    with st.expander(f"📋 Características do Grupo {cluster_id}"):
                        cluster_data = dados_com_cluster[dados_com_cluster['Cluster'] == cluster_id]
                        st.write(f"**Tamanho do grupo:** {len(cluster_data)} respondentes")
                        
                        # Mostrar distribuição de algumas variáveis importantes
                        col9, col10 = st.columns(2)
                        
                        with col9:
                            st.write("**Conhecimento de PrEP:**")
                            st.write(cluster_data['Conhecimento_PrEP'].value_counts())
                        
                        with col10:
                            st.write("**Faixa Etária:**")
                            st.write(cluster_data['Faixa_etaria'].value_counts())
            else:
                st.warning("Não há dados suficientes para análise de agrupamento após o pré-processamento.")
        except Exception as e:
            st.error(f"Erro na análise de machine learning: {str(e)}")
    else:
        st.info("🧠 A análise com Inteligência Artificial será exibida quando houver pelo menos 3 respostas.")
    
    # Estatísticas descritivas
    try:
        st.markdown("---")
        st.subheader("Estatísticas Descritivas")
        st.dataframe(st.session_state.dados.describe(include='all'))
    except Exception as e:
        st.error(f"Erro ao exibir estatísticas: {str(e)}")
    
    # Download dos dados
    try:
        st.subheader("Exportar Dados")
        csv = st.session_state.dados.to_csv(index=False)
        st.download_button(
            label="Baixar dados completos (CSV)",
            data=csv,
            file_name="dados_prep_sp.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Erro ao preparar download: {str(e)}")

else:
    st.info("📝 Não há dados coletados ainda. As visualizações serão exibidas aqui quando houver respostas suficientes.")

# Rodapé
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #000000; margin-top: 2rem;">
    <p>Desenvolvido com Streamlit | Pesquisa sobre Prevenção ao HIV</p>
</div>
""", unsafe_allow_html=True)