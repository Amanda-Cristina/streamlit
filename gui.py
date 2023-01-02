import streamlit as st
import pandas as pd
import pickle
import base64

from configs import load_configs

# https://discuss.streamlit.io/t/streamlit-pycaret-an-end-to-end-machine-learning-web-application/9623


class BASEAPP:
    """
    Classe base da integração com interface Pill utilizando o Streamlit
    """

    def __init__(self, path: str) -> None:
        """
        Inicialização do objeto APP

        Parameters
        ----------
        path: str
            Caminho do arquivo YAML de configurações
        """
        self.configs = load_configs(path)
        self.df_dataset = {}
        self.inputs_model = {}
        self.config_posprocessing = {}
        self.inputs_posprocessing = {}

    def header(self, use_image=True):
        
        st.header(self.configs.nome)
        st.text(f"Versão: {self.configs.versao}")
        with st.expander("See explanation"):
            st.text(self.configs.descricao)

            if use_image:
                st.image("imgs/rascience2.png")

        # revisar
        # Valida se tests existe para chamar função de configs
        # if "tests" in self.configs.__dict__:
        #     self.__load_test_configs()

    def footer(self):
        """
        Sessão final da interface que carrega
        dados presentes no YAML como links_referencias
        """
        st.markdown(f"#### References \n")
        with st.expander("See all"):
            for nome, link in self.configs.links_referencias.items():
                st.markdown(f"- [{nome}]({link})")

    def __load_test_configs(self):
        """
        Carrega configurações presentes em self.configs.tests
        e atualiza os atributos de self.inputs.
        """
        self.use_test_data = st.checkbox("Usar DF de testes")
        if self.use_test_data:
            self.inputs = self.configs.get_tests()
            st.json(self.inputs)

    @st.cache(max_entries = 5)  
    def __upload_dataset(self, dataset):
        """
        Método interno para salvar dataset em DataFrame

        Parameters
        ----------
        title:str
            Nome do widget
        key:str
            Identificador unico do widget
        """

        return pd.read_csv(dataset)

    def container_dataset(self):          
        with st.sidebar.expander("Upload Datasets", expanded=False):
            datasets = self.configs.parametros["dataset"]
            for nome, opcoes in datasets.items():
                titulo = opcoes["titulo"] +' *' if "obrigatorio" in opcoes else opcoes["titulo"] 
            
                dataset = st.file_uploader(titulo, key=nome, help = opcoes["descricao"])
                
                if (dataset is not None):
                    self.df_dataset[nome] = self.__upload_dataset(dataset)
                    
        if (len(self.df_dataset) == len(datasets)):
            st.session_state['df_dataset'] = self.df_dataset 
            
    def show_dataset(self):
        datasets = self.configs.parametros["dataset"]
  
        for nome,df in st.session_state['df_dataset'].items(): 
            label = datasets[nome]["titulo"]
            st.markdown(f"##### {label}")          
            st.text(f"Shape: {df.shape}")
            st.dataframe(df.head())



    def __convert_input(self, nome, titulo, descricao, tipo, valor, opcao, obrigatorio, controle):
        """
        Método interno de conversão das informações de
        configurações para a interface do streamlit

        Parameters
        ----------
        """
       
        titulo = titulo +' *' if obrigatorio else titulo 
        
        if tipo == "date":
            return st.text_input(titulo, value="" if valor is None else valor, key = nome, help = descricao)
        
        if tipo == "int": 
            if controle is None:                   
                return st.number_input(titulo, value= 1 if valor is None else valor, key = nome, help = descricao)
            else:
                return st.number_input(titulo, value= 1 if valor is None else valor, key = nome, help = descricao, disabled = False if st.session_state[controle] else True)
            
        if tipo == "float":
            return st.number_input(titulo, value=1.0 if valor is None else valor, key = nome, help = descricao)
        
        if tipo == "str":
            return st.text_input(titulo, value="" if valor is None else valor, key = nome, help = descricao)
        
        if tipo == "column":
            return st.selectbox(titulo,options=[] if 'df_dataset' not in st.session_state else st.session_state.df_dataset["df"].columns, key = nome, help = descricao)
        
        if tipo == "columns":
            if controle is None:
                return st.multiselect(titulo,options=[] if 'df_dataset' not in st.session_state else st.session_state.df_dataset[opcao].columns, key = nome, help = descricao)
            else:
                return st.multiselect(titulo,options=[] if 'df_dataset' not in st.session_state else st.session_state.df_dataset[opcao].columns, key = nome, help = descricao, disabled = False if st.session_state[controle] else True)
            
        if tipo == "list":
            return st.text_input(titulo, value="" if valor is None else valor, key = nome, help = descricao)
        
        if tipo == "bool":
            return st.checkbox(titulo, value=False if valor is None else valor, key = nome, help = descricao)

    def container_inputs(self, model):
        """
        Bloco que converte inputs do YAML de
        configs > parametros > input em widgets
        para o usuário utilizar.
        """

        titulos_blocos = self.configs.parametros["input"]["config"]["blocos"]
        
        for bloco, topicos in self.configs.parametros["input"]["config"]["topicos"].items():
            st.sidebar.title(titulos_blocos[bloco])
            for topico, titulo in topicos.items():
                with st.sidebar.expander(titulo, expanded=False):
                    for nome, opcoes in self.configs.parametros["input"]["widgets"][bloco][topico].items():
                        parametros = [nome,
                                    opcoes["titulo"], 
                                    opcoes["descricao"], 
                                    opcoes["tipo"], 
                                    opcoes["valor"] if "valor" in opcoes else None,
                                    opcoes["opcao"] if "opcao" in opcoes else None,
                                    opcoes["obrigatorio"] if "obrigatorio" in opcoes else False,
                                    opcoes["controle"] if "controle" in opcoes else None]
                        
                        if ('posprocesso' in opcoes):
                            self.config_posprocessing[nome] = self.__convert_input(*parametros)
                        else: 
                            self.inputs_model[nome] = self.__convert_input(*parametros)
        with st.sidebar:
            # Botão execução Modelo
            botao_model = st.button("Executar", help="Vai dar bom, confia")
            if botao_model:
                if(self.config_posprocessing): st.session_state['config_posprocessing'] = self.config_posprocessing
                
                st.session_state['inputs_model'] = self.inputs_model
             
                
                st.session_state['outputs_model'] = model.run_model(st.session_state.df_dataset["df"],**st.session_state.inputs_model)
    
                    

    def __convert_outputs(self,  nome: str, type: str, text: str, value):
        """
        Método interno de conversão das informações de
        configurações para a interface do streamlit

        Parameters
        ----------
        type: type
            Representa o tipo da conversão
            podendo ser [DataFrame, plotly, str]
        text: str
            Texto de título da insersão
        value: Any
            Valor que será aplicado na função de display

        """
        st.markdown(f"##### {text}")
        if type == "DataFrame":
            st.dataframe(value)
            self.__download_output(nome,value)
        if type == "download":
            st.download_button(label=text, data=value)
        if type == "str":
            st.text(value)
        if type == "plot":
            st.pyplot(value)
        if type == "fig":
            st.pyplot(value)
        if type == "plotly":
            st.plotly_chart(value)
        if type == "model":
            self.__download_model(value)

    def container_output(self,state):
        """
        Bloco que converte outputs do método exec_button
        e os apresenta na interface com base no YAML de
        configs > parametros > output
        """
        for nome, valor in self.configs.parametros[state].items():
            self.__convert_outputs(
                nome,valor["tipo"], valor["descricao"], st.session_state[state][nome]
            )

    def container_posprocessing(self, func, model):
          
        with st.form("my_form"):
            self.inputs_posprocessing = func.container_posprocessing(st.session_state['inputs_model'],st.session_state['outputs_model'],st.session_state['config_posprocessing'])
            
            botao_posprocessing = st.form_submit_button("Executar")
                
            if(botao_posprocessing):
                st.session_state['outputs_posprocessing'] = model.posprocessing(**self.inputs_posprocessing)
                
    # @st.cache
    def __convert_df(self, df):
        """
        Metodo que converte df para csv
        """
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        # se for usar o cache, tem que hashear corretamente, comentei por enquanto
        return df.to_csv(encoding="utf-8")

    def __download_output(
        self, nome, df, file_name="output_csv.csv", label="Download CSV"
    ):
        csv = self.__convert_df(df)

        st.download_button(
            label=label,
            data=csv,
            file_name=file_name,
            mime="text/csv",
            key = nome
        )
    
    def __download_model(self, model):
        output_model = pickle.dumps(model)
        b64 = base64.b64encode(output_model).decode()
        # href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
        # st.markdown(href, unsafe_allow_html=True)
        st.download_button(
            label="Download PKL",
            data=b64,
            file_name="output_model.pkl",
            mime="application/octet-stream",
        )


def main(config_path: str, model_functions, config_posprocessing ):
    """
    Função compilada da interface Streamlit
    com base

    Parameters
    ----------
    exc_function: function
        Função de execução da automação
    config_path: str
        Caminho relativo da pasta de configurações
    """

    app = BASEAPP(path=config_path)
    
    app.header()
    
    app.container_dataset()
    
    if 'df_dataset' in st.session_state: 
        st.markdown("#### Data Preview") 
        with st.expander("See explanation"): 
            app.show_dataset()
        
        app.container_inputs(model_functions)
    
    if 'outputs_model' in st.session_state:
        st.markdown("#### Model Results")
        with st.expander("See explanation"):
            app.container_output('outputs_model')
            
    if 'config_posprocessing' in st.session_state: 
        if(config_posprocessing.check_posprocessing(st.session_state['config_posprocessing'])): 
            st.markdown("#### Analysis PosProcessing")
            with st.expander("See explanation"):
                app.container_posprocessing(config_posprocessing, model_functions)
                
                
                if 'outputs_posprocessing' in st.session_state:
                    app.container_output('outputs_posprocessing')
                    del st.session_state['outputs_posprocessing']
                    
    app.footer()
