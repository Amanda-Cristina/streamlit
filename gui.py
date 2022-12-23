from configs import load_configs
import streamlit as st
import pandas as pd
import pickle
import base64
from main import analysis as clusters_analysis
from main import classes as classes

import numpy as np


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

       
    def header(self, use_image=True):

        st.header(self.configs.nome)
        st.text(f"Versão: {self.configs.versao}")
        with st.expander("See explanation"):
            st.text(self.configs.descricao)

            if use_image:
                st.image("imgs/rascience2.png")

        

    def footer(self):
        """
        Sessão final da interface que carrega
        dados presentes no YAML como links_referencias
        """
        st.markdown(f"#### References \n")
        with st.expander("See all"):
            for nome, link in self.configs.links_referencias.items():
                st.markdown(f"- [{nome}]({link})")
            
            
    @st.cache(max_entries = 1)  
    def upload_csv(self):
        """
        Método interno para criar widget que recebe um DataFrame

        Parameters
        ----------
        title:str
            Nome do widget
        key:str
            Identificador unico do widget
        """
        #with st.sidebar: 
            #uploaded_file = st.file_uploader(title, key=key)
        if self.uploaded_file is not None:
            print('upload')
            # To read file as bytes:
            bytes_data = self.uploaded_file.getvalue()
        #st.text(f"Carregando DataFrame...")
            st.session_state['base'] = pd.read_csv(self.uploaded_file)

            
            #st.text(f"Shape: {df.shape}")
            #st.dataframe(df)
            #return df
    
    
    def show_dataset(self):
        with st.sidebar:
            self.uploaded_file = st.file_uploader("Escolha um arquivo CSV", key="Dowload")
        self.upload_csv()
        
        
        if 'base' in st.session_state:          
            st.markdown("#### Data Preview") 
            with st.expander("See explanation"):             
                st.text(f"Shape: {st.session_state.base.shape}")
                st.dataframe(st.session_state.base.head())
            return st.session_state.base
    
    def change_disable(self,control,change):
        
        if st.session_state[control]:
            if change in st.session_state:
                st.session_state[change] = 3
        
    def __convert_input(self, name, label, value, preset, mandatory,control):
        """
        Método interno de conversão das informações de
        configurações para a interface do streamlit

        Parameters
        ----------
        """
       
        if 'base' in st.session_state: 
            
            with st.sidebar:
                label = label+' *' if mandatory else label 
                if value == "date":
                    return st.text_input(label, value="" if preset is None else preset, key = name)
                if value == "int": 
                    if control is None:                   
                        return st.number_input(label, value= 1 if preset is None else preset, key = name)
                    else:
                        return st.number_input(label, value= 1 if preset is None else preset, key = name, disabled = False if st.session_state[control] else True)
                if value == "float":
                    return st.number_input(label, value=1.0 if preset is None else preset, key = name)
                if value == "str":
                    return st.text_input(label, value="" if preset is None else preset, key = name)
                if value == "column":
                    return st.selectbox(label,options=[] if 'base' not in st.session_state else st.session_state.base.columns, key = name)
                if value == "columns":
                    if control is None:
                        return st.multiselect(label,options=[] if 'base' not in st.session_state else st.session_state.base.columns, key = name)
                    else:
                        return st.multiselect(label,options=[] if 'base' not in st.session_state else st.session_state.base.columns, key = name, disabled = False if st.session_state[control] else True)
                if value == "list":
                    return st.text_input(label, value="" if preset is None else preset, key = name)
                if value == "bool":
                    return st.checkbox(label, value=False if preset is None else preset, key = name)
               
                

    def container_inputs(self):
        """
        Bloco que converte inputs do YAML de
        configs > parametros > input em widgets
        para o usuário utilizar.
        """

        self.inputs = {}
        self.widgets_analysis = {}
        
        for nome, opcoes in self.configs.parametros["input"].items():
            parametros = [nome,
                        opcoes["descricao"], 
                        opcoes["tipo"], 
                        opcoes["valor"] if "valor" in opcoes else None,
                        opcoes["obrigatorio"] if "obrigatorio" in opcoes else False,
                        opcoes["controle"] if "controle" in opcoes else None]
            if(nome == 'df'):
                self.inputs[nome] = self.show_dataset()
            elif ('analysis' in opcoes):
                self.widgets_analysis[nome] = self.__convert_input(*parametros)
            else: 
                self.inputs[nome] = self.__convert_input(*parametros)
      
                
    def container_inputs_analysis(self):
        self.inputs_analysis = {}
        columns_order = []
        columns_behavior = []
        if (self.widgets_analysis['control_importance_features']):
            st.markdown(f"#### Model Hierarchy \n")
            with st.expander("See all"):
                with st.form("my_form"):
                    col1, col2 = st.columns(2)
                    feature = 1
                    for input in self.widgets_analysis['importance_features']:
                        label = 'Feature' + str(feature)
                        column = col1.selectbox(label,options= self.widgets_analysis['importance_features'])
                        columns_order.append(column)
                        
                        behavior = col2.selectbox('Comportamento Crescente',options= [True, False], key = label)
                        columns_behavior.append(behavior)
                        feature += 1
                    self.inputs_analysis['columns_order'] = columns_order
                    self.inputs_analysis['columns_behavior'] = columns_behavior
                    if (self.widgets_analysis['control_custom_features']):
                        self.inputs_analysis['classes_name'] = self.container_inputs_categorys()
                    else:
                        self.inputs_analysis['classes_name'] = None
                    submitted = st.form_submit_button("Submit")
                    
                if(submitted ):
                    st.text(f"Executando...")
                    st.session_state['outputs_analysis'] = clusters_analysis(st.session_state['outputs']['clustering_raw'],st.session_state['inputs']['colunas_caracteristicas'],**self.inputs_analysis)
                    self.container_output('outputs_analysis')
        
        elif(self.widgets_analysis['control_custom_features']):
            print("só classe")
            st.markdown(f"#### Model Hierarchy \n")
            with st.expander("See all"):
                with st.form("my_form"):
                    self.inputs_analysis['classes_name'] = self.container_inputs_categorys()
                    submitted = st.form_submit_button("Submit")
                    
                if(submitted ):
                    st.text(f"Executando...")
                    st.session_state['outputs_analysis'] = classes(st.session_state['outputs']['clustering_raw'],st.session_state['inputs']['colunas_caracteristicas'],**self.inputs_analysis)
                    self.container_output('outputs_analysis')
                        
            
    def container_inputs_categorys(self):   
        classes_name = []
        st.markdown(f"#### Custom Classes \n")
        clusters = np.sort(st.session_state['outputs']['clustering_raw']['clusters'].unique())
        for cluster in clusters:
            label =  'Cluster ' + str(cluster)
            classe = st.text_input(label, key = label)
            classes_name.append(classe)
        return classes_name  

        
    def exec_button(self, exc_function):
        """
        Botão de execução da função main que preenche
        o atributo self.outputs

        Parameters
        ----------
        exc_function: function
            Função que receberá parametros do self.inputs
        """
        with st.sidebar:
            exc_botao = st.button("Executar", help="Vai dar bom, confia")
            if exc_botao:
                st.text(f"Executando...")
                st.session_state['inputs'] = self.inputs
                st.session_state['outputs'] = exc_function(**self.inputs)
                
    def __convert_outputs(self, nome: str, type: str, text: str, value):
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
        
        st.markdown(f"#### {text}")
        
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
    
    # @st.cache
    def __convert_df(self, df):
        """
        Metodo que converte df para csv
        """
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        # se for usar o cache, tem que hashear corretamente, comentei por enquanto
        return df.to_csv(encoding="utf-8")
    
                    
    def __download_output(
        self,nome, df, file_name="output_csv.csv", label="Download CSV"
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


def main(exc_function, config_path: str):
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
    
    
    
    app.container_inputs()
    if 'base' in st.session_state:
        app.exec_button(exc_function=exc_function)
        
    # Validando se existe atributo output
    if 'outputs' in st.session_state:
        
        st.markdown("#### Results")
        with st.expander("See explanation"):
            app.container_output('outputs')
        

        app.container_inputs_analysis()
        
           
        
        app.footer()  
    
        
    
#if value == "int":return self.form.number_input(label, value=1 if preset is None else preset, key='number', disabled= True if self.inputs is None else not self.inputs[control] )

#on_change = self.change_disable, args=(name,control)