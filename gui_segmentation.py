import streamlit as st
import pandas as pd
import numpy as np


class SEGMENTATION:
    def __init__(self) -> None:
        """
        Inicialização do objeto APP

        path: str
            Caminho do arquivo YAML de configurações
        """
            
    
    def check_posprocessing(self, config):
        return True if (config['control_importance_features'] or config['control_custom_features']) else False
    
    def container_posprocessing(self, inputs, outputs, config):

        inputs_posprocessing = {}
        
        inputs_posprocessing['df'] = outputs['clustering_raw']
        inputs_posprocessing['coluna_id'] = inputs['coluna_id']
        inputs_posprocessing['colunas_caracteristicas'] = inputs['colunas_caracteristicas']
        inputs_posprocessing['columns_order'] = []
        inputs_posprocessing['columns_behavior'] = []
        inputs_posprocessing['classes_name']  = []
         
        def set_clusters_importance():

            col1, col2 = st.columns(2)
            feature = 1
            while feature <= len(config['importance_features']):
                label = 'Feature' + str(feature)
                column = col1.selectbox(label,options= config['importance_features'])
                inputs_posprocessing['columns_order'].append(column)
                
                behavior = col2.selectbox('Comportamento Crescente',options= [True, False], key = label)
                inputs_posprocessing['columns_behavior'].append(behavior)
                feature += 1  

        def set_clusters_categories():   
            clusters = np.sort(outputs['clustering_raw']['clusters'].unique())
            for cluster in clusters:
                label =  'Cluster ' + str(cluster)
                classe = st.text_input(label, key = label)
                inputs_posprocessing['classes_name'].append(classe)
        

        
        if (config['control_importance_features']):
            st.markdown("##### Custom Classes")
            set_clusters_importance()
            
            
        if (config['control_custom_features']):
            st.markdown("##### Importance Features")
            set_clusters_categories()
        
        return inputs_posprocessing

               
      