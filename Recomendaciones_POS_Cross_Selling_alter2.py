#recomendaciones
from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix,coo_matrix, hstack
#generales
import scipy
from ast import literal_eval
from scipy.sparse import vstack
from scipy import sparse
from scipy.sparse import load_npz
import csv
import numpy as np
import pandas as pd
import polars as pl
import pickle   
import os
from tqdm import tqdm
import concurrent.futures
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from joblib import Parallel, delayed
import multiprocessing
from functools import partial
from nvrecommender.data import Interactions
from nvrecommender.models import WideDeep
from nvrecommender.train import Trainer
import nvtabular as nvt
import nvrecommender as nvr

class clase_recomendadores:

    def inferencias_pos_checkout(self,recomendador='pos_checkout',cliente=None,productos=None
                                ,interaction_matrix='INTERACTIONS_POS_202401.npz'
                                ,name_user_features_matrix='user_features_INTERACTIONS_POS_202401.npz'
                                ,name_item_features_matrix='item_features_INTERACTIONS_POS_202401.npz'
                                ,factor_pond_item=1,nro_recos=3,num_threads=1,name_model='RECOS_POS_202401.pkl',name_dataset='dataset_pos_checkout.pkl'):
        """Método de generación de inferencias para Recomendacion por Post-Compra
        Args:
            cliente (_int_): cliente al que se realizará la inferencia.
            productos (_list_): productos que el cliente está comprando.
            interaction_matrix (_str_): nombre de la matriz de interacciones.
            name_user_features_matrix (_str_): nombre de la matriz de caracteristicas de usuario.
            name_item_features_matrix (_str_): nombre de la matriz de caracteristicas de items.
            factor_pond_item (_int_): Factor de ponderación que se le otorga a los items que se están llevando por sobre las compras historicas.
            nro_recos (_int_): Cantidad de recomendaciones a generar por cada cliente.
            name_model (_str_): nombre del modelo entrenado.
            name_dataset (_str_): nombre del dataset del modelo.
        """

        user_features_matrix = load_npz(name_user_features_matrix)
        item_features_matrix = load_npz(name_item_features_matrix)
        interaction_matrix = load_npz(interaction_matrix)
        products=pd.read_csv('products_'+recomendador+'.csv').products.tolist()
        users=pd.read_csv('customers_'+recomendador+'.csv').customers.tolist()
        skus_farma=pd.read_csv('skus_farma.csv').sku_farma.tolist()
        exclusiones=pd.read_csv('exclusiones_'+recomendador+'.csv').exclusiones.tolist()

        ############ MUESTREO ############

        interacciones_csr = interaction_matrix.tocsr()

        # Proceso de muestreo
        row, col = interaction_matrix.nonzero()
        sample_size = int(len(row) * 0.02)
        indices = np.random.choice(len(row), sample_size, replace=False)

        # Usamos los índices muestreados para extraer los datos
        sampled_row = row[indices]
        sampled_col = col[indices]
        sampled_data = np.array([interacciones_csr[i, j] for i, j in zip(sampled_row, sampled_col)])

        # Crear la nueva matriz sparse muestreada
        interaction_matrix = csr_matrix((sampled_data, (sampled_row, sampled_col)), shape=interaction_matrix.shape)

        ##################################


        with open(name_model, "rb") as file:
            model = pickle.load(file)
        with open(name_dataset, "rb") as file:
            dataset = pickle.load(file)    

        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

        dicc_cli_char=pd.read_csv('SR_CARACTERISTICAS.csv',sep=';',encoding='latin')
        dicc_prod_char=pd.read_csv('diccionario_productos.csv',sep=';',usecols=['CODIGO','UNIDAD_COMERCIAL','MACROCATEGORIA','CATEGORIA_NIVEL_1','CATEGORIA_NIVEL_2_FCV','DESCRIPTOR_LARGO','MARCA'])
        car_cli=dicc_cli_char[dicc_cli_char.RUT_CLIENTE==cliente][['GENERO']]
        car_prod=dicc_prod_char[dicc_prod_char.CODIGO==productos[0]][['MARCA']]

        #Si cliente no existe se incorpora el cliente/caracteristica a la matriz de usuarios
        if cliente not in users:
            new_user_features_matrix = coo_matrix((1, user_features_matrix.shape[1])) #vacia (1 = tamaño filas = n° usuarios)
            for var in car_cli.columns: # por cada caracteristica del cliente
                gender_feature_index = user_feature_map[var+':'+car_cli[var][0]] # Mapea el Indice asociado al valor de la caracteristica ingresada
                data = [1.0]
                row = [0]  # Índice de fila para el nuevo usuario
                col = [gender_feature_index]    # Índice de columna para la característica ej: 'GENERO:FEM'
                new_user_features_matrix += coo_matrix((data, (row, col)), shape=new_user_features_matrix.shape) # Insertar la información en la matriz
            combined_user_features_matrix = vstack([user_features_matrix, new_user_features_matrix]) 
            user_ids=max(user_id_map.values())+1
            user_id=user_ids
        else:
            combined_user_features_matrix = user_features_matrix
            user_ids=max(user_id_map.values())
            user_id=users.index(cliente)

        #Si producto no existe se incorpora el producto/caracteristica a la matriz de items
        if productos not in products:
            new_item_features_matrix = coo_matrix((1, item_features_matrix.shape[1])) #vacia (1 = tamaño filas = n° usuarios)

            self.item_features_matrix=item_features_matrix
            self.new_item_features_matrix=new_item_features_matrix

            for var in car_prod.columns: # por cada caracteristica del cliente
                marca_feature_index = item_feature_map[var+':'+car_prod[var].item()] # Mapea el Indice asociado al valor de la caracteristica ingresada
                data = [1.0] # factor de ponderacion para el nuevo item incorporado (en literal corresponde al numero de items del dataset inicialmente conformado)
                row = [0]  # Índice de fila para el nuevo producto
                col = [marca_feature_index]    # Índice de columna para la característica
                new_item_features_matrix += coo_matrix((data, (row, col)), shape=new_item_features_matrix.shape) # Insertar la información en la matriz    
            combined_item_features_matrix = vstack([item_features_matrix, new_item_features_matrix]) 
            item_ids=np.arange(len(products)+1)
            item_id=len(products) # el ultimo index anterior a este nuevo es el len()-1 ya que parten desde 0
            products=products+productos
        else:
            combined_item_features_matrix = item_features_matrix
            item_ids=np.arange(len(products))
            item_id=products.index(productos)

        #Añadir cliente/producto a las interacciones
        interaction_matrix = csr_matrix(interaction_matrix)
        if productos not in products: #si se agregó un nuevo producto, se añade nueva columna correspondiente a nuevo producto
            new_columns = coo_matrix((interaction_matrix.shape[0], 1), dtype=np.float32)
            interaction_matrix = hstack([interaction_matrix, new_columns]) 
        
        mask = (interaction_matrix.row == user_id) & (interaction_matrix.col == item_id) # Crea una máscara para encontrar la interacción existente
        if mask.sum() > 0: # interacción usuario/item ya existe; sumar "n" interacciones adicionales
            interaction_matrix.data[mask] += int(factor_pond_item)
        else: # La interacción no existe (tupla usuario/producto), agregar una nueva    
            interaction_matrix.row = np.append(interaction_matrix.row, user_id)
            interaction_matrix.col = np.append(interaction_matrix.col, item_id)
            interaction_matrix.data = np.append(interaction_matrix.data, int(factor_pond_item))
        
        





        interactions = Interactions(user_id, item_ids, combined_user_features_matrix, combined_item_features_matrix)

        model = nvr.WideDeep()
        trainer = Trainer(
            model,
            loss="bpr",  # Puedes elegir una función de pérdida adecuada según tu problema
            optimizer="adam",  # Elige un optimizador apropiado
            lr=0.001,  # Tasa de aprendizaje
            num_epochs=10,  # Número de épocas de entrenamiento
            batch_size=64,  # Tamaño del lote (batch)
            device="cuda",  # Usar GPU si está disponible
        )

        # Entrenar el modelo
        trainer.fit(interactions)
        
        scores = trainer.predict(user_id, nro_recos)




        # model.fit_partial(interactions=interaction_matrix
        #                  ,user_features=combined_user_features_matrix
        #                  ,item_features=combined_item_features_matrix
        #                  ,num_threads=num_threads)

        # params = {"user_ids": user_id #corresponde al indice de usuario consultado
        #          ,"item_ids": item_ids #Corresponde al array completo de indices de productos
        #          ,"user_features": combined_user_features_matrix
        #          ,"item_features": combined_item_features_matrix}
        
        # scores = model.predict(**params)

        ###### PROCESO DE PRIORIZACIÓN DE PRODUCTOS
        top_items=pd.DataFrame({'CODIGO_PRODUCTO': products}).iloc[np.argsort(-scores)]
        top_items['index']=top_items.index
        #top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(df_muestra_usuario)) #Excluir productos comprados
        top_items=top_items[~top_items['CODIGO_PRODUCTO'].isin(skus_farma)]
        self.top_items=top_items[~top_items['CODIGO_PRODUCTO'].isin(exclusiones)]
        top_items=top_items.CODIGO_PRODUCTO[:nro_recos].tolist()


        self.interaction_matrix=interaction_matrix
        self.user_id=user_id
        self.user_ids=user_ids
        self.item_id=item_id
        self.item_ids=item_ids
        self.products=products
        
        return top_items    
