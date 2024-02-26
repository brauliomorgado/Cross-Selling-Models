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

class clase_recomendadores:

    def carga_data(self,recomendador,base_transacciones,base_navegaciones=None,bioeq_oficial=None,bioeq_api=None,diccionario=None,bskt_rules=None,n_partes=1,parte_ejecutar=1,user_features=[],item_features=[],test=False):
        """Método que carga datos 
        Args:
            recomendador (_str_): Nombre del recomendador que se desea utilizar.
                                  Posibles valores: 'cliente_antiguo','navegacion', 'postcompra'.
            base_transacciones (_str_): Nombre y extensión (.csv) del archivo que contiene los datos transaccionales para modelar y/o inferir. Debe explicitar la extension del archivo.
            base_navegaciones (_str_)(opcional): Nombre y extensión (.csv) del archivo con datos de navegación. Requerido si recomendador es 'navegacion' o 'postcompra'.
            bioeq_oficial (_str_)(opcional): Nombre y extensión (.csv) del archivo con datos oficiales de bioequivalencia. Requerido si recomendador es 'navegacion'.
            bioeq_api (_str_)(opcional): Nombre y extensión (.csv) del archivo con datos de bioequivalencia desde API. Requerido si recomendador es 'navegacion'.
            diccionario (_str_): Nombre y extensión (.csv) del archivo que contiene la información de productos. Debe explicitar la extension del archivo.
            bskt_rules (_str_)(opcional): Nombre y extensión (.csv) del archivo con reglas de asociación para canasta de compras. Requerido si recomendador es 'navegacion' o 'postcompra'.
            n_partes (_float_)(opcional): Número de fragmentos en que se divide el dataset para su ejecución. Valor por defecto es 1. Si es menor a 1, se divide en 2 partes según porcentajes especificados.
            parte_ejecutar (_int_)(opcional): Índice del fragmento del dataset a cargar en caso de división. Se cuenta desde 1. Relevante solo si n_partes es mayor a 1 o fracción.
            features (_list_)(opcional): Lista de características adicionales a considerar en el modelo.
        """
        self.user_features=user_features
        self.item_features=item_features
        self.df=pd.read_csv(base_transacciones,sep=';',encoding='latin')
        self.df['SUM_NRO_PROD'] = self.df['SUM_NRO_PROD'].fillna(1.0)
        self.diccionario=pd.read_csv(diccionario,sep=';',usecols=['CODIGO','UNIDAD_COMERCIAL','MACROCATEGORIA','CATEGORIA_NIVEL_1','CATEGORIA_NIVEL_2_FCV','DESCRIPTOR_LARGO','MARCA'])
        self.diccionario.columns=['CODIGO_PRODUCTO','UNIDAD_COMERCIAL','MACROCATEGORIA','CATEGORIA_NIVEL_1','CATEGORIA_NIVEL_2_FCV','DESCRIPTOR','MARCA']
        self.df=pd.merge(self.df,self.diccionario[['CODIGO_PRODUCTO','DESCRIPTOR','CATEGORIA_NIVEL_1','CATEGORIA_NIVEL_2_FCV','MACROCATEGORIA','MARCA']],on='CODIGO_PRODUCTO',how='left')
        self.df['RUT_CLIENTE']=((self.df['SK']+1599)/2).astype(int)
        self.caracteristicas=pd.read_csv('SR_CARACTERISTICAS.csv',sep=';',encoding='latin')
        if any(item in self.user_features for item in self.caracteristicas.columns.to_list()[1:]): #Busca features requeridas en tabla de caracteristicas de clientes, luego las incorpora al df
            add_user_features=[item for item in user_features if item in self.caracteristicas.columns.to_list()[1:]] #agrega las que existen en tabla demograficas
            add_user_features=[item for item in add_user_features if item not in self.df.columns.to_list()[1:]] #excluye las que ya existen en el df
            self.df=pd.merge(self.df,self.caracteristicas[['RUT_CLIENTE']+add_user_features],on='RUT_CLIENTE',how='left')

        if n_partes>1: # Si se divide en "x" partes dado un numero establecido
            ruts_unicos = self.df['RUT_CLIENTE'].unique()
            ruts_divididos = np.array_split(ruts_unicos, n_partes) #realiza la particion del dataframe
            self.df = [self.df[self.df['RUT_CLIENTE'].isin(ruts_divididos[i])] for i in range(n_partes)][parte_ejecutar-1] #almacena solo la parte correspondiente a ejecutar

        if n_partes<1: # Si se divide en 2 partes en forma de porcentaje
            partes_ruts=[]
            cut_point = int(len(self.df['RUT_CLIENTE'].unique()) * n_partes) #n_partes --> representa un porcentaje
            partes_ruts.append(self.df['RUT_CLIENTE'].unique()[:cut_point])
            partes_ruts.append(self.df['RUT_CLIENTE'].unique()[cut_point:])
            self.df = self.df[self.df['RUT_CLIENTE'].isin(partes_ruts[parte_ejecutar-1])]

        if recomendador in ('navegacion', 'postcompra'):
            if recomendador=='navegacion':
                self.switch_bioeq=pl.read_csv(bioeq_oficial,separator=';',encoding='UTF-8')
                self.switch_bioeq_ecomm=pl.read_csv(bioeq_api,separator=';',encoding='UTF-8')
            self.ucom_macro=self.diccionario[['UNIDAD_COMERCIAL','MACROCATEGORIA']].drop_duplicates()
            self.macros_farma=list(self.ucom_macro[self.ucom_macro.UNIDAD_COMERCIAL=='FARMA'].MACROCATEGORIA)

            self.df_nav=pd.read_csv(base_navegaciones,sep=';')

            self.df_nav['SUM_NRO_PROD']=1
            if recomendador=='postcompra':
                self.df_nav['SUM_NRO_PROD']=self.df_nav['CANTIDAD_DISTRIBUIDA']
            self.df_nav['SK']=(self.df_nav['RUT_CLIENTE']*2)-1599
            #self.df_ucom=list(self.diccionario[self.diccionario.UNIDAD_COMERCIAL.isin(['BELLEZA Y CUIDADO PERSONAL','B Y CP','BIENESTAR Y CONSUMO','WELLNESS'])].CODIGO_PRODUCTO) #no incluye farma
            self.df_ucom=list(self.diccionario.CODIGO_PRODUCTO) #incluye farma
            self.df_nav=self.df_nav[self.df_nav.CODIGO_PRODUCTO.isin(self.df_ucom)]
            self.df_hist=self.df[(self.df.CODIGO_PRODUCTO.isin(self.df_ucom)) & (~self.df.RUT_CLIENTE.isin(list(self.df_nav.RUT_CLIENTE.unique())))] #no se considera histórico de compra de clientes que navegaron para asi asociar exclusivamente en base a su navegación
            self.df=pd.concat([self.df_hist,self.df_nav],axis=0)

            tuples_hist = list(zip(self.df_hist["RUT_CLIENTE"], self.df_hist["CODIGO_PRODUCTO"],self.df_hist["SUM_NRO_PROD"]))
            tuples_nav = list(zip(self.df_nav["RUT_CLIENTE"], self.df_nav["CODIGO_PRODUCTO"],self.df_nav["SUM_NRO_PROD"]))
            pd.DataFrame({'tuples_hist': tuples_hist }).to_csv('tuples_hist_'+recomendador+'.csv', index=False)   
            pd.DataFrame({'tuples_nav': tuples_nav }).to_csv('tuples_nav_'+recomendador+'.csv', index=False)

            self.df_nav = pl.from_pandas(self.df_nav)
            self.cat_rel_nav=pd.read_csv(bskt_rules,sep=';',encoding='latin',decimal=',')
            self.cat_rel_nav=pl.from_pandas(self.cat_rel_nav[['antecedents','consequents','lift']])   
        
        if recomendador in ('cliente_antiguo', 'pos_checkout'):
            tuples = list(zip(self.df["RUT_CLIENTE"], self.df["CODIGO_PRODUCTO"],self.df["SUM_NRO_PROD"]))
            pd.DataFrame({'tuples': tuples }).to_csv('tuples_'+recomendador+'.csv', index=False)

        self.user_features_values=[]
        for var_dem in user_features:
            self.df[var_dem] = self.df[var_dem].fillna("DESCONOCIDO")
            self.df[var_dem] = self.df[var_dem].replace({'Missing': 'DESCONOCIDO'})
            self.user_features_values.extend([f'{var_dem}:{val}' for val in self.df[var_dem].unique()])
        if test==False:
            pd.DataFrame({'user_features_values': self.user_features_values }).to_csv('user_features_values_'+recomendador+'.csv', index=False)

        self.item_features_values=[]
        for item_car in item_features:
            self.df[item_car] = self.df[item_car].fillna("DESCONOCIDO")
            self.df[item_car] = self.df[item_car].replace({'Missing': 'DESCONOCIDO'})
            self.item_features_values.extend([f'{item_car}:{val}' for val in self.df[item_car].unique()])
        if test==False:    
            pd.DataFrame({'item_features_values': self.item_features_values }).to_csv('item_features_values_'+recomendador+'.csv', index=False)    

        self.df = pl.from_pandas(self.df)
        self.dicc = pl.from_pandas(self.diccionario)
        self.mcfma=self.dicc.filter(self.dicc['UNIDAD_COMERCIAL']=='FARMA')['CODIGO_PRODUCTO'].to_list() #PRODUCTOS FARMA
        pd.DataFrame({'sku_farma': self.mcfma }).to_csv('skus_farma.csv', index=False)

        self.users = self.df["RUT_CLIENTE"].unique().to_list()
        self.items_cods = np.array(self.df["CODIGO_PRODUCTO"].unique())
        self.n_items = len(self.items_cods)
        self.user_dict = {user: i for i, user in enumerate(self.users)}
        if test==False:
            pd.DataFrame({'items_cods': self.items_cods }).to_csv('products_model_'+recomendador+'.csv', index=False)
            pd.DataFrame({'users_dni': self.users }).to_csv('users_model_'+recomendador+'.csv', index=False)

        self.customers = self.df["RUT_CLIENTE"].unique().cast(int).to_list()
        self.products = self.df["CODIGO_PRODUCTO"].unique().cast(int).to_list()
        if test==False:    
            pd.DataFrame({'customers': self.customers }).to_csv('customers_'+recomendador+'.csv', index=False)
            pd.DataFrame({'products': self.products }).to_csv('products_'+recomendador+'.csv', index=False)

        user_car_unique=self.df.to_pandas()[['RUT_CLIENTE']+self.user_features].drop_duplicates()
        item_car_unique=self.df.to_pandas()[['CODIGO_PRODUCTO']+self.item_features].drop_duplicates()
        if test==False:    
            user_car_unique.to_csv('user_car_unique_'+recomendador+'.csv', index=False)
            item_car_unique.to_csv('item_car_unique_'+recomendador+'.csv', index=False)

        self.n_partes=n_partes
        self.parte_ejecutar=parte_ejecutar
    
        return None

    def exclusiones(self,archivo_exclusiones,recomendador='cliente_antiguo'):
        """Método que define los SKU's a excluir como posibles recomendaciones
        Args:
            archivo_exclusiones (_str_): Nombre y extensión (.csv) del archivo externo que contenga listado de SKU's requeridos de excluir como regla dura.
        """    
        #SE IDENTIFICAN EXCLUSIONES:
        #por descriptor
        self.excl_otros = [str(elemento) for elemento in list(self.diccionario[self.diccionario.DESCRIPTOR.isin(['DESPACHO','DESP.A','RETIRO DE RECETA','PROMOCION DTE','HORA VACUNACION','INSUMOS BOLSAS','SIN DESCRIPTOR','AJUSTE POR REDONDEO'])]['CODIGO_PRODUCTO'])] +\
                          [str(elemento) for elemento in list(self.diccionario[self.diccionario.DESCRIPTOR.str.contains('BOLSA PAPEL|BOLS|DESPACHO|PROMOCION|CUPON|INYECCION|MEDICION|MEMBRESIA|REGALO|2020|2021|2022|MUNDIAL')]['CODIGO_PRODUCTO'])] +\
                          [str(elemento) for elemento in list(self.diccionario[self.diccionario.MACROCATEGORIA.isin(['ENFERMERIA ACCESORIOS Y PROCEDIMIENTOS','HOGAR','RECARGAS VARIAS','BEBIDAS Y AGUAS','ACCESORIOS MODA','TECNOLOGIA','AUTOLIQUIDABLES REGALO ARTICULOS VARIOS','INSUMOS MACRO','EQUIPOS QUIRURGICOS','POR CLASIFICAR'])]['CODIGO_PRODUCTO'])] +\
                          [str(elemento) for elemento in list(self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.isin(['MASCARILLAS PRIMEROS AUXILIOS'])]['CODIGO_PRODUCTO'])] +\
                          [str(elemento) for elemento in list(self.diccionario[self.diccionario.UNIDAD_COMERCIAL.isin(['INSTITUCIONAL'])]['CODIGO_PRODUCTO'])]
        #por código
        cods_excluir = pd.read_csv(archivo_exclusiones,sep=';').excluir.values.tolist() + self.excl_otros
        self.cods_excluir = [int(cod) for cod in cods_excluir]
        
        if recomendador=='pos_checkout':
            excl_pos = [str(elemento) for elemento in list(self.diccionario[self.diccionario.DESCRIPTOR.isin([''])]['CODIGO_PRODUCTO'])] +\
                       [str(elemento) for elemento in list(self.diccionario[self.diccionario.DESCRIPTOR.str.contains('')]['CODIGO_PRODUCTO'])] +\
                       [str(elemento) for elemento in list(self.diccionario[self.diccionario.MACROCATEGORIA.isin([''])]['CODIGO_PRODUCTO'])] +\
                       [str(elemento) for elemento in list(self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.isin(['DISFUNCION ERECTIL'])]['CODIGO_PRODUCTO'])]
            self.cods_excluir=self.cods_excluir+excl_pos
        pd.DataFrame({'exclusiones': self.cods_excluir }).to_csv('exclusiones_'+recomendador+'.csv', index=False)

        #CATEGORIAS PREREQUISITO DE COMPRA
        self.prereq_compra = self.diccionario[self.diccionario.MACROCATEGORIA.isin(['HIGIENE ADULTO','DEPORTE','ACCESORIOS INFANTILES','TEXTIL INFANTIL','CUIDADO BEBE DERMO','ALIMENTOS INFANTIL','TOCADOR INFANTIL','JUGUETERIA','MAQUILLAJE','COLORACION','DEPILACION','DISPOSITIVOS ORGANOS DE LOS SENTIDOS','PAÑALES DESECHABLES',
                                                                                    'AUTOLIQUIDABLES REGALO INFANTIL','AUTOLIQUIDABLES REGALO BELLEZA','JERINGAS','SONDAS'])].CODIGO_PRODUCTO.to_list() +\
                             self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.isin(['ACCESORIOS PROTESIS'])].CODIGO_PRODUCTO.to_list() +\
                             self.diccionario[self.diccionario.CATEGORIA_NIVEL_2_FCV.isin([''])].CODIGO_PRODUCTO.to_list() +\
                             self.diccionario[self.diccionario.MACROCATEGORIA.str.contains('MASCOTA|otraspalabrasqueagregar')].CODIGO_PRODUCTO.to_list() +\
                             self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.str.contains('MASCOTA|ACCESORIOS PROTESIS')].CODIGO_PRODUCTO.to_list() +\
                             self.diccionario[self.diccionario.CATEGORIA_NIVEL_2_FCV.str.contains('MASCOTA|otraspalabrasqueagregar')].CODIGO_PRODUCTO.to_list() +\
                             self.diccionario[self.diccionario.DESCRIPTOR.str.contains('MASCOTA|otraspalabrasqueagregar')].CODIGO_PRODUCTO.to_list()

        #EXCLUSIONES POR GENERO
        #incorporar prereq hombre MACROS: DEPILACION,COLORACION
        self.excl_mas = self.diccionario[self.diccionario.MACROCATEGORIA.isin(['ACCESORIOS BELLEZA','PROTECCION SANITARIA FEMENINA','BELLEZA Y SPA','SPA','MAQUILLAJE','MAQUILLAJE DERMO','ELECTRO CAPILAR'])].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.MACROCATEGORIA.str.contains('MUJER|FEMENINA|FEMENINO')].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.isin(['DEO MUJER','LACA FIJADORES','FRAGANCIAS MUJER','MOUSSE Y ESPUMAS'])].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.str.contains('MUJER|FEMENINA|FEMENINO')].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_2_FCV.isin(['UÑAS POSTIZAS'])].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_2_FCV.str.contains('MUJER|FEMENINA|FEMENINO')].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.DESCRIPTOR.str.contains('MUJER|WOMAN|WOMEN|MASCARA|FEMENINA|FEMENINO')].CODIGO_PRODUCTO.to_list()
        self.excl_fem = self.diccionario[self.diccionario.MACROCATEGORIA.isin(['CUIDADO MASCULINO DERMO'])].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.MACROCATEGORIA.str.contains('HOMBRE|MASCULINO')].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.isin(['FRAGANCIAS HOMBRE','GEL STYLING','ACCESORIOS BARBERIA'])].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_1.str.contains('HOMBRE|MASCULINO')].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_2_FCV.isin(['SHAMPOO ANTICAIDA DERMO'])].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.CATEGORIA_NIVEL_2_FCV.str.contains('HOMBRE|MASCULINO')].CODIGO_PRODUCTO.to_list() +\
                        self.diccionario[self.diccionario.DESCRIPTOR.str.contains('HOMBRE| MEN |MASCULINO')].CODIGO_PRODUCTO.to_list() 

    def define_stop_words(self,):
        # SE CONFORMAN STOPWORDS PARA UTILIZARLAS POSTERIORMENTE EN UN BLOQUE DE CODIGO QUE VERIFICA QUE LAS RECOMENDACIONES NO REPITAN TEXTOS CON MÁS DE 2 PALABRAS IGUALES
        self.stop_words = set(stopwords.words('spanish'))
        self.patron = r'\b[xX]\d+\b'
        stopwords_adicionales = ['.','x','-','UNIDADES','UNID']
        self.stop_words.update(stopwords_adicionales)  
        return None
    
    def interaction_feature_matrix(self,recomendador='cliente_antiguo',name_output_interaction_matrix='interactions_navegaciones.npz',model_feature=False):
        """Método de generación de la matriz de interacciones
        Args:
            recomendador (_str_): Nombre del recomendador que se desea utilizar.
                                  Posibles valores: 'cliente_antiguo','navegacion', 'postcompra'.
            name_output_interaction_matrix (_str_): Nombre y extensión (.npz) que se desea dar al archivo con las interacciones.
            model_feature (Boolean): Indica si se utilizarán las características del modelo (True) o no (False).
        """    
        self.customers=pd.read_csv('customers_'+recomendador+'.csv').customers.tolist()
        self.products=pd.read_csv('products_'+recomendador+'.csv').products.tolist()
        user_car_unique=pd.read_csv('user_car_unique_'+recomendador+'.csv')
        item_car_unique=pd.read_csv('item_car_unique_'+recomendador+'.csv')
        self.user_features_values=pd.read_csv('user_features_values_'+recomendador+'.csv').user_features_values.tolist()
        self.item_features_values=pd.read_csv('item_features_values_'+recomendador+'.csv').item_features_values.tolist()

        self.name_output_interaction_matrix=name_output_interaction_matrix
        dataset = Dataset()

        params = {"users": self.customers, "items": self.products}
        if model_feature==True:
            params["user_features"] = self.user_features_values
            params["item_features"] = self.item_features_values

        dataset.fit(**params)
 
        self.user_features_list = []
        for index, row in user_car_unique.iterrows():
            cliente_id = row['RUT_CLIENTE']
            cliente_vars = [f'{col}:{row[col]}' for col in user_car_unique.columns if col != 'RUT_CLIENTE']
            self.user_features_list.append((cliente_id, cliente_vars))
        self.item_features_list = []
        for index, row in item_car_unique.iterrows():
            item_id = row['CODIGO_PRODUCTO']
            item_vars = [f'{col}:{row[col]}' for col in item_car_unique.columns if col != 'CODIGO_PRODUCTO']
            self.item_features_list.append((item_id, item_vars))

        if model_feature==True:
                self.user_features_matrix = dataset.build_user_features(self.user_features_list)
                self.item_features_matrix = dataset.build_item_features(self.item_features_list)
                sparse.save_npz('user_features_'+name_output_interaction_matrix, self.user_features_matrix) #user features historico
                sparse.save_npz('item_features_'+name_output_interaction_matrix, self.item_features_matrix) #item features historico    
                
        if recomendador in ('navegacion', 'postcompra'):
            tuples_hist=pd.read_csv('tuples_hist_'+recomendador+'.csv')
            tuples_hist = [literal_eval(tupla) for tupla in tuples_hist.iloc[:, 0]]
            tuples_nav=pd.read_csv('tuples_nav_'+recomendador+'.csv')
            tuples_nav = [literal_eval(tupla) for tupla in tuples_nav.iloc[:, 0]]

            self.df_coo_hist_not_weight, self.df_coo_hist = dataset.build_interactions(tuples_hist)
            self.df_coo_nav_not_weight, self.df_coo_nav = dataset.build_interactions(tuples_nav) 
            sparse.save_npz('hist_'+name_output_interaction_matrix, self.df_coo_hist) #interaction historico

            view_weights = np.ones_like(self.df_coo_nav.data) * 2  # Ponderación (x2) de los pesos asociados a productos vistos en la navegación
            self.df_coo_nav.data = view_weights  
            sparse.save_npz('nav_'+name_output_interaction_matrix, self.df_coo_nav)  #interaction navegaciones periodico   
             
        else: #recommender=='cliente_antiguo'
            tuples=pd.read_csv('tuples_'+recomendador+'.csv')
            tuples = [literal_eval(tupla) for tupla in tuples.iloc[:, 0]]

            self.df_coo_not_weight, self.df_coo = dataset.build_interactions(tuples)
            sparse.save_npz(name_output_interaction_matrix, self.df_coo)

        with open('dataset_'+recomendador+'.pkl', 'wb') as file:
            pickle.dump(dataset, file)


    def train_cliente_antiguo(self,name_output_model,name_interaction_matrix='INTERACTIONS_THCN_202401.npz',model_feature=False,epoch=20,num_threads=12,user_alpha=0.0,item_alpha=0.0,random_state=0):
        """Método para el entrenamiento utilizado para Perfil de Compra
        Args:
            name_output_model (_str_): Nombre y extensión (.pkl) que se desea dar al archivo con el modelo entrenado.
            epoch (_int_): Numero de veces que se desea iterar el conjunto de datos en el algoritmo.
            num_threads (_int_) = Numero de hilos disponibles para el entrenamiento
        """
        df_coo = sparse.load_npz(name_interaction_matrix)
        model = LightFM(loss='warp',user_alpha=user_alpha,item_alpha=item_alpha,random_state=random_state)
        if model_feature==False:
            model.fit(df_coo, epochs=epoch, num_threads=num_threads)
        else: # Utiliza Demografica
            model.fit(df_coo, user_features=self.user_features_matrix, item_features=self.item_features_matrix,
            epochs=epoch, num_threads=num_threads) 

        #EXPORTAR DF A PICKLE
        with open(name_output_model, "wb") as f:
            pickle.dump(model, f)
        return None
    

    def train_navegaciones_postcompra(self,name_output_model,name_interaction_matrix='interactions_navegaciones.npz',model_feature=False,epoch=10,num_threads=16):
        """Método para el entrenamiento utilizado para Perfil de Compra
        Args:
            name_output_model (_str_): Nombre y extensión (.pkl) que se desea dar al archivo con el modelo entrenado.
            epoch (_int_): Numero de veces que se desea iterar el conjunto de datos en el algoritmo.
            num_threads (_int_) = Numero de hilos disponibles para el entrenamiento
        """
        df_coo_hist = sparse.load_npz('hist_'+name_interaction_matrix)
        df_coo_nav = sparse.load_npz('nav_'+name_interaction_matrix)
        model = LightFM(loss='warp')
        if model_feature==False:
            model.fit(df_coo_hist+df_coo_nav, epochs=epoch, num_threads=num_threads)
        else: # Utiliza Demografica
            model.fit(df_coo_hist+df_coo_nav, user_features=self.user_features_matrix, item_features=self.item_features_matrix,
            epochs=epoch, num_threads=num_threads)

        #EXPORTAR DF A PICKLE
        with open(name_output_model, "wb") as f:
            pickle.dump(model, f)
        return None


    def generar_inferencias(self,recomendador,name_prefix_output='',name_model='',partitions_size=1000000,batch_size=100000,n_sublistas=16,cases_list=[]
                           ,name_user_features_matrix=None,name_item_features_matrix=None
                           ,archivo_precios=None,umbral_precio=0):
        """Método que genera recomendaciones 
        Args:
            Recomendador (_function_): Sistema Recomendador que se utilizará (Nombre de la función).
                                       Posibles valores: class_reco.inferencias_perfil_compra_warmstart, class_reco.inferencias_navegaciones, class_reco.inferencias_postcompra,
            name_output_interaction_matrix (_str_) (opcional): Nombre y extensión (.npz) del archivo de interacciones (solo en caso de utilizar variables demográficas)
            name_prefix_output (_str_): Nombre del prefijo que contendran los csv de salida.
            model (_str_): nombre y extensión (.pkl) del archivo que contiene el modelo entrenado.
            partitions_size (_int_): Tamaño de las particiones que se realiza sobre el total de clientes para iterar el avance.
            batch_size (_int_): tamaño subparticiones del partitions_size.
            n_sublistas (_int_): numero de sublistas en las que se divide el batch_size para procesar en paralelo.
            archivo_precios (_str_) (opcional): Nombre y extensión (.csv) del archivo con el listado de precios por SKU
            umbral_precio (_int_) (opcional): limite inferior de precio de los productos disponibles para recomendar.
        """
        self.precios=None
        self.archivo_precios=archivo_precios
        self.umbral_precio=umbral_precio
        if archivo_precios!=None:
            self.precios = pl.from_pandas(pd.read_csv(archivo_precios,names=['CODIGO_PRODUCTO', 'PRECIO'], header=0))

        if recomendador.__func__.__name__=='inferencias_perfil_compra_warmstart':
            df_universo=self.df
        elif recomendador.__func__.__name__=='inferencias_navegaciones' or recomendador.__func__.__name__=='inferencias_postcompra':
            df_universo=self.df_nav

        with open(name_model, "rb") as file:
            model = pickle.load(file)
        ruts_total= len(df_universo['RUT_CLIENTE'].unique()) # RECORRER FULL --> len(df['RUT_CLIENTE'].unique())
        size_parte = partitions_size

        #RECORRE EL DATAFRAME EN PARTES QUE SE ALMACENAN EN DIFERENTES .CSV
        rango=range(size_parte, ruts_total+size_parte, size_parte)
        for c,parte in enumerate(rango):
            print('Processing Part '+str(c+1))

            archivo_recos=name_prefix_output+'_P'+str(self.parte_ejecutar)+'_Part'+str(c+1)+'.csv'
            if self.n_partes==1:
                archivo_recos=name_prefix_output+'.csv'

            structure=['SK','RUT','RANKING','CODIGO_PRODUCTO','CATEGORIA_NIVEL_1','CATEGORIA_NIVEL_2_FCV','MACROCATEGORIA','DESCRIPTOR']
            if recomendador.__func__.__name__=='inferencias_navegaciones':
                structure=['SK','RUT','RANKING','CODIGO_PRODUCTO','CATEGORIA_NIVEL_1','CATEGORIA_NIVEL_2_FCV','MACROCATEGORIA','DESCRIPTOR','CANAL']

            df_users_recos=pd.DataFrame(columns=structure)
            df_users_recos.to_csv(archivo_recos,index=True)

            batch_size=batch_size
            total=len(df_universo['RUT_CLIENTE'].unique()[c*size_parte:parte])
            total_iterations = total // batch_size  # Número total de iteraciones
            #RECORRE CADA PARTE EN SUB-PARTES QUE SE ESCRIBEN EN BUCLE EN EL ARCHIVO .CSV DE LA PARTE
            for i in tqdm(range(batch_size, total + batch_size, batch_size), total=total_iterations):
                listado_ruts =list(df_universo['RUT_CLIENTE'].unique()[parte-size_parte:parte])[i-batch_size:i]
                self.df_muestra = self.df.filter(pl.col('RUT_CLIENTE').is_in(listado_ruts)) #ruts especificos requeridos (utilizado después para mirar historico de productos comprados/navegados por cliente)

                self.usuarios_recos=[]
                #Genera sublistas para la lista completa de clientes a evaluar
                n_sublistas = n_sublistas 
                tam_sublista = len(listado_ruts) // n_sublistas
                listado_partes = [listado_ruts[i:i+tam_sublista] for i in range(0, len(listado_ruts), tam_sublista)]

                #Fragmenta dataframes fragmentados que contienen solo los clientes de cada lista
                self.dataframes = {}
                for iter in range(0, len(listado_partes)):
                    name_df = "df" + str(iter)
                    if recomendador.__func__.__name__=='inferencias_perfil_compra_warmstart':
                        self.dataframes[name_df] = df_universo[['RUT_CLIENTE','CODIGO_PRODUCTO','MACROCATEGORIA']].filter(df_universo["RUT_CLIENTE"].is_in(listado_partes[iter]))
                    elif recomendador.__func__.__name__=='inferencias_navegaciones' or recomendador.__func__.__name__=='inferencias_postcompra':
                        self.dataframes[name_df] = df_universo[['RUT_CLIENTE','CODIGO_PRODUCTO','MACROCATEGORIA','CANAL']].filter(df_universo["RUT_CLIENTE"].is_in(listado_partes[iter]))        
                self.dataframes.keys()

                # Crear un ThreadPoolExecutor con el número de hilos deseados (por ejemplo, 4)
                n_hilos = os.cpu_count()
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_hilos) as executor:
                    # Ejecutar el procesamiento en paralelo para cada muestra
                    self.listado_partes_enumerated = list(enumerate(listado_partes))
                    recomendador_partial = partial(recomendador
                                                  ,model=model 
                                                  ,cases_list=cases_list
                                                  ,name_user_features_matrix=name_user_features_matrix
                                                  ,name_item_features_matrix=name_item_features_matrix
                                                  ) #permite permite "pre-cargar" parametros al llamar funcion con concurrent.futures
                    usuarios_result = list(executor.map(recomendador_partial, self.listado_partes_enumerated))

                recos = [a for a in usuarios_result[0]]

                with open(archivo_recos, 'a', encoding='UTF8', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(recos)
        return None


    def inferencias_perfil_compra_warmstart(self,sublista,model,cases_list=[],nro_recos=8,name_user_features_matrix=None,name_item_features_matrix=None):
        """Método de generación de inferencias para Perfil de Compra
        Args:
            sublista (_list_): listado de clientes a los que se realizará la inferencia.
            coo_matrix (_str_) (opcional): Nombre y extensión (.npz) del archivo de interacciones (solo en caso de utilizar variables demográficas)
            stock (_boolean_): Solicitud de clientes puntuales (False) o stock de entrenamiento (True)
            nro_recos (_int_): Cantidad de recomendaciones a generar por cada cliente.
        """
        params = {"item_ids": np.arange(self.n_items)}
        if name_user_features_matrix!=None:
            user_features_matrix = load_npz(name_user_features_matrix)
            params["user_features"] = user_features_matrix
        if name_item_features_matrix!=None:
            item_features_matrix = load_npz(name_item_features_matrix)
            params["item_features"] = item_features_matrix

        idx,sublist=sublista    
        if cases_list==[]:
            indices_muestra = [self.user_dict[r] for r in sublist]
        else:
            indices_muestra = [self.user_dict[r] for r in sublist if self.user_dict[r] in cases_list]

        for c, u in enumerate(indices_muestra): #para cada cliente de la base...
            filter_df = self.dataframes['df'+str(idx)].filter(self.dataframes['df'+str(idx)]["RUT_CLIENTE"] == self.users[u])
            df_muestra_usuario = filter_df['CODIGO_PRODUCTO'].to_list()
            cat_consumidas = filter_df.groupby('MACROCATEGORIA').agg(pl.count('MACROCATEGORIA').alias('conteo')).sort('conteo').reverse()['MACROCATEGORIA'].to_list()

            params.update({"user_ids": u})
            scores= model.predict(**params)

            top_items=pd.DataFrame(self.items_cods[np.argsort(-scores)],columns=['CODIGO_PRODUCTO'])
            top_items['index']=top_items.index
            top_items = pl.from_pandas(top_items)
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(df_muestra_usuario)) #excluir productos que clientes ha comprado en el pasado
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(self.mcfma))
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(self.cods_excluir)) #SE EXCLUYEN PRODUCTOS DEL LISTADO DE EXCLUSIÓN

            #EXCLUSIONES GENERO
            if self.df_muestra.filter(self.df_muestra['RUT_CLIENTE'] == self.users[u])['GENERO'].unique().to_list()[0]=='MAS':
                top_items = top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(self.excl_mas))
            else: 
                top_items = top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(self.excl_fem))

            top_items=top_items.with_columns([pl.arange(0, top_items.shape[0]).alias('new_index')]).drop('index')
            top_items=top_items.with_columns(top_items['new_index'].alias('index')).drop('new_index')
            
            #REGLAS DE NEGOCIO PARA REPRIORIZAR EL ORDEN DE RECOMENDACIONES (score se utiliza como segundo criterio de priorización)
            #Las primeras 4 recomendaciones serán asociadas a Macrocategorias que el cliente haya consumido, las siguientes 4 se dejaran naturalmente a lo que priorice el score. 
            top_items=top_items.join(self.dicc,on='CODIGO_PRODUCTO',how='inner').sort('index')

            if self.archivo_precios!=None: # Establecer un minimo de precios correspondiente a "sobre la mediana de la categoria" o "Mayor a X monto" (umbral)
                top_items=top_items.join(self.precios,on='CODIGO_PRODUCTO',how='left').sort('index')
                #precio_med_cat = top_items.groupby('CATEGORIA_NIVEL_1').agg(PRECIO_MED=pl.col('PRECIO').median())
                #top_items = top_items.join(precio_med_cat, on='CATEGORIA_NIVEL_1', how='left')
                #top_items = top_items.filter((pl.col('PRECIO') > pl.col('PRECIO_MED')) | (pl.col('PRECIO') > self.umbral_precio)) 
                top_items = top_items.filter(pl.col('PRECIO') >= self.umbral_precio) 

            cat_cons1=top_items.filter(top_items['MACROCATEGORIA'].is_in(cat_consumidas))
            cat_cons1=cat_cons1.groupby('MACROCATEGORIA', maintain_order=True).first()['CATEGORIA_NIVEL_1'][:4].to_list()
            cat_cons2=top_items.filter(~top_items['CATEGORIA_NIVEL_1'].is_in(cat_cons1))['CATEGORIA_NIVEL_1'].unique().to_list()
            orden_cats=cat_cons1 + cat_cons2 #CATEGORIAS_N1 dentro de 4 MACROCATEGORIAS más consumidas + CATEGORIAS N1 ordenadas por SCORE
            top_items=top_items.groupby('CATEGORIA_NIVEL_1', maintain_order=True).first() #SE EXCLUYE REPETICION DE CATEGORIAS_NIVEL_1
            catn1_index = top_items['CATEGORIA_NIVEL_1'].apply(lambda x: orden_cats.index(x))
            top_items = top_items.insert_at_idx(0, pl.Series('CATN1_INDEX', catn1_index)).sort(['CATN1_INDEX','index']) #indexa Priorizando categorias
            top_items=top_items.with_columns([pl.arange(0, top_items.shape[0]).alias('new_index')]).drop('index')
            top_items=top_items.with_columns(top_items['new_index'].alias('index')).drop('new_index')
            recomendaciones=[]
            cat_contenidas = set()
            cat_contenidas2 = []
            descriptores = set()

            #BUCLE DE INCORPORACIÓN DE RECOMENDACIONES
            for n,i in enumerate(top_items['CODIGO_PRODUCTO']):
                if len(recomendaciones)>=nro_recos:
                    recomendaciones=[]
                    break                                 
                item = top_items.filter(top_items['CODIGO_PRODUCTO'] == i)
                if (
                    (cat_contenidas2.count(item['MACROCATEGORIA'].to_list()[0]) < 2) and # no más de 2 veces la MACROCATEGORIA
                    (item['CODIGO_PRODUCTO'].to_list()[0] not in self.prereq_compra or item['MACROCATEGORIA'].to_list()[0] in list(cat_consumidas)) #and # no ser parte de categorias definidas con pre-requisito de consumo, o bien haber consumido antes la macrocategoria.
                    ):
                    
                    #BLOQUE PARA NO RECOMENDAR SI EL DESCRIPTOR DEL PRODUCTO ES SIMILAR A RECOMENDACIONES ANTERIORES
                    tokens = re.findall(r'\b\w+\b|\w+(?:[-.]\w+)+', item['DESCRIPTOR'].to_list()[0])
                    #exclusión stopwords y otros...
                    descriptor_actual = [palabra for palabra in tokens if palabra.lower() not in self.stop_words and not re.match(self.patron, palabra) and len(palabra) > 3] # tokenización / sin stopwords / palabras minimo 3 letras
                    descriptores_anteriores = []
                    for desc in descriptores: #tokenizado de recomendaciones (descriptores) anteriores
                        tokens = re.findall(r'\b\w+\b|\w+(?:[-.]\w+)+', desc)
                        desc_n = [palabra for palabra in tokens if palabra.lower() not in self.stop_words and not re.match(self.patron, palabra) and len(palabra) > 3]
                        descriptores_anteriores.append(desc_n)
                    cantidad_coincidentes=0
                    if len(descriptores_anteriores) > 0:
                        cantidad_coincidentes = max(len(set(descriptor_actual) & set(elemento)) for elemento in descriptores_anteriores) #maximo de palabras coincidentes en alguno de los descriptores anteriores

                    #SI LOS DESCRIPTORES NO SON SIMILARES, INCORPORACIÓN DE RECOMENDACION
                    if cantidad_coincidentes < 3: # Si la cantidad de palabras coincidentes en todas las listas es menor a 3 entonces se agrega como recomendación  
                        # Se almacenan las recomendaciones
                        recomendaciones.append(item['CODIGO_PRODUCTO'].to_list()[0])
                        self.usuarios_recos.append([u] + [(self.users[u]*2)-1599] +[self.users[u]] + [str(len(recomendaciones))] + [item['CODIGO_PRODUCTO'].to_list()[0]] + [item['CATEGORIA_NIVEL_1'].to_list()[0]] + [item['CATEGORIA_NIVEL_2_FCV'].to_list()[0]] + [item['MACROCATEGORIA'].to_list()[0]] + [item['DESCRIPTOR'].to_list()[0]])
                        cat_contenidas.add(item['CATEGORIA_NIVEL_1'].to_list()[0])
                        cat_contenidas2.append(item['MACROCATEGORIA'].to_list()[0])
                        descriptores.add(item['DESCRIPTOR'].to_list()[0])
        return self.usuarios_recos
    

    def inferencias_perfil_compra_coldstart(self,sublista,interaction_matrix,name_user_features_matrix=None,name_item_features_matrix=None,nro_recos=8,model='',dataset=''):
        ####### ALTERNATIVA 1 UTILIZANDO CARACTERISTICAS DE CLIENTE

        #sublist=[sublista[0]] # solo configurado para responder a un unico cliente (el primero)
        user_features_matrix = load_npz(name_user_features_matrix)
        #item_features_matrix = load_npz(name_item_features_matrix)
        interaction_matrix = load_npz(interaction_matrix)
        products=pd.read_csv('products_model.csv')

        with open(model, "rb") as file:
            model = pickle.load(file)

        with open(dataset, "rb") as file:
            dataset = pickle.load(file)    

        user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

        caracteristicas=pd.read_csv('SR_CARACTERISTICAS.csv',sep=';',encoding='latin')
        car_cli=caracteristicas[caracteristicas.RUT_CLIENTE==sublista[0]][['GENERO']] #,'GSE'
        
        new_user_features_matrix = coo_matrix((1, user_features_matrix.shape[1])) #vacia (1 = tamaño filas = n° usuarios)

        for var in car_cli.columns: # por cada caracteristica del cliente
            gender_feature_index = user_feature_map[var+':'+car_cli[var][0]] # Mapea el Indice asociado al valor de la caracteristica ingresada
            data = [1.0]
            row = [0]  # Índice de fila para el nuevo usuario
            col = [gender_feature_index]    # Índice de columna para la característica ej: 'GENERO:FEM'

            # Insertar la información en la matriz
            new_user_features_matrix += coo_matrix((data, (row, col)), shape=new_user_features_matrix.shape)

        combined_user_features_matrix = vstack([user_features_matrix, new_user_features_matrix]) 

        model.fit_partial(interactions=interaction_matrix, user_features=combined_user_features_matrix)

        scores = model.predict(max(user_id_map.values())+1
                                  ,np.arange(len(products))
                                  ,user_features=combined_user_features_matrix) # max(user_id_map.values()) corresponde al ultimo indice del stock entrenado, se le suma +1 por que es el nuevo usuario agregado  
        
        ####### ALTERNATIVA 2 UTILIZANDO CARACTERISTICAS DE ITEMS

        # ...

        # ...
        
        ###### PENDIENTE DE AGREGAR RESTO DE PROCESO DE FILTRADO DE PRODUCTOS POR SCORING Y FILTROS

        # ...

        # ...


        return scores    

    
    def inferencias_navegaciones(self,sublista,model,cases_list=[],nro_recos=5,name_user_features_matrix=None,name_item_features_matrix=None):
        """Método de generación de inferencias para Recomendacion por Navegaciones
        Args:
            sublista (_list_): listado de clientes a los que se realizará la inferencia.
            nro_recos (_int_): Cantidad de recomendaciones a generar por cada cliente.
        """
        params = {"item_ids": np.arange(self.n_items)}
        if name_user_features_matrix!=None:
            user_features_matrix = load_npz(name_user_features_matrix)
            params["user_features"] = user_features_matrix
        if name_item_features_matrix!=None:
            item_features_matrix = load_npz(name_item_features_matrix)
            params["item_features"] = item_features_matrix

        idx,sublist=sublista
        indices_muestra = [self.user_dict[r] for r in sublist]
        usuarios_recos=[]
        for c, u in enumerate(indices_muestra): #para cada cliente de la base...
            filter_df = self.dataframes['df'+str(idx)].filter(self.dataframes['df'+str(idx)]["RUT_CLIENTE"] == self.users[u])
            df_muestra_usuario = filter_df['CODIGO_PRODUCTO'].to_list() #productos navegados
            canal_usuario=filter_df.filter(filter_df['RUT_CLIENTE'] == self.users[u])['CANAL'].unique().to_list()[0]

            prod_nav_cli=self.df_nav.filter(self.df_nav['RUT_CLIENTE']==self.users[u])['CODIGO_PRODUCTO'].to_list()
            bioq_nav_cli=self.switch_bioeq.filter(self.switch_bioeq['CODIGO_PRODUCTO'].is_in(prod_nav_cli))['CODIGO_PRODUCTO_BIOEQ'].to_list()
            self.cat_nav_usr=self.df_nav.filter(self.df_nav['RUT_CLIENTE']==self.users[u])['MACROCATEGORIA'].unique().to_list() #categorías que navegó el cliente (no farma)
            
            
            #Determinar Navegaciones en productos con bioequivalente y almacenar el bioequivalente para insertarlo al final en la lista de recomendaciones
            self.bioq_nav=self.df_nav.filter((self.df_nav['RUT_CLIENTE']==self.users[u]) & (self.df_nav['CODIGO_PRODUCTO'].is_in(list(self.switch_bioeq['CODIGO_PRODUCTO']))))
            self.bioq_rec=self.bioq_nav.join(self.switch_bioeq,on='CODIGO_PRODUCTO',how='inner')[['CODIGO_PRODUCTO','CODIGO_PRODUCTO_BIOEQ']]
            self.bioq_rec=self.bioq_rec.join(self.dicc,left_on='CODIGO_PRODUCTO_BIOEQ',right_on='CODIGO_PRODUCTO',how='left')\
                          [['MACROCATEGORIA','CODIGO_PRODUCTO_BIOEQ','DESCRIPTOR','CATEGORIA_NIVEL_1']].rename({"CODIGO_PRODUCTO_BIOEQ": "CODIGO_PRODUCTO"})
            
            macros_rel=list()
            self.macros_rel_full=[]
            if len(self.cat_rel_nav.filter(self.cat_rel_nav['antecedents'].is_in(self.cat_nav_usr)))!=0: # si esto no se cumple es que no hay categorías relacionadas generadas en el basket analysis
                self.cat_rel_nav_usr=self.cat_rel_nav.filter(self.cat_rel_nav['antecedents'].is_in(self.cat_nav_usr)).sort(by='lift', descending=True) #filtra
                self.macros_rel_full=list(dict.fromkeys(self.cat_rel_nav_usr['consequents'].to_list())) #rankea macrocategorías mayormente relacionadas

            params.update({"user_ids": u})
            scores= model.predict(**params)

            top_items=pd.DataFrame(self.items_cods[np.argsort(-scores)],columns=['CODIGO_PRODUCTO'])
            top_items['index']=top_items.index
            top_items = pl.from_pandas(top_items)
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(df_muestra_usuario)) #Excluir productos navegados
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(self.mcfma))
            self.macros_rel=self.macros_rel_full

            self.macros_reco=list(set([cat for cat in self.cat_nav_usr if self.cat_nav_usr not in self.macros_farma]))  +  self.macros_rel 
            dicc_filt=pl.DataFrame({'CODIGO_PRODUCTO': 
                                    self.dicc.filter( (self.dicc['CODIGO_PRODUCTO'].is_in(top_items['CODIGO_PRODUCTO'])) & 
                                                     (self.dicc['MACROCATEGORIA'].is_in(self.macros_reco)) )['CODIGO_PRODUCTO']
                                   }) 
            top_items=top_items.join(dicc_filt,on='CODIGO_PRODUCTO',how='inner').sort(by='index') #conserva solo aquellos pertenecientes a las macrocategorías de la navegación
            top_items=top_items.join(self.dicc,on='CODIGO_PRODUCTO',how='inner').sort(by='index')
            top_items=top_items.filter( (~top_items['CODIGO_PRODUCTO'].is_in(self.cods_excluir)) # Producto no se encuentre en listado de exclusiones
                                           & ( (top_items['UNIDAD_COMERCIAL']!='FARMA') | (top_items['CODIGO_PRODUCTO'].is_in(bioq_nav_cli)) )# Producto no sea FARMA o bien corresponda a un BIOEQUIVALENTE de los navegados por cliente
                                           )
            
            mapping_dict = {value: idx for idx, value in enumerate(self.macros_reco)}
            macro_index_expr = pl.col("MACROCATEGORIA").apply(lambda value: mapping_dict.get(value, None),skip_nulls=False).alias("MACRO_INDEX")
            
            top_items = top_items.with_columns([macro_index_expr])

            top_items_b=top_items.sort(['CATEGORIA_NIVEL_1', 'index']).drop("index") # Genera un segundo ordenamiento por CATEGORIA_NIVEL_1 en caso que hubiera que incorporar MACROCATEGORIAS repetidas
            new_index_values = pl.arange(0, top_items_b.shape[0])
            self.top_items_b = top_items_b.with_columns(new_index_values.alias("index"))
            top_items_c=top_items.sort(by=['index'])['CODIGO_PRODUCTO'] #Categorias simplemente ordenadas por SCORE

            top_items=top_items.sort(['MACRO_INDEX', 'index']).drop(['MACRO_INDEX','index']) # Ordena por Macrocategoría y en segunda instancia Score
            new_index_values = pl.arange(0, top_items.shape[0])
            top_items = top_items.with_columns(new_index_values.alias("index"))

            top_items=top_items.groupby('MACROCATEGORIA', maintain_order=True).first() #Solo toma el primer elemento por categoría (es decir el que tiene mayor score)

            a=self.bioq_rec.clone()
            a=a.to_pandas()
            b=top_items[['MACROCATEGORIA','CODIGO_PRODUCTO','DESCRIPTOR','CATEGORIA_NIVEL_1']].clone()
            b=b.to_pandas()
            #top_items=self.bioq_rec.vstack(top_items[['MACROCATEGORIA','CODIGO_PRODUCTO','DESCRIPTOR','CATEGORIA_NIVEL_1']]) #Insertar bioequivalente como primera recomendación
            top_items=pd.concat([a,b] , axis=0).reset_index(drop=True)
            

            top_items_b=top_items_b.groupby('CATEGORIA_NIVEL_1', maintain_order=True).first() #Solo toma el primer elemento por categoría (es decir el que tiene mayor score)
            
            cat_uniques=len(top_items['MACROCATEGORIA'].unique())
            cat_uniques2=len(top_items_b['CATEGORIA_NIVEL_1'].unique())

            top_items=top_items['CODIGO_PRODUCTO']
            top_items_b=top_items_b['CODIGO_PRODUCTO']

            recomendaciones=[]
            cat_contenidas = set()
            cat_contenidas2 = set()
            descriptores = set()
            
            for n,i in enumerate(top_items):
                #print('user: ' + str(u) + 'recos: ' + str(recomendaciones))
                if len(recomendaciones) >= min(cat_uniques,nro_recos):
                    break
                item = self.diccionario[self.diccionario.CODIGO_PRODUCTO == i]
                if (item.MACROCATEGORIA.values[0] not in cat_contenidas):
                    # Se almacenan las recomendaciones
                    recomendaciones.append(item.CODIGO_PRODUCTO.values[0])
                    usuarios_recos.append([c] + [(self.users[u]*2)-1599] +[self.users[u]] + [str(len(recomendaciones))] + [item.CODIGO_PRODUCTO.values[0]] + [item.CATEGORIA_NIVEL_1.values[0]] + [item.CATEGORIA_NIVEL_2_FCV.values[0]] + 
                                            [item.MACROCATEGORIA.values[0]] + [item.DESCRIPTOR.values[0]] + [canal_usuario])
                    cat_contenidas.add(item.MACROCATEGORIA.values[0])
                    cat_contenidas2.add(item.CATEGORIA_NIVEL_1.values[0])
                    descriptores.add(item.DESCRIPTOR.values[0])
            
            if min(cat_uniques,nro_recos)<nro_recos: # Complementa las recomendaciones faltantes en caso de no haber suficientes MACROCRATEGORIAS utilizando criterio de distintas CATEGORIA_NIVEL_1, de no ser suficiente se complementarían en el siguiente bloque de codigo 
                for n,i in enumerate(top_items_b):
                    #print('user: ' + str(u) + 'recos: ' + str(recomendaciones))
                    if len(recomendaciones) >= min(cat_uniques+cat_uniques2,nro_recos):
                        break
                    item = self.diccionario[self.diccionario.CODIGO_PRODUCTO == i]
                    if (
                        #(i not in (recomendaciones) ) and
                        (item.CATEGORIA_NIVEL_1.values[0] not in cat_contenidas2)
                        ):
                    # Se almacenan las recomendaciones
                        recomendaciones.append(item.CODIGO_PRODUCTO.values[0])
                        usuarios_recos.append([c] + [(self.users[u]*2)-1599] + [self.users[u]] + [str(len(recomendaciones))] + [item.CODIGO_PRODUCTO.values[0]] + [item.CATEGORIA_NIVEL_1.values[0]] + [item.CATEGORIA_NIVEL_2_FCV.values[0]] + 
                                                [item.MACROCATEGORIA.values[0]] + [item.DESCRIPTOR.values[0]] + [canal_usuario])
                        cat_contenidas2.add(item.CATEGORIA_NIVEL_1.values[0])
                        descriptores.add(item.DESCRIPTOR.values[0])

            if min(cat_uniques+cat_uniques2,nro_recos)<nro_recos: #Rellena recomendaciones con categorias repetidas en caso de no haber tenido suficientes categorias diferentes en los 2 bloques de código anteriores
                for n,i in enumerate(top_items_c):
                #print('user: ' + str(u) + 'recos: ' + str(recomendaciones))
                    if len(recomendaciones) >= nro_recos:
                        break
                    item = self.diccionario[self.diccionario.CODIGO_PRODUCTO == i]
                    if (
                        (i not in (recomendaciones) )
                        ):
                        #tokenizado
                        tokens = re.findall(r'\b\w+\b|\w+(?:[-.]\w+)+', item.DESCRIPTOR.values[0])
                        #exclusión stopwords y otros...
                        descriptor_actual = [palabra for palabra in tokens if palabra.lower() not in self.stop_words and not re.match(self.patron, palabra) and len(palabra) > 3] # tokenización / sin stopwords / palabras minimo 3 letras

                        descriptores_anteriores = []
                        for desc in descriptores: #tokenizado de recomendaciones (descriptores) anteriores
                            tokens = re.findall(r'\b\w+\b|\w+(?:[-.]\w+)+', desc)
                            desc_n = [palabra for palabra in tokens if palabra.lower() not in self.stop_words and not re.match(self.patron, palabra) and len(palabra) > 3]
                            descriptores_anteriores.append(desc_n)

                        cantidad_coincidentes = max(len(set(descriptor_actual) & set(elemento)) for elemento in descriptores_anteriores) #maximo de palabras coincidentes en alguno de los descriptores anteriores

                        # Si la cantidadde palabras coincidentes en todas las listas es menor a 3 entonces se agrega como recomendación
                        if cantidad_coincidentes < 3:   
                            # Se almacenan las recomendaciones
                            recomendaciones.append(item.CODIGO_PRODUCTO.values[0])
                            usuarios_recos.append([c] + [(self.users[u]*2)-1599] + [self.users[u]] + [str(len(recomendaciones))] + [item.CODIGO_PRODUCTO.values[0]] + [item.CATEGORIA_NIVEL_1.values[0]] + [item.CATEGORIA_NIVEL_2_FCV.values[0]] + 
                                                [item.MACROCATEGORIA.values[0]] +[item.DESCRIPTOR.values[0]] + [canal_usuario])
                           
        return usuarios_recos
    

    def inferencias_postcompra(self,sublista,model,cases_list=[],nro_recos=5,name_user_features_matrix=None,name_item_features_matrix=None):
        """Método de generación de inferencias para Recomendacion por Post-Compra
        Args:
            sublista (_list_): listado de clientes a los que se realizará la inferencia.
            nro_recos (_int_): Cantidad de recomendaciones a generar por cada cliente.
        """
        params = {"item_ids": np.arange(self.n_items)}
        if name_user_features_matrix!=None:
            user_features_matrix = load_npz(name_user_features_matrix)
            params["user_features"] = user_features_matrix
        if name_item_features_matrix!=None:
            item_features_matrix = load_npz(name_item_features_matrix)
            params["item_features"] = item_features_matrix

        idx,sublist=sublista
        indices_muestra = [self.user_dict[r] for r in sublist]
        usuarios_recos=[]
        for c, u in enumerate(indices_muestra): #para cada cliente de la base...
            #print('Caso N°: '+str(c)+' Usuario: '+str(u)+' RUT: '+str(users[u]))
            filter_df = self.dataframes['df'+str(idx)].filter(self.dataframes['df'+str(idx)]["RUT_CLIENTE"] == self.users[u])
            df_muestra_usuario = filter_df['CODIGO_PRODUCTO'].to_list() #productos comprados
            #canal_usuario=filter_df.filter(filter_df['RUT_CLIENTE'] == self.users[u])['CANAL'].unique().to_list()[0]

            self.cat_nav_usr=self.df_nav.filter(self.df_nav['RUT_CLIENTE']==self.users[u])['MACROCATEGORIA'].unique().to_list() #categorías que compró el cliente (no farma)
            
            macros_rel=list()
            self.macros_rel_full=[]
            if len(self.cat_rel_nav.filter(self.cat_rel_nav['antecedents'].is_in(self.cat_nav_usr)))!=0: # si esto no se cumple es que no hay categorías relacionadas generadas en el basket analysis
                self.cat_rel_nav_usr=self.cat_rel_nav.filter(self.cat_rel_nav['antecedents'].is_in(self.cat_nav_usr)).sort(by='lift', descending=True) #filtra
                self.macros_rel_full=list(dict.fromkeys(self.cat_rel_nav_usr['consequents'].to_list())) #rankea macrocategorías mayormente relacionadas

            params.update({"user_ids": u})
            scores= model.predict(**params)

            top_items=pd.DataFrame(self.items_cods[np.argsort(-scores)],columns=['CODIGO_PRODUCTO'])
            top_items['index']=top_items.index
            top_items = pl.from_pandas(top_items)
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(df_muestra_usuario)) #Excluir productos comprados
            top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(self.mcfma))
            self.macros_rel=self.macros_rel_full

            self.macros_reco=list(set([cat for cat in self.cat_nav_usr if self.cat_nav_usr not in self.macros_farma]))  +  self.macros_rel 
            dicc_filt=pl.DataFrame({'CODIGO_PRODUCTO': 
                                    self.dicc.filter( (self.dicc['CODIGO_PRODUCTO'].is_in(top_items['CODIGO_PRODUCTO'])) & 
                                                     (self.dicc['MACROCATEGORIA'].is_in(self.macros_reco)) )['CODIGO_PRODUCTO']
                                   }) 
            top_items=top_items.join(dicc_filt,on='CODIGO_PRODUCTO',how='inner').sort(by='index') #conserva solo aquellos pertenecientes a las macrocategorías de la compra
            top_items=top_items.join(self.dicc,on='CODIGO_PRODUCTO',how='inner').sort(by='index')
            top_items=top_items.filter( (~top_items['CODIGO_PRODUCTO'].is_in(self.cods_excluir)) # Producto no se encuentre en listado de exclusiones
                                           & ( (top_items['UNIDAD_COMERCIAL']!='FARMA') 
                                              #| (top_items['CODIGO_PRODUCTO'].is_in(bioq_nav_cli)) 
                                             )# Producto no sea FARMA
                                           )
            
            mapping_dict = {value: idx for idx, value in enumerate(self.macros_reco)}
            macro_index_expr = pl.col("MACROCATEGORIA").apply(lambda value: mapping_dict.get(value, None),skip_nulls=False).alias("MACRO_INDEX")
            
            top_items = top_items.with_columns([macro_index_expr])

            top_items_b=top_items.sort(['CATEGORIA_NIVEL_1', 'index']).drop("index") # Genera un segundo ordenamiento por CATEGORIA_NIVEL_1 en caso que hubiera que incorporar MACROCATEGORIAS repetidas
            new_index_values = pl.arange(0, top_items_b.shape[0])
            self.top_items_b = top_items_b.with_columns(new_index_values.alias("index"))
            top_items_c=top_items.sort(by=['index'])['CODIGO_PRODUCTO'] #Categorias simplemente ordenadas por SCORE

            top_items=top_items.sort(['MACRO_INDEX', 'index']).drop(['MACRO_INDEX','index']) # Ordena por Macrocategoría y en segunda instancia Score
            new_index_values = pl.arange(0, top_items.shape[0])
            top_items = top_items.with_columns(new_index_values.alias("index"))

            top_items=top_items.groupby('MACROCATEGORIA', maintain_order=True).first() #Solo toma el primer elemento por categoría (es decir el que tiene mayor score)

            top_items=top_items.to_pandas()
            top_items=top_items[['MACROCATEGORIA','CODIGO_PRODUCTO','DESCRIPTOR','CATEGORIA_NIVEL_1']].reset_index(drop=True)
            
            top_items_b=top_items_b.groupby('CATEGORIA_NIVEL_1', maintain_order=True).first() #Solo toma el primer elemento por categoría (es decir el que tiene mayor score)
            
            cat_uniques=len(top_items['MACROCATEGORIA'].unique())
            cat_uniques2=len(top_items_b['CATEGORIA_NIVEL_1'].unique())

            top_items=top_items['CODIGO_PRODUCTO']
            top_items_b=top_items_b['CODIGO_PRODUCTO']

            recomendaciones=[]
            cat_contenidas = set()
            cat_contenidas2 = set()
            descriptores = set()
            
            for n,i in enumerate(top_items):
                #print('user: ' + str(u) + 'recos: ' + str(recomendaciones))
                if len(recomendaciones) >= min(cat_uniques,nro_recos):
                    break
                item = self.diccionario[self.diccionario.CODIGO_PRODUCTO == i]
                if (item.MACROCATEGORIA.values[0] not in cat_contenidas):
                    # Se almacenan las recomendaciones
                    recomendaciones.append(item.CODIGO_PRODUCTO.values[0])
                    usuarios_recos.append([c] + [(self.users[u]*2)-1599] +[self.users[u]] + [str(len(recomendaciones))] + [item.CODIGO_PRODUCTO.values[0]] + [item.CATEGORIA_NIVEL_1.values[0]] + [item.CATEGORIA_NIVEL_2_FCV.values[0]] + 
                                            [item.MACROCATEGORIA.values[0]] + [item.DESCRIPTOR.values[0]])# + [canal_usuario])
                    cat_contenidas.add(item.MACROCATEGORIA.values[0])
                    cat_contenidas2.add(item.CATEGORIA_NIVEL_1.values[0])
                    descriptores.add(item.DESCRIPTOR.values[0])
            
            if min(cat_uniques,nro_recos)<nro_recos: # Complementa las recomendaciones faltantes en caso de no haber suficientes MACROCRATEGORIAS utilizando criterio de distintas CATEGORIA_NIVEL_1, de no ser suficiente se complementarían en el siguiente bloque de codigo 
                for n,i in enumerate(top_items_b):
                    #print('user: ' + str(u) + 'recos: ' + str(recomendaciones))
                    if len(recomendaciones) >= min(cat_uniques+cat_uniques2,nro_recos):
                        break
                    item = self.diccionario[self.diccionario.CODIGO_PRODUCTO == i]
                    if (
                        #(i not in (recomendaciones) ) and
                        (item.CATEGORIA_NIVEL_1.values[0] not in cat_contenidas2)
                        ):
                    # Se almacenan las recomendaciones
                        recomendaciones.append(item.CODIGO_PRODUCTO.values[0])
                        usuarios_recos.append([c] + [(self.users[u]*2)-1599] + [self.users[u]] + [str(len(recomendaciones))] + [item.CODIGO_PRODUCTO.values[0]] + [item.CATEGORIA_NIVEL_1.values[0]] + [item.CATEGORIA_NIVEL_2_FCV.values[0]] + 
                                                [item.MACROCATEGORIA.values[0]] + [item.DESCRIPTOR.values[0]])# + [canal_usuario])
                        cat_contenidas2.add(item.CATEGORIA_NIVEL_1.values[0])
                        descriptores.add(item.DESCRIPTOR.values[0])

            if min(cat_uniques+cat_uniques2,nro_recos)<nro_recos: #Rellena recomendaciones con categorias repetidas en caso de no haber tenido suficientes categorias diferentes en los 2 bloques de código anteriores
                for n,i in enumerate(top_items_c):
                #print('user: ' + str(u) + 'recos: ' + str(recomendaciones))
                    if len(recomendaciones) >= nro_recos:
                        break
                    item = self.diccionario[self.diccionario.CODIGO_PRODUCTO == i]
                    if (
                        (i not in (recomendaciones) )
                        ):
                        #tokenizado
                        tokens = re.findall(r'\b\w+\b|\w+(?:[-.]\w+)+', item.DESCRIPTOR.values[0])
                        #exclusión stopwords y otros...
                        descriptor_actual = [palabra for palabra in tokens if palabra.lower() not in self.stop_words and not re.match(self.patron, palabra) and len(palabra) > 3] # tokenización / sin stopwords / palabras minimo 3 letras

                        descriptores_anteriores = []
                        for desc in descriptores: #tokenizado de recomendaciones (descriptores) anteriores
                            tokens = re.findall(r'\b\w+\b|\w+(?:[-.]\w+)+', desc)
                            desc_n = [palabra for palabra in tokens if palabra.lower() not in self.stop_words and not re.match(self.patron, palabra) and len(palabra) > 3]
                            descriptores_anteriores.append(desc_n)

                        cantidad_coincidentes = max(len(set(descriptor_actual) & set(elemento)) for elemento in descriptores_anteriores) #maximo de palabras coincidentes en alguno de los descriptores anteriores

                        # Si la cantidadde palabras coincidentes en todas las listas es menor a 3 entonces se agrega como recomendación
                        if cantidad_coincidentes < 3:   
                            # Se almacenan las recomendaciones
                            recomendaciones.append(item.CODIGO_PRODUCTO.values[0])
                            usuarios_recos.append([c] + [(self.users[u]*2)-1599] + [self.users[u]] + [str(len(recomendaciones))] + [item.CODIGO_PRODUCTO.values[0]] + [item.CATEGORIA_NIVEL_1.values[0]] + [item.CATEGORIA_NIVEL_2_FCV.values[0]] + 
                                                [item.MACROCATEGORIA.values[0]] +[item.DESCRIPTOR.values[0]])# + [canal_usuario])                        

        return usuarios_recos


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
        
        model.fit_partial(interactions=interaction_matrix
                         ,user_features=combined_user_features_matrix
                         ,item_features=combined_item_features_matrix
                         ,num_threads=num_threads)

        params = {"user_ids": user_id #corresponde al indice de usuario consultado
                 ,"item_ids": item_ids #Corresponde al array completo de indices de productos
                 ,"user_features": combined_user_features_matrix
                 ,"item_features": combined_item_features_matrix}
        
        scores = model.predict(**params)

        ###### PROCESO DE PRIORIZACIÓN DE PRODUCTOS
        top_items=pd.DataFrame({'CODIGO_PRODUCTO': products}).iloc[np.argsort(-scores)]
        top_items['index']=top_items.index
        #top_items=top_items.filter(~top_items['CODIGO_PRODUCTO'].is_in(df_muestra_usuario)) #Excluir productos comprados
        top_items=top_items[~top_items['CODIGO_PRODUCTO'].isin(skus_farma)]
        self.top_items=top_items[~top_items['CODIGO_PRODUCTO'].isin(exclusiones)]
        top_items=top_items.CODIGO_PRODUCTO[:nro_recos].tolist()


        return top_items    
    
        #self.interaction_matrix=interaction_matrix
        #self.user_id=user_id
        #self.user_ids=user_ids
        #self.item_id=item_id
        #self.item_ids=item_ids
        #self.products=products