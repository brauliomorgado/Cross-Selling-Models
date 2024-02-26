# ==================================================================================================
# Librerías
# ==================================================================================================

import warnings
import numpy as np
import pandas as pd
import re
import pyodbc
import cx_Oracle
import gc
import psutil
from sympy import Q
from tqdm.auto import tqdm
from IPython.display import display
from scipy.stats import chi2
from decimal import DivisionByZero
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler

# statsmodels
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# sklearn
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix,accuracy_score, classification_report,roc_auc_score #, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import RocCurveDisplay as plot_roc_curve
from sklearn.metrics import PrecisionRecallDisplay as plot_precision_recall_curve

from xverse.transformer import WOE
from joblib import dump, load

# Opciones
warnings.filterwarnings('ignore')

# ==================================================================================================
# Funciones Auxiliares
# ==================================================================================================

# Stepwise LRT

def stepwise_lrt(X, y, pvalue_in, pvalue_out, variables):
    """Realiza seleccion stepwise por test de razon de verosimilitud.

    Args:
        X (_type_): Datos sin el target y sin intercepto
        y (_type_): Target.
        pvalue_in (_type_): Significancia para comparar con valor-p de entrada.
        pvalue_out (_type_): Significancia para comparar con valor-p de salida.
        variables (_type_): Conjunto de variables candidatas al modelo.

    Returns:
        _type_: _description_
    """    
    var_h0 = ['INTERCEPTO']
    var_h1 = variables.copy()
    X_model = X.copy()
    X_model['INTERCEPTO'] = 1

    while True:
        df_lrt_in = pd.DataFrame({'variable': [], 'valor_p': []})

        modelo_h0 = sm.Logit(y, X_model[var_h0])
        modelo_h0_fit = modelo_h0.fit(disp=0)
        ll_h0 = modelo_h0.loglike(modelo_h0_fit.params)

        # Entra Variable
        for new_var in list(set(var_h1) - set(['INTERCEPTO'])):
            
            var_test_in = var_h0 + [new_var]
            try:
                modelo_h1 = sm.Logit(y,  X_model[var_test_in])
                modelo_h1_fit = modelo_h1.fit(disp=0)
                ll_h1 = modelo_h1.loglike(modelo_h1_fit.params)

                estadistico = -2 * (ll_h0 - ll_h1)
                valor_p = 1-chi2.cdf(estadistico, 1)
                
                df_lrt_in = pd.concat([
                    df_lrt_in,
                    pd.DataFrame({'variable': [new_var],'valor_p': [valor_p]})
                ], axis=0)

            except (np.linalg.LinAlgError, ConvergenceWarning, RuntimeWarning) as mi_error:
                pass

        df_lrt_in = df_lrt_in.sort_values(
            by='valor_p', 
            ascending=True
        ).reset_index(drop=True)

        if df_lrt_in.shape[0] == 0 or df_lrt_in.valor_p[0] > pvalue_in:
            break # Para el while si no hay mas variables que entren
        else:
            var_h0 = var_h0 + [df_lrt_in.variable[0]]
            var_h1 = list(set(var_h1) - set([df_lrt_in.variable[0]]))

        # Sale Variable    
        while len(var_h0) > 2:
            
            df_lrt_out = pd.DataFrame({'variable': [], 'valor_p': []})
            
            for old_var in list(set(var_h0) - set(['INTERCEPTO'])): 

                var_h0_aux = list(set(var_h0) - set([old_var]))

                modelo_h0 = sm.Logit(y, X_model[var_h0_aux])
                modelo_h0_fit = modelo_h0.fit(disp=0)
                ll_h0 = modelo_h0.loglike(modelo_h0_fit.params)

                modelo_h1 = sm.Logit(y,  X_model[var_h0])
                modelo_h1_fit = modelo_h1.fit(disp=0)
                ll_h1 = modelo_h1.loglike(modelo_h1_fit.params)

                estadistico = -2 * (ll_h0 - ll_h1)
                valor_p = 1 - chi2.cdf(estadistico, 1)

                df_lrt_out = pd.concat([
                    df_lrt_out,
                    pd.DataFrame({
                            'variable': [old_var],
                            'valor_p': [valor_p]
                    })
                ], axis=0)

            df_lrt_out = df_lrt_out.sort_values(
                by='valor_p', 
                ascending=False
            ).reset_index(drop=True)
            
            if df_lrt_out.valor_p[0] > pvalue_out:
                var_h0 = list(set(var_h0) - set([df_lrt_out.variable[0]]))
                var_h1 = list(set(var_h1) - set([df_lrt_out.variable[0]]))

            else: 
                break # Para el while si no hay mas variables que salgan
            
    return var_h0

# ==================================================================================================
    
# Obtener datos desde SQL Server BCHBD53

def extraer_data(querystring,server):
    if server == 'SQL SERVER':
        conn_bchbd53 = pyodbc.connect(
        'Driver={SQL Server};'
        'Server=BCHBD53;'
        'Trusted_Connection=yes;'
        )
        dataset = pd.read_sql(querystring, conn_bchbd53)
        conn_bchbd53.close()

    if server == 'ORACLE':
        connection = cx_Oracle.connect(user="braulio_morgado", password='SCFBM2022',dsn="CRMFCV_xxxx")

        dataset = pd.read_sql(querystring, con=connection)
        connection.close()

    return dataset

# ==================================================================================================
    
# Imprimir una linea
def imprimir_linea(tipo='_', caracteres=120):
    print(tipo * caracteres)

# ==================================================================================================
    
# Calcula el gini con datos agrupados

def get_gini(buenos, malos):
    porc_buenos = buenos/sum(buenos)
    porc_malos = malos/sum(malos)
    buenos_cum = list(np.cumsum(porc_buenos))
    malos_cum = list(np.cumsum(porc_malos))
    gini_aux = []
    for k in range(len(buenos)):
        if k == 0:
            gini_aux.append(buenos_cum[0] * malos_cum[0])
        if k != 0:
            gini_aux.append(
                    (buenos_cum[k] + buenos_cum[k-1]) *
                    (malos_cum[k] - malos_cum[k-1])
                )
    return 1 - sum(gini_aux)

# ==================================================================================================

def get_roc(buenos, malos):
    roc = (get_gini(buenos, malos) + 1)/2
    return roc

# ==================================================================================================
    
# Calcula el ic con datos agrupados

def get_ic(aux_ic):
    aux_ic = np.array(aux_ic)
    total = sum(aux_ic)
    porc = aux_ic/total
    porc2 = porc**2
    ic = sum(porc2)
    return ic

# ==================================================================================================
    
# Suma la multiplicacion entre la resta de dos valores y  el log de la division de lo valores
# Es util para distintos calculos 

def resta_log(val1, val2):
    aux1 = np.array(val1)/sum(val1)
    aux2 = np.array(val2)/sum(val2)
    return sum((aux1 - aux2) * np.log(aux1/aux2))

# ==================================================================================================

def get_ks(buenos, malos):
    ks = np.max(np.abs(
        buenos.cumsum()/buenos.sum() -
        malos.cumsum()/malos.sum()
    ))
    return ks

# ==================================================================================================

# Genera matriz de confusión basada en las predicciones

def confusion_matrix(df_,th):
    df_["pred"]=np.where(df_.iloc[:,0]>th, 1, 0)
    confusion_matrix = pd.crosstab(df_["pred"],df_.iloc[:,1])
    sens,espc,prec,npv,csi,acc,f1score,BalAcc = confusion_matrix_metrics(confusion_matrix)
    print('sens+espc: ',sens+espc)
    print('KS: ',sens-(1-espc))
    print('sensibility: ',sens)
    print('specificity: ',espc)
    print('precision: ',prec)
    print('negative predictive value: ',npv)
    print('Critical Success Index: ',csi)
    print('accuracy: ',acc)
    print('f1score: ',f1score)
    print('BalancedAccuracy: ',BalAcc)    
    return confusion_matrix

# ================================================================================================== 
 
# Obtiene metricas desde una matriz de confusión

def confusion_matrix_metrics(confusion_matrix):
    try: TN=confusion_matrix.iloc[0,0]
    except: TN=0
    try: FN=confusion_matrix.iloc[0,1] 
    except: FN=0
    try: FP=confusion_matrix.iloc[1,0] 
    except: FP=0
    try: TP=confusion_matrix.iloc[1,1] 
    except: TP=0
    try: sensibility=TP/(TP+FN)
    except: sensibility='indef'  
    try: specificity=TN/(TN+FP)
    except: specificity='indef'  
    try: precision=TP/(TP+FP)
    except: precision='indef' 
    try: npv=TN/(TN+FN)
    except: npv='indef'
    try: csi=TP/(TP+FP+FN)
    except: csi='indef'    
    try: accuracy=(TP+TN)/(TP+TN+FP+FN)
    except: accuracy='indef' 
    try: f1score=(2*(precision*sensibility))/(precision+sensibility)
    except: f1score='indef'     
    try: BalAcc=((TP/(TP+FN))+(TN/(TN+FP)))/2
    except: BalAcc='indef' 
    return sensibility,specificity,precision,npv,csi,accuracy,f1score,BalAcc

# ==================================================================================================
    
# Entrega las metricas optimas de modelo basado en un criterio de optimización

def modelo_optimos(df_,how='roc'):   
    ##CALCULA METRICAS DE ACUERDO A LA UTILIZACIÓN DEL PUNTO OPTIMO EN LA CURVA ROC,CSI y BALANCED ACCURACY
    threshold=np.linspace(0,1,1001)
    opt=0
    
    for i in threshold:
        th = i
        df_["pred"]=np.where(df_.iloc[:,0]>th, 1, 0)
        confusion_matrix = pd.crosstab(df_["pred"],df_.iloc[:,1])
        sens,espc,prec,npv,csi,acc,f1score,BalAcc = confusion_matrix_metrics(confusion_matrix)
        if how=='roc':
            optimizer=sens+espc
        elif how=='f1':
            optimizer=f1score
        elif how=='precision':
            optimizer=prec
        else: 
            optimizer=sens+espc

        if optimizer=='indef': 
            optimizer=0    

        if optimizer>=opt:
            opt=optimizer
            sens_opt,espc_opt,prec_opt,npv_opt,csi_opt,acc_opt,f1score_opt,BalAcc_opt=sens,espc,prec,npv,csi,acc,f1score,BalAcc
            th_opt=i

    df_["pred"]=np.where(df_.iloc[:,0]>th_opt, 1, 0)
    confusion_matrix = pd.crosstab(df_["pred"],df_.iloc[:,1])
    print('"'+str(how).upper()+' OPTIMO"\n')
    print('sens+espc: ',opt)
    print('KS: ',sens_opt-(1-espc_opt))
    print('threshold optimo: ',th_opt)
    print('sensibility: ',sens_opt)
    print('specificity: ',espc_opt)
    print('precision: ',prec_opt)
    print('negative predictive value: ',npv_opt)
    print('Critical Success Index: ',csi_opt)
    print('accuracy: ',acc_opt)
    print('f1score: ',f1score_opt)
    print('BalancedAccuracy: ',BalAcc_opt)
    return confusion_matrix,th_opt


# ==================================================================================================
# Definicion de la clase del pipeline
# ==================================================================================================

class clase_pipeline:

    def inicializar(self):
        self.categorizadas_nulos = []
        return None

    def recibe_data(
        self, dataset_o_query, target='', muestra='', rut='RUT', 
        ano_mes='', categoricas=[], variables_consideradas=[],excluir=[], tipo='query',server='ORACLE',delimiter=';'
    ): 
        """Método que incorpora el dataset al metodo del pipeline.
        Args:
            dataset_o_query (_pandas.core.frame.DataFrame_): Query al dataset de la forma: open("archivo_query.txt","r").read()
                                                           : Nombre Dataset de la forma: 'dataset.csv'
            target (_str_): Nombre de la variable incumplimiento.
        """
        print('Procesando...')

        if tipo == 'query':
            self.dataset = extraer_data(dataset_o_query,server=server)
        elif tipo == 'dataframe':
            print('cargar previamente pipeline_bm.dataset = nombre_dataframe')
        else:
            self.dataset = pd.read_csv(dataset_o_query,delimiter=delimiter)

        self.target = target.upper()
        self.muestra = muestra.upper()
        self.ano_mes = ano_mes.upper()
        self.rut = rut.upper()
        if muestra == '' and ano_mes == '': 
            self.no_considerar = [self.target,self.rut]
        if muestra == '' and ano_mes != '': 
            self.no_considerar = [self.target,self.ano_mes,self.rut] 
        if muestra != '' and ano_mes == '': 
            self.no_considerar = [self.target,self.muestra,self.rut] 
        if muestra != '' and ano_mes != '': 
            self.no_considerar = [self.target,self.ano_mes,self.muestra,self.rut]
        if target=='':
            self.no_considerar.remove('')
        self.no_considerar = self.no_considerar+excluir    
        self.dataset.columns = self.dataset.columns.str.upper()
        self.listado_categoricas = [str(k).upper() for k in categoricas]
        self.listado_continuas = [str(k).upper() for k in (self.dataset.columns) if k not in (self.listado_categoricas+self.no_considerar)]
        
        for k in self.listado_categoricas:
            self.dataset[k] = self.dataset[k].astype(str).replace('nan', np.nan)
            self.dataset[k] = self.dataset[k].replace('None', np.nan)
  
        self.dataset_raw = self.dataset.copy()
        if len(variables_consideradas) != 0:
            self.dataset = self.dataset[variables_consideradas + self.no_considerar]

        if target!='':
            self.dataset[target] = self.dataset[target].fillna(0)

        imprimir_linea()
        print('Datos recibidos: OK.')
        print('Número de filas y columnas del dataset: {}'.format(
            self.dataset.shape
        ))
        imprimir_linea()
        return None

# ==================================================================================================

    def nulos_artificiales(self, nulo_artificial=[-999999, 999999, -999991, -999990]):
        self.dataset = self.dataset.applymap(
            lambda x: np.nan if x in nulo_artificial else x
        )

        print('Los valores {} reemplazados por {}: OK.'.format(
                nulo_artificial, np.nan
        ))
        imprimir_linea()
        return None

# ==================================================================================================

    def categoriza_continuas_nulos(self, corte_nulos=0.05, cuantiles_inicio=10):
        """Categoriza una variable en caso de tener un porcentaje de 
        nulos mayor a 'corte_nulos', intenta hacer 
        "cuantiles_inicio" cuantiles y en caso de error intenta con
        "cuantiles_inicio" - 1 y asi sucesivamente.

        Args:
            corte_nulos (_float_): corte porcentaje de nulos para tramificar.
            cuantiles_inicio (_int_): número de categorias de inicio.
        """

        print('Tramificando...')
        #df_aux_nulos = self.dataset[self.dataset[self.muestra] == 'DEV'].copy()
        df_aux_nulos = self.dataset.copy()
        df_aux = df_aux_nulos.isnull().sum()/df_aux_nulos.shape[0]
        col_nulos = df_aux.loc[df_aux > corte_nulos].index
        col_nulos_cont = set(col_nulos) - set(
            self.no_considerar + self.listado_categoricas
        )
        col_nulos_cat = set(col_nulos) - set(
            list(col_nulos_cont) + self.no_considerar
        )

        colname_except = []
        #self.categorizadas_nulos = []
        
        for colname_cat in self.listado_categoricas:
            if colname_cat in col_nulos:
                self.dataset[colname_cat] = self.dataset[
                    colname_cat
                ].fillna('cat_nulo')
        for colname in tqdm(col_nulos_cont):
            self.dataset[colname] = self.dataset[colname].astype(float)
            df_aux_nulos[colname] = df_aux_nulos[colname].astype(float)
            for k in range(cuantiles_inicio):
                try:
                    df_aux_nulos[colname], my_bins = pd.qcut(df_aux_nulos[colname], cuantiles_inicio-k, retbins=True)
                    self.dataset[colname] = pd.cut(self.dataset[colname], bins=my_bins)
                    break

                except:
                    if k == (cuantiles_inicio-1):
                        colname_except.append(colname)
                            
            if colname not in colname_except:

                try:
                    self.dataset[colname]=self.dataset[colname].astype('string')
                    self.dataset[colname]=self.dataset[colname].fillna('cat_nulo')
                    self.dataset[colname]=self.dataset[colname].astype('category')
                except:
                    self.dataset[colname]=self.dataset[colname].cat.add_categories('Nulos')
                    self.dataset[colname]=self.dataset[colname].fillna('Nulos')
                self.categorizadas_nulos.append(colname)
           
        for colname in tqdm(col_nulos_cat):
            self.dataset[colname] = self.dataset[colname].fillna('cat_nulo')


        print('Número de variables sin categorizar: {}'.format(len(colname_except)))
        print(colname_except)        
        print('Discretización de {0} variables con sobre {1}% de nulos: OK.'.format(len(self.categorizadas_nulos),corte_nulos * 100))
        imprimir_linea()
        return None
    
# ==================================================================================================

    def imputacion_nulos(self, estadistico_o_metodo='media', n_vecinos=3, imputar_0=[]):
        """Imputa los nulos por la media o mediana de 
        la variable solo considerando los Malos.
        Args:
            variable (_str_): Nombre de la variable.
            estadistico (_str_): Estadistico a utilizar, "media" o "mediana"
        """
        if len(imputar_0)!=0:
            self.vec_var_imput_0=[]
            for i in imputar_0:
                self.vec_var_imput_0 = self.vec_var_imput_0+[col for col in self.dataset.columns if i in col]

            for j in self.vec_var_imput_0:
                self.dataset[j].fillna(0, inplace = True)
        else:
            col_nulos = set(self.dataset.columns) - set(
                    self.no_considerar + \
                    self.listado_categoricas + \
                    self.categorizadas_nulos
            )

            if estadistico_o_metodo in ('media', 'mediana'):
                
                for colname in tqdm(col_nulos):
                    if estadistico_o_metodo == 'mediana':
                        #imputacion = self.dataset[self.dataset[self.muestra] == 'DEV'][colname].median()
                        imputacion = self.dataset[colname].median()
                    if estadistico_o_metodo=='media':
                        estadistico = 'media'
                        #imputacion = self.dataset[self.dataset[self.muestra] == 'DEV'][colname].mean()
                        imputacion = self.dataset[colname].mean()

                    self.dataset[colname] = self.dataset[colname].fillna(imputacion)

            else:
                if estadistico_o_metodo in ('KNNImputer', 'KNN', 'knn'):
                    imputador = KNNImputer(n_neighbors=n_vecinos)

                if estadistico_o_metodo in ('IterativeImputer','iter','ii','II'):
                    imputador = IterativeImputer(random_state=7)

                columnas_transformar = col_nulos.copy()
                
                #imputador.fit(self.dataset[columnas_transformar][self.dataset[self.muestra] == 'DEV'])
                imputador.fit(self.dataset[columnas_transformar])

                df_transformado = pd.DataFrame(
                    imputador.transform(self.dataset[columnas_transformar]),
                    columns=columnas_transformar
                )

                self.dataset = pd.concat(
                    [df_transformado, 
                    self.dataset[list(set(self.listado_categoricas + \
                        self.categorizadas_nulos + self.no_considerar))]], axis=1
                    )

        print('Variables fueron imputadas por {}s: OK.'.format(estadistico_o_metodo))
        imprimir_linea()
        return None


# ==================================================================================================

    def aplicar_log(self):
        columnas_transformar = [k for k in self.dataset.columns if k not in \
            self.listado_categoricas + \
            self.no_considerar +\
            self.categorizadas_nulos
        ]
        for k in columnas_transformar:
            try:
                self.dataset[k] = np.log(
                    self.dataset[k] + np.abs(min(self.dataset[k])) + 1
                )
            except:
                print(k)

        print('Transformacion logaritmica: OK.')
        imprimir_linea()
        return None
        
# ==================================================================================================

    def rangos_continuas(self,ruta_df_woes='D:/SOCOFAR/clf_woe.joblib',considerar_cat=''):
        """Reemplaza por el limite inferior/superior aquellas variables fuera del rango de los valores utilizados en el woeizado al momento de entrenar el modelo
        """
        for var in [v for v in self.dataset[self.listado_continuas].columns if v not in (considerar_cat)]:

            self.clf = load(ruta_df_woes)

            lim_inf=self.clf.woe_df[self.clf.woe_df['Variable_Name']==var]['Category'].iloc[0].left
            lim_sup=self.clf.woe_df[self.clf.woe_df['Variable_Name']==var]['Category'].iloc[-1].right

            self.dataset[var]=np.where(self.dataset[var] < lim_inf, lim_inf+0.01, self.dataset[var])
            self.dataset[var]=np.where(self.dataset[var] > lim_sup, lim_sup-0.01, self.dataset[var])

        return None   

    def Woeizar2(self,ruta_exporta='D:/SOCOFAR/clf_woe.joblib'):
        """Genera un dataset_woes, contiene las variables woeizadas en remplazo de las variables brutas
        """
        #transforma int to floats para evitar errores
        #for i in self.dataset[self.listado_continuas].columns:
        #    if self.dataset[i].dtype=='int64':
        #        self.dataset[i]=self.dataset[i].astype(float)
        #

        self.df_woeizar= self.dataset
        if self.target!='': # Esta condición es para cuando se está realizando el Woeizado en la etapa de modelación
            self.clf = WOE()
            self.clf.fit(self.df_woeizar[[i for i in self.df_woeizar.columns if i not in (self.no_considerar)]], self.df_woeizar[self.target])

            #codigo para subsanar problema de tramos que no se especifican y se rellenan con 'NA'
            #for name in self.clf.woe_df[(self.clf.woe_df.Category=='NA')]['Variable_Name']: # variables con categorias con tramos nulos
            #    lim_inf=self.clf.woe_df[(self.clf.woe_df.Variable_Name==name) & (self.clf.woe_df.Category!='NA')]['Category'].iloc[0].left #toma limite inferior de categoria no nula
            #    self.clf.woe_df[(self.clf.woe_df['Variable_Name']==name)] = self.clf.woe_df.replace('NA',pd.Interval(left=-100, right=lim_inf)) #remplaza NA's de la categoría por tramo entre '100' y el limite inferior de la otra categoria

            dump(self.clf, ruta_exporta)
            self.dataset=self.clf.transform(self.df_woeizar)
        else: # Condición para cuando se está Woeizando en productivo, basado en los woes de la modelación.
            self.clf = load(ruta_exporta)
            self.dataset=self.clf.transform(self.df_woeizar)

        print('Woeizado de Variables: OK.')
        imprimir_linea()           

        return None
# ==================================================================================================

    def feature_scaling(self, metodo='estandar'):

        columnas_transformar = [k for k in self.dataset.columns if k not in \
            (
                self.listado_categoricas + \
                self.no_considerar + \
                self.categorizadas_nulos
            )
        ]

        if metodo == 'estandar':
            escalador = StandardScaler()
        if metodo == 'min_max':
            escalador = MinMaxScaler()    

        escalador.fit(
            #self.dataset[self.dataset[self.muestra] == 'DEV'][columnas_transformar]
            self.dataset[columnas_transformar]
        )

        self.df_transformado = pd.DataFrame(
            escalador.transform(self.dataset[columnas_transformar]), 
            columns=columnas_transformar
        )

        self.dataset = pd.concat(
            [self.df_transformado, 
            self.dataset[list(set(self.listado_categoricas + \
                self.categorizadas_nulos + self.no_considerar))]], axis=1
            )

        print('Transformacion {}: OK.'.format(metodo))
        imprimir_linea()
        return None

# ================================================================================================== 

    def pca(self,n_components=0.9):

        ACP = PCA(n_components = n_components)
        variables_independientes = [str(k).upper() for k in (self.dataset.columns) if k not in (list(self.target)+self.no_considerar)]
        ACP.fit(self.dataset[variables_independientes])
        X_TRANSFORMED_PCA = ACP.transform(self.dataset[variables_independientes])

        cum_explained_var = []
        for i in range(0, len(ACP.explained_variance_ratio_)):
            if i == 0:
                cum_explained_var.append(ACP.explained_variance_ratio_[i])
            else:
                cum_explained_var.append(ACP.explained_variance_ratio_[i] + 
                                        cum_explained_var[i-1])

        print('TOTAL DE VARIABLES INICIALES: ')
        print(len(self.dataset[variables_independientes].columns))
        print('VARIABLES PCA para n_components = '+ str(n_components)+': ')
        print(len(cum_explained_var))
        print('MATRIZ DE VARIANZAS EXPLICADAS ACUMULADAS POR COMPONENTE:')
        print(cum_explained_var)

        plt.plot(np.cumsum(ACP.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        imprimir_linea()

        columns = ['pca_%i' % i for i in range(len(cum_explained_var))]
        df_pca = pd.DataFrame(data = X_TRANSFORMED_PCA, columns = columns)
        self.dataset = pd.concat([df_pca, self.dataset[self.target]], axis=1)
        return None
# ==================================================================================================
        
    def dumifica(self, botar_primera_col=True):
        """Dummyfica una lista de variables y la replaza en el dataset

        Args:
            categoricas (_type_): _description_

        Returns:
            _type_: _description_
        """        
        var_dummies = list(
            set(
                self.categorizadas_nulos+self.listado_categoricas
            ) - set(self.no_considerar)
        )

        if len(var_dummies) != 0:
            self.dataset = pd.get_dummies(
                self.dataset, columns=var_dummies, drop_first=botar_primera_col
            )

        regex = re.compile(r"\[|\]|<", re.IGNORECASE)
        self.dataset.columns = [regex.sub(")", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in self.dataset.columns.values]

        self.listado_dummies = [str(k).upper() for k in (self.dataset.columns) if k not in (list(self.target)+self.no_considerar+self.listado_continuas)]

        print('Variables dummies: OK.')
        imprimir_linea()
        return None

# ==================================================================================================

    def separa_muestras(self,method=None,test=0.3):
        x=[x for x in self.dataset.columns if x not in (self.no_considerar)]
        y=[self.target]

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.dataset[x]
                                                                                ,self.dataset[y].values.ravel()
                                                                                ,stratify=self.dataset[y].values.ravel()
                                                                                ,test_size = test
                                                                                ,random_state=0)
        
        self.X_train_id, self.X_test_id, self.Y_train_id, self.Y_test_id = train_test_split(self.dataset
                                                                                ,self.dataset[y].values.ravel()
                                                                                ,stratify=self.dataset[y].values.ravel()
                                                                                ,test_size = test
                                                                                ,random_state=0)

        if method=='undersampling':
            under_sampler = RandomUnderSampler()
            self.X_train,self.Y_train = under_sampler.fit_resample(self.X_train,self.Y_train)

        print('Separacion de muestras: OK.')
        imprimir_linea()
        return None

# ==================================================================================================

    def descarte_variables(
            self,
            modelo='RandomForest',
            metrica = 'roc_auc',
            cv = 3,
        ):

        print('Descartando Variables...')

        param_grid = {
              'n_jobs': [-1],
              'oob_score': [True], #,False
              'n_estimators': [100,500], # 100,500,1000
              'class_weight': [None], #'balanced','balanced_subsample'
              'criterion': ['gini','entropy'], #'entropy'
              'bootstrap': [True], #,False
              'random_state': [0]
              }
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3,cv=cv, scoring=metrica)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.modelo_descarte = grid.fit(self.X_train,self.Y_train)

        print('best params: '+ str(self.modelo_descarte.best_params_))
        print('best estimator: '+ str(self.modelo_descarte.best_estimator_))

        feature_names = self.X_train.columns
        self.feature_importance=pd.concat([pd.DataFrame(feature_names,columns=['variable']),
                               pd.DataFrame(self.modelo_descarte.best_estimator_.feature_importances_,columns=['f_importance'])],axis=1)\
                               .sort_values(by='f_importance', ascending=False)
        
        print('Descarte de variables: OK.')
        imprimir_linea()
        return None

# ==================================================================================================

    def modelacion(self, algoritmo='LogisticRegression',perf_eval='roc_auc',ruta_exporta='D:/SOCOFAR/modelo.joblib'):
        
        self.algoritmo=algoritmo
        #####################== LOGISTIC REGRESSION ==#####################
        if algoritmo == 'LogisticRegression':
            param_grid = {
                'penalty': ['l2','elasticnet','l1'],
                'C': [0.1,1,10],
                'class_weight': [None,'balanced'],
                'solver': ['saga','lbfgs','sag'],
                'random_state': [0]
                }
            grid = GridSearchCV(LogisticRegression(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        #####################== DECISION TREE ==#####################
        if algoritmo == 'DecisionTree':
            param_grid = {
                'splitter': ['best'], #'random'
                'criterion': ['gini'], #'entropy'
                'class_weight': [None], #,'balanced'
                'max_depth': [15], #2,6,10,
                'min_samples_leaf': [1],
                'random_state': [0]
                }
            grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        ###################== K-Nearest Neighbor ==##################
        if algoritmo == 'K-Nearest Neighbor':
            param_grid = {
                'n_neighbors': [3], #,5,10
                'weights': ['distance'], #'uniform',
                'algorithm': ['auto'], #,'ball_tree','kd_tree','brute'
                'leaf_size': [30], #,15,60
                'p': [2], #,1
                'metric': ['minkowski'], #,'euclidean','manhattan','seuclidean'
                }
            grid = GridSearchCV(neighbors.KNeighborsClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        #####################== RANDOM FOREST ==#####################
        if algoritmo == 'RandomForest':
            param_grid = {
              'n_jobs': [-1],
              'oob_score': [True], #,False
              'n_estimators': [50], # 100,500,1000
              'class_weight': ['balanced'], #'None','balanced','balanced_subsample'
              'criterion': ['gini'], #'entropy'
              'bootstrap': [True], #,False
              'random_state': [0]
              }
            grid = GridSearchCV(RandomForestClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        #####################== SUPPORT VECTOR MACHINE ==#####################
        if algoritmo == 'SVC':
            param_grid = {'C': [1], #1, 10, 100, 1000
              'gamma': ['scale'], #1,0.1, 0.01, 0.001, 0.0001
              'kernel': ['rbf'], #,'sigmoid'
              'probability': [True],
              'class_weight': ['balanced']
              }
            grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
         #####################== XGBOOST ==#####################
        if algoritmo == 'XGBoost':
            param_grid = {
                'min_child_weight': [1], #, 5, 10
                'gamma': [0.5], #, 1, 1.5, 2, 5
                'subsample': [0.6], #, 0.8, 1.0
                'colsample_bytree': [0.6], #, 0.8, 1.0
                'max_depth': [5] #3, 4
                }
            xgboost = xgb.XGBClassifier(learning_rate=0.3, n_estimators=100, objective='binary:logistic',
                    silent=True, nthread=1)
            grid = GridSearchCV(xgboost, param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)    
        ###################== Bagging Classifier ==##################
        if algoritmo == 'Bagging Classifier':
            param_grid = {
                'n_estimators': [100],
                #'max_samples': [0.1],
                'random_state': [0]
                }
            grid = GridSearchCV(DecisionTreeClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        ###################== Gradient Boosting Classifier ==##################
        if algoritmo == 'Gradient Boosting Classifier':
            param_grid = {
                'n_estimators': [100],
                'random_state': [0]
                }
            grid = GridSearchCV(GradientBoostingClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        ###################== Extra Trees Classifier ==##################
        if algoritmo == 'Extra Trees Classifier':
            param_grid = {
                'n_estimators': [1000],
                'n_jobs': [8],
                'class_weight': ['balanced'],
                'criterion': ['gini'],
                'random_state': [0]
                }
            grid = GridSearchCV(ExtraTreesClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        ###################== AdaBoost ==##################
        if algoritmo == 'AdaBoost':
            param_grid = {
                'n_estimators': [100],
                'learning_rate': [1],
                'algorithm': ['SAMME.R'],
                'random_state': [0]
                }
            grid = GridSearchCV(AdaBoostClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        ###################== MLPClassifier ==##################
        if algoritmo == 'MLPClassifier':
            param_grid = {
                'activation': ['logistic'], #{‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
                'solver': ['sgd'], #{‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
                'alpha': [0.0001], #default 0.0001
                'batch_size': ['auto'], #int, optional, default ‘auto’
                'learning_rate': ['constant'], #{‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
                'power_t': [0.5], #double, optional, default 0.5 // solver=’sgd’.
                'max_iter': [200], #int, optional, default 200
                'tol': [0.0001], #float, optional, default 1e-4
                'warm_start': [False], #bool, optional, default False
                'momentum': [0.9], #float, default 0.9 // solver=’sgd’
                'nesterovs_momentum': [True], #boolean, default True // solver=’sgd’ and momentum > 0
                'early_stopping': [False], #bool, default False // solver=’sgd’ or ‘adam’
                'validation_fraction': [0.1], #float, optional, default 0.1 // early_stopping=True
                'beta_1': [0.9], #float, optional, default 0.9 // solver=’adam’
                'beta_2': [0.999], #float, optional, default 0.999 // solver=’adam’
                'epsilon': [1e-8], #float, optional, default 1e-8 // solver=’adam’
                'n_iter_no_change': [10], #int, optional, default 10 // solver=’sgd’ or ‘adam’
                'hidden_layer_sizes': [(500,100)], #tuple, length = n_layers - 2, default (100,)
                'random_state': [0]
                }

            grid = GridSearchCV(MLPClassifier(), param_grid, refit = True, verbose = 3,cv=3, scoring=perf_eval)
        ###################== FIN DEFINICION MODELOS ==##################

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.model = grid.fit(self.X_train,self.Y_train)

        dump(self.model, ruta_exporta) 

        print('best params: '+ str(self.model.best_params_))
        print('best estimator: '+ str(self.model.best_estimator_))

        self.pred_prob_model=self.model.predict_proba(self.X_test)
        print('ROC: '+str(roc_auc_score(self.Y_test,self.pred_prob_model[:,1])))


        #se almacenan las feature importance del Random Forest por si se desea realizar descarte de variables
        if algoritmo == 'RandomForest':
            feature_names = self.X_train.columns
            self.rf_feature_importance=pd.concat([pd.DataFrame(feature_names,columns=['variable']),
                               pd.DataFrame(self.model.best_estimator_.feature_importances_,columns=['f_importance'])],axis=1)\
                               .sort_values(by='f_importance', ascending=False)

        print('Modelación: OK.')
        imprimir_linea()
        return None

# ==================================================================================================

    def metricas_modelo_optimo(self,metrica='roc'):

        df_pred_real = {'inc_pred_proba': self.pred_prob_model[:,1], self.target: self.Y_test}
        df_pred_real = pd.DataFrame(df_pred_real,columns=['inc_pred_proba',self.target])
        self.confusion_matrix,self.threshold=modelo_optimos(df_pred_real,metrica)

        return None        

    def met(self,metrica='roc'):

        df_pred_real = {'inc_pred_proba': self.dataset['P1'], self.target: self.dataset['TARGET']}
        df_pred_real = pd.DataFrame(df_pred_real,columns=['inc_pred_proba','TARGET'])
        self.confusion_matrix,self.threshold=modelo_optimos(df_pred_real,metrica)

        return None 