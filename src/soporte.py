
# Para gestionar el feature scaling
# -----------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler, RobustScaler

# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import math

import seaborn as sns
import matplotlib.pyplot as plt


def add_sufix(columnas, suf):

    salida = []
    for i in columnas:
        salida.append(i + suf)
    
    return salida


def standar_datos(df,columnas):

    escalador_robust = RobustScaler()
    datos_transf_robust = escalador_robust.fit_transform(df[columnas])
    columnas_rename = add_sufix(columnas, "_robust")
    df[columnas_rename] = datos_transf_robust

    escalador_min_max = MinMaxScaler()
    datos_transf_min_max = escalador_min_max.fit_transform(df[columnas])
    columnas_rename = add_sufix(columnas, "_min_max")
    df[columnas_rename]  = datos_transf_min_max

    escalador_norm = Normalizer()
    datos_transf_norm = escalador_norm.fit_transform(df[columnas])
    columnas_rename = add_sufix(columnas, "_norm")
    df[columnas_rename]  = datos_transf_norm

    escalador_estandar = StandardScaler()
    datos_transf_estandar = escalador_estandar.fit_transform(df[columnas])
    columnas_rename = add_sufix(columnas, "_estandar")
    df[columnas_rename]  = datos_transf_estandar

    return df

def visualizar_tablas(dataframe, lista_col):

    num_filas = math.ceil(len(lista_col) / 5)

    fig, axes = plt.subplots(nrows=num_filas, ncols=5, figsize=(25, 15))
    axes = axes.flat

    for indice, columna in enumerate(lista_col):
        sns.boxplot(x=columna, data=dataframe, ax=axes[indice])
        axes[indice].set_title(f"{columna}")
        axes[indice].set_xlabel("")

    # if len(lista_col) % 2 != 0:
    #     fig.delaxes(axes[-1])

    fig.suptitle("")
    plt.tight_layout()
    plt.show()