import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, YearLocator

## ATENCION: Debe colocar la direccion en la que ha sido publicada la aplicacion en la siguiente linea\
# url = 'https://tp8-555555.streamlit.app/'

def mostrar_informacion_alumno():
    with st.container(border=True):
        st.markdown('**Legajo:** 58879')
        st.markdown('**Nombre:** Andrés Ruiu')
        st.markdown('**Comisión:** C9')

def cargar_datos(archivo_csv):
    if archivo_csv is not None:
        df = pd.read_csv(archivo_csv)
        return df
    return None

def calcular_metricas(df, producto=None):
    # Conversión de fecha más eficiente
    df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str), format='%Y-%m')
    
    # Cálculos vectorizados con manejo de división por cero
    df['Precio_promedio'] = np.divide(df['Ingreso_total'], df['Unidades_vendidas'], 
                                       out=np.zeros_like(df['Ingreso_total'], dtype=float), 
                                       where=df['Unidades_vendidas']!=0)
    
    df['Margen_promedio'] = np.divide(df['Ingreso_total'] - df['Costo_total'], 
                                       df['Ingreso_total'], 
                                       out=np.zeros_like(df['Ingreso_total'], dtype=float), 
                                       where=df['Ingreso_total']!=0)
    
    # Ordenamiento más eficiente
    df_ordenado = df.sort_values(['Producto', 'Año', 'Mes'])
    
    # Uso de transform para cálculos de columnas anteriores
    df_ordenado['Precio_promedio_anterior'] = df_ordenado.groupby('Producto')['Precio_promedio'].transform(lambda x: x.shift(1))
    df_ordenado['Margen_promedio_anterior'] = df_ordenado.groupby('Producto')['Margen_promedio'].transform(lambda x: x.shift(1))
    df_ordenado['Unidades_vendidas_anterior'] = df_ordenado.groupby('Producto')['Unidades_vendidas'].transform(lambda x: x.shift(1))
    
    def calcular_variacion(valor_actual, valor_anterior):
        # Función vectorizada para cálculo de variación
        return np.where(valor_anterior != 0, 
                        (valor_actual - valor_anterior) / valor_anterior * 100, 
                        0)
    
    if producto:
        df_producto = df_ordenado[df_ordenado['Producto'] == producto]
        
        metricas_producto = df_producto.groupby('Producto').agg(
            precio_promedio=pd.NamedAgg(column='Precio_promedio', aggfunc='mean'),
            precio_promedio_anterior=pd.NamedAgg(column='Precio_promedio_anterior', aggfunc='mean'),
            margen_promedio=pd.NamedAgg(column='Margen_promedio', aggfunc='mean'),
            margen_promedio_anterior=pd.NamedAgg(column='Margen_promedio_anterior', aggfunc='mean'),
            unidades_vendidas=pd.NamedAgg(column='Unidades_vendidas', aggfunc='sum'),
            unidades_vendidas_anterior=pd.NamedAgg(column='Unidades_vendidas_anterior', aggfunc='sum')
        ).reset_index()
        
        row = metricas_producto.iloc[0]
        
        # Uso de función de variación vectorizada
        var_precio = calcular_variacion(row['precio_promedio'], row['precio_promedio_anterior'])
        var_margen = (row['margen_promedio'] - row['margen_promedio_anterior']) * 100
        var_unidades = calcular_variacion(row['unidades_vendidas'], row['unidades_vendidas_anterior'])
        
        return (
            row['precio_promedio'], var_precio,
            row['margen_promedio'] * 100, var_margen,
            row['unidades_vendidas'], var_unidades
        )
    
    return df_ordenado.groupby('Producto').agg(
        precio_promedio=pd.NamedAgg(column='Precio_promedio', aggfunc='mean'),
        precio_promedio_anterior=pd.NamedAgg(column='Precio_promedio_anterior', aggfunc='mean'),
        margen_promedio=pd.NamedAgg(column='Margen_promedio', aggfunc='mean'),
        margen_promedio_anterior=pd.NamedAgg(column='Margen_promedio_anterior', aggfunc='mean'),
        unidades_vendidas=pd.NamedAgg(column='Unidades_vendidas', aggfunc='sum'),
        unidades_vendidas_anterior=pd.NamedAgg(column='Unidades_vendidas_anterior', aggfunc='sum')
    ).reset_index()

def calcular_tendencia(x, y):
    # Usar polyfit de NumPy para estimación de tendencia más robusta
    return np.polyfit(x, y, 1)

def crear_grafico_ventas(df, producto):
    df_producto = df[df['Producto'] == producto].copy()
    
    # Conversión de fecha más eficiente
    df_producto['Fecha'] = pd.to_datetime(df_producto['Año'].astype(str) + '-' + 
                                        df_producto['Mes'].astype(str), format='%Y-%m')
    
    ventas_mensuales = df_producto.groupby('Fecha')['Unidades_vendidas'].sum().reset_index()
    
    # Usar numpy para cálculos
    x = np.arange(len(ventas_mensuales))
    y = ventas_mensuales['Unidades_vendidas'].values
    
    # Usar polyfit para cálculo de tendencia
    slope, intercept = calcular_tendencia(x, y)
    tendencia = slope * x + intercept
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ventas_mensuales['Fecha'], ventas_mensuales['Unidades_vendidas'], 
            label=producto, color='#5192be', linewidth=2)
    
    ax.plot(ventas_mensuales['Fecha'], tendencia, 
            label='Tendencia', color='red', linestyle='--', linewidth=2)
    
    ax.set_title('Evolución de Ventas Mensual')
    ax.set_xlabel('Año-Mes')
    ax.set_ylabel('Unidades Vendidas')
    
    ax.set_ylim(bottom=0)
    
    ax.grid(True, which='major', axis='both', linestyle='-', alpha=0.5, linewidth=2.0) 
    ax.grid(True, which='minor', axis='x', linestyle='-', alpha=0.2, linewidth=0.5)
    
    ax.xaxis.set_minor_locator(MonthLocator())
    ax.xaxis.set_major_locator(YearLocator())
    
    ax.legend()
    
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def main():
    # Usar todo el ancho disponible
    st.set_page_config(layout="wide")

    with st.sidebar:
        st.title("Cargar archivo de datos")
        archivo_csv = st.file_uploader("Subir archivo CSV", type=['csv'])
        
        if archivo_csv is not None:
            df = cargar_datos(archivo_csv)
            sucursales = ['Todas'] + list(df['Sucursal'].unique())
            sucursal_seleccionada = st.selectbox('Seleccionar Sucursal', sucursales)
        else:
            sucursal_seleccionada = None
            df = None

    if df is None:
        st.header("Por favor, sube un archivo CSV desde la barra lateral")
        mostrar_informacion_alumno()
    
    if df is not None:
        if sucursal_seleccionada != 'Todas':
            df = df[df['Sucursal'] == sucursal_seleccionada]
        
        st.title("**Datos de Todas las Sucursales**" if sucursal_seleccionada == 'Todas' 
                else f"Datos de {sucursal_seleccionada}")
        
        for producto in df['Producto'].unique():
            with st.container(border=True):
                st.subheader(producto)
                
                (precio_promedio, var_precio, 
                margen_promedio, var_margen,
                unidades_vendidas, var_unidades) = calcular_metricas(df, producto)
                
                col_metricas, col_grafico = st.columns([1, 2], gap="large") 
                
                with col_metricas:
                    st.write("### Métricas")
                    st.metric("Precio Promedio", 
                            f"${precio_promedio:,.0f}", 
                            f"{var_precio:+.2f}%")
                    
                    st.metric("Margen Promedio",
                            f"{margen_promedio:.0f}%",
                            f"{var_margen:+.2f}%")
                    
                    st.metric("Unidades Vendidas",
                            f"{unidades_vendidas:,.0f}",
                            f"{var_unidades:+.2f}%")
                
                with col_grafico:
                    st.write("### Evolución de Ventas")
                    fig = crear_grafico_ventas(df, producto)
                    fig.set_figwidth(10)
                    fig.set_figheight(6)
                    st.pyplot(fig)
                
                plt.close()

if __name__ == "__main__":
    main()