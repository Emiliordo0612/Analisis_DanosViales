# -*- coding: utf-8 -*-
"""
Proyecto de Clasificación de Daños en Carreteras - Versión Mejorada con Visualización
Técnicas implementadas:
- Clasificación de tipo de daño (5 categorías)
- Clasificación de gravedad (baja, media, alta)
- Visualización de imágenes de prueba
- Preprocesamiento adaptado a condiciones Saudíes
- Resultados detallados con ejemplos
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Declaraciones de variables quemadas con los directorios donde se alojan los recursos a usar para el entrenamiento del modelo
CARPETA_PROYECTO = '/home/emilio/roadnet_pruebaTecnica'
CARPETA_IMAGENES = os.path.join(CARPETA_PROYECTO, 'imagenes_extraidas')
CARPETA_CLASIFICADAS = os.path.join(CARPETA_PROYECTO, 'imagenes_clasificadas')
CARPETA_RESULTADOS = os.path.join(CARPETA_PROYECTO, 'resultados')

#Creacion de los directorios en caso de que no se encuentren creados para alojar las imagenes procesadas
os.makedirs(CARPETA_IMAGENES, exist_ok=True)
os.makedirs(CARPETA_CLASIFICADAS, exist_ok=True)
os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

def mostrar_ejemplos_antes_procesamiento(imagenes, num_ejemplos=5):
    """Muestra ejemplos de imágenes antes del procesamiento"""
    print("\n🖼️ Ejemplos de imágenes antes del procesamiento:")
    
    plt.figure(figsize=(15, 8))
    for i in range(min(num_ejemplos, len(imagenes))):
        plt.subplot(1, num_ejemplos, i+1)
        img_rgb = cv2.cvtColor(imagenes[i], cv2.COLOR_BGR2RGB) if len(imagenes[i].shape) == 3 else imagenes[i]
        plt.imshow(img_rgb, cmap='gray' if len(imagenes[i].shape) == 2 else None)
        plt.title(f"Imagen {i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'ejemplos_antes_procesamiento.jpg'))
    plt.show()

def preprocesar_imagen(img):
    """Preprocesamiento mejorado para condiciones Saudíes"""
    # Alternativa al balance de blancos
    if len(img.shape) == 3:  # Solo si es imagen a color
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # Convertir a escala de grises si no lo está
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Mejorar contraste
    eq = cv2.equalizeHist(gray)
    
    # Reducción de ruido adaptada
    blur = cv2.bilateralFilter(eq, 9, 75, 75)
    
    return cv2.resize(blur, (150, 150))

def mostrar_ejemplos_despues_procesamiento(imagenes, num_ejemplos=5):
    """Muestra ejemplos de imágenes después del procesamiento"""
    print("\n🖼️ Ejemplos de imágenes después del procesamiento:")
    
    plt.figure(figsize=(15, 8))
    for i in range(min(num_ejemplos, len(imagenes))):
        img_procesada = preprocesar_imagen(imagenes[i])
        plt.subplot(1, num_ejemplos, i+1)
        plt.imshow(img_procesada, cmap='gray')
        plt.title(f"Imagen {i+1} procesada")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'ejemplos_despues_procesamiento.jpg'))
    plt.show()

def calcular_gravedad(imagen):
    """Determina la gravedad del daño con visualización"""
    # Procesamiento para visualización
    if len(imagen.shape) == 2:
        img_vis = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        img_vis = imagen.copy()
    
    # Detección de bordes
    edges = cv2.Canny(imagen, 50, 150)
    
    # Calcular métricas
    area_total = imagen.shape[0] * imagen.shape[1]
    area_daño = np.sum(edges > 0) / area_total
    
    # Encontrar contornos y dibujarlos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_vis, contours, -1, (0, 255, 0), 1)
    
    ancho_promedio = 0
    if len(contours) > 0:
        ancho_promedio = np.mean([cv2.boundingRect(cnt)[2] for cnt in contours])
    
    # Determinar gravedad
    if area_daño < 0.05 or ancho_promedio < 3:
        gravedad = "baja"
        color = (0, 255, 0)  # Verde
    elif area_daño < 0.2 or ancho_promedio < 10:
        gravedad = "media"
        color = (0, 255, 255)  # Amarillo
    else:
        gravedad = "alta"
        color = (0, 0, 255)  # Rojo
    
    # Añadir texto de gravedad
    cv2.putText(img_vis, f"Gravedad: {gravedad}", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return gravedad, img_vis

def visualizar_clasificacion_gravedad(categorias, gravedades, num_ejemplos=3):
    """Muestra ejemplos visuales de clasificación con gravedad"""
    print("\n📊 Visualización de clasificación con gravedad:")
    
    fig, axs = plt.subplots(len(categorias), num_ejemplos, figsize=(15, 12))
    if len(categorias) == 1:
        axs = [axs]  # Para manejar caso de una sola categoría
    
    for i, categoria in enumerate(categorias):
        # Obtener imágenes de esta categoría
        ruta_categoria = os.path.join(CARPETA_CLASIFICADAS, categoria)
        imagenes_categoria = [f for f in os.listdir(ruta_categoria) if f.startswith('daño_')][:num_ejemplos]
        
        for j, img_nombre in enumerate(imagenes_categoria):
            img_path = os.path.join(ruta_categoria, img_nombre)
            img = cv2.imread(img_path)
            
            # Calcular gravedad con visualización
            gravedad, img_vis = calcular_gravedad(preprocesar_imagen(img))
            img_vis = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
            
            # Mostrar imagen
            ax = axs[i][j] if len(categorias) > 1 else axs[j]
            ax.imshow(img_vis)
            ax.set_title(f"{categoria}\n{img_nombre}\nGravedad: {gravedad}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'clasificacion_gravedad.jpg'))
    plt.show()

def extraer_imagenes(ruta_imagen):
    """Divide la imagen compuesta en imágenes individuales"""
    print("\n" + "="*50)
    print("Paso 1: Extrayendo imágenes individuales...")
    
    img = cv2.imread(ruta_imagen)
    if img is None:
        raise ValueError("No se pudo cargar la imagen. Verifica la ruta.")
    
    # Mostrar imagen original
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Imagen compuesta original")
    plt.axis('off')
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'imagen_original.jpg'))
    plt.show()
    
    # Dividir en 5 columnas y 6 filas (30 imágenes)
    imagenes = []
    alto, ancho = img.shape[:2]
    for fila in range(6):
        for col in range(5):
            x1 = col * (ancho // 5)
            x2 = (col + 1) * (ancho // 5)
            y1 = fila * (alto // 6)
            y2 = (fila + 1) * (alto // 6)
            
            img_recortada = img[y1:y2, x1:x2]
            nombre_archivo = f"img_{fila+1}_{col+1}.jpg"
            cv2.imwrite(os.path.join(CARPETA_IMAGENES, nombre_archivo), img_recortada)
            imagenes.append(img_recortada)
    
    # Mostrar ejemplos antes/durante el procesamiento
    mostrar_ejemplos_antes_procesamiento(imagenes)
    
    print(f"\n✅ {len(imagenes)} imágenes guardadas en {CARPETA_IMAGENES}")
    return imagenes

def clasificar_imagenes(imagenes):
    """Clasifica imágenes según el mapeo específico de filas a categorías"""
    print("\n" + "="*50)
    print("Paso 2: Clasificando imágenes por tipo y gravedad de daño...")
    
    # Mapeo de filas a categorías según tu especificación
    mapeo_categorias = {
        0: 'grietas_longitudinales_transversales',  # Fila 1 (D0)
        1: 'grietas_cocodrilo',                    # Fila 2 (D1)
        2: 'grietas_borde',                        # Fila 3 (D2)
        3: 'baches',                               # Fila 4 (D3)
        4: 'depresiones',                          # Fila 5 (D4)
        5: 'empuje'                                # Fila 6 (D5)
    }
    
    categorias = list(mapeo_categorias.values())
    niveles_gravedad = ['baja', 'media', 'alta']
    
    # Crear subcarpetas para cada categoría
    for categoria in categorias:
        os.makedirs(os.path.join(CARPETA_CLASIFICADAS, categoria), exist_ok=True)
    
    gravedades = {}
    
    # Asignación según filas (cada fila = 1 categoría específica)
    for fila in range(6):  # 6 filas en la imagen compuesta
        categoria_actual = mapeo_categorias[fila]
        
        # Procesar las 5 columnas de esta fila
        for col in range(5):
            indice_imagen = fila * 5 + col
            if indice_imagen >= len(imagenes):
                continue
                
            img = imagenes[indice_imagen]
            nombre_archivo = f"daño_f{fila+1}c{col+1}.jpg"  # Ej: daño_f1c3.jpg (fila 1, col 3)
            ruta_guardado = os.path.join(CARPETA_CLASIFICADAS, categoria_actual, nombre_archivo)
            
            # Guardar imagen
            cv2.imwrite(ruta_guardado, img)
            
            # Calcular y almacenar gravedad
            gravedad, _ = calcular_gravedad(preprocesar_imagen(img))
            gravedades[nombre_archivo] = {
                'categoria': categoria_actual,
                'gravedad': gravedad,
                'ubicacion_original': f"Fila {fila+1}, Col {col+1}"  # Para trazabilidad
            }
    
    # Mostrar resumen de clasificación
    print("\nResumen de clasificación por filas:")
    for fila, categoria in mapeo_categorias.items():
        print(f"Fila {fila+1}: {categoria} ({len([k for k,v in gravedades.items() if v['categoria']==categoria])} imágenes)")
    
    # Mostrar ejemplos de clasificación con gravedad
    visualizar_clasificacion_gravedad(categorias, gravedades)
    
    print("\n✅ Clasificación completada según mapeo filas->categorías")
    return categorias, niveles_gravedad, gravedades

def entrenar_y_evaluar(X, y, categorias, gravedades):
    """Entrena y evalúa modelos mostrando tipo y gravedad"""
    print("\n" + "="*50)
    print("Paso 3: Entrenando y evaluando modelos...")
    
    # Preprocesamiento
    X_flat = X.reshape(len(X), -1)
    
    # Verificar distribución de clases
    print("\n📈 Distribución de clases en los datos:")
    for i, cat in enumerate(categorias):
        print(f"{cat}: {np.sum(y == i)} muestras")
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.3, random_state=42, stratify=y)
    
    # 1. Árbol de Decisión
    print("\n🔍 Evaluando Árbol de Decisión...")
    arbol = DecisionTreeClassifier(
        max_depth=5, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    arbol.fit(X_train, y_train)
    
    # Visualizar árbol (primeros 2 niveles)
    plt.figure(figsize=(20, 10))
    plot_tree(arbol, filled=True, feature_names=[f"pixel_{i}" for i in range(X_flat.shape[1])], 
              class_names=categorias, max_depth=2, fontsize=8)
    plt.title("Árbol de Decisión (primeros 2 niveles)")
    plt.savefig(os.path.join(CARPETA_RESULTADOS, 'arbol_decision.jpg'))
    plt.show()
    
    # 2. KNN
    print("\n🔍 Evaluando KNN...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='euclidean'
    )
    knn.fit(X_train_scaled, y_train)
    
    # Evaluación comparativa
    modelos = {
        "Árbol de Decisión": arbol,
        "KNN": knn
    }
    
    for nombre, modelo in modelos.items():
        print(f"\n📊 Resultados de {nombre}:")
        if nombre == "KNN":
            y_pred = modelo.predict(X_test_scaled)
        else:
            y_pred = modelo.predict(X_test)
        
        # Mostrar reporte de clasificación
        print(classification_report(
            y_test, y_pred, 
            target_names=categorias, 
            zero_division=1
        ))
        
        # Matriz de confusión
        plt.figure(figsize=(8,6))
        sns.heatmap(confusion_matrix(y_test, y_pred), 
                    annot=True, fmt='d', 
                    xticklabels=categorias, 
                    yticklabels=categorias,
                    cmap='Blues')
        plt.title(f"Matriz de Confusión - {nombre}")
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(os.path.join(CARPETA_RESULTADOS, f'matriz_confusion_{nombre.lower().replace(" ", "_")}.jpg'))
        plt.show()
        
        # Mostrar ejemplos con gravedad
        print("\n🔍 Ejemplos de clasificación con gravedad:")
        for i in range(min(3, len(X_test))):
            idx = np.where(X_flat == X_test[i][0])[0][0]
            img_original = X[idx//X_flat.shape[1]]
            pred_categoria = categorias[y_pred[i]]
            real_categoria = categorias[y_test[i]]
            
            # Calcular gravedad con visualización
            gravedad, img_vis = calcular_gravedad(img_original.reshape(150, 150))
            
            # Mostrar imagen con resultados
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(img_original.reshape(150, 150), cmap='gray')
            plt.title(f"Real: {real_categoria}")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
            plt.title(f"Pred: {pred_categoria}\nGravedad: {gravedad}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(CARPETA_RESULTADOS, f'ejemplo_resultado_{i}.jpg'))
            plt.show()
    
    return modelos

def main():
    try:
        print("="*50)
        print("INICIO DEL PROCESO DE CLASIFICACIÓN DE DAÑOS EN CARRETERAS")
        print("="*50)
        
        # Ruta a la imagen compuesta
        ruta_imagen_compuesta = os.path.join(CARPETA_PROYECTO, 'imagen', 'roadnet_imagenCalles.png')
        
        # 1. Extraer imágenes individuales
        imagenes = extraer_imagenes(ruta_imagen_compuesta)
        
        # Mostrar ejemplos después del procesamiento
        mostrar_ejemplos_despues_procesamiento(imagenes)
        
        # 2. Clasificar imágenes (tipo + gravedad)
        categorias, niveles_gravedad, gravedades = clasificar_imagenes(imagenes)
        
        # 3. Cargar imágenes clasificadas para entrenamiento
        X, y = [], []
        min_muestras = 2
        
        for i, categoria in enumerate(categorias):
            ruta_categoria = os.path.join(CARPETA_CLASIFICADAS, categoria)
            imagenes_categoria = [f for f in os.listdir(ruta_categoria) if f.startswith('daño_')]
            
            if len(imagenes_categoria) < min_muestras:
                print(f"⚠️ Advertencia: Categoría {categoria} tiene pocas muestras ({len(imagenes_categoria)})")
                continue
                
            for img_nombre in imagenes_categoria:
                img = cv2.imread(os.path.join(ruta_categoria, img_nombre))
                if img is not None:
                    img_procesada = preprocesar_imagen(img)
                    X.append(img_procesada)
                    y.append(i)
        
        X = np.array(X)
        y = np.array(y)
        
        if len(X) == 0:
            raise ValueError("No se encontraron imágenes válidas para entrenamiento")
        
        # 4. Entrenar y evaluar modelos
        modelos = entrenar_y_evaluar(X, y, categorias, gravedades)
        
        # Resumen final
        print("\n" + "="*50)
        print("✅ ¡Proceso completado exitosamente!")
        print("="*50)
        
        print("\n📋 Resumen de clasificación de gravedad en muestras:")
        for img, datos in list(gravedades.items())[:5]:
            print(f"- {img}: {datos['categoria']} (Gravedad: {datos['gravedad']})")
        
        print(f"\n📁 Todos los resultados se han guardado en: {CARPETA_RESULTADOS}")
    
    except Exception as e:
        print("\n" + "="*50)
        print(f"❌ Error: {str(e)}")
        print("="*50)
        print("\nPosibles soluciones:")
        print("1. Verificar que la imagen existe en la ruta correcta")
        print("2. Asegurar permisos de escritura en las carpetas")
        print("3. Verificar que OpenCV esté instalado correctamente")
        print("4. Revisar que haya imágenes en cada categoría")

if __name__ == "__main__":
    main()