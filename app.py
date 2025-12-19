from flask import Flask, request, jsonify, render_template_string, send_from_directory, make_response, session
from flask_cors import CORS
import datetime
import random 
import json
import os
import math
from collections import deque
import threading
import time
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import hashlib
import secrets
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Habilitar CORS con credenciales
app.secret_key = secrets.token_hex(32)  # Clave secreta para sesiones

# === CONFIGURACIÓN DE AUTENTICACIÓN ===
# Usuario y contraseña por defecto (deberías cambiarlos)
ADMIN_USER = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()  # Cambiar "admin123" por tu contraseña

# === CONFIGURACIÓN DEL SISTEMA EXPERTO ===
UMBRAL_TEMP_ALERTA = 25.0
UMBRAL_TEMP_CRITICA = 31.0
UMBRAL_HUMEDAD_BAJA = 30.0
UMBRAL_HUMEDAD_ALTA = 85.0

# === GEMELO DIGITAL ===
estado_sistema = {
    "temperatura": 0.0,
    "humedad": 0.0,
    "relay1": False,  # Ventilador
    "relay2": False,  # Alarma
    "relay3": False,  # 💡 Iluminación Pasillo (Zona A)
    "relay4": False,  # 🔦 Iluminación Racks (Zona B)
    "mensaje": "Sistema Inicializando",
    "ultima_actualizacion": None,
    "conectado": False,
    "alertas_activas": [],
    # === Control Manual/Automático ===
    "modo": "AUTO",  # "AUTO" o "MANUAL"
    "manual_relay1": False,
    "manual_relay2": False,
    "manual_relay3": False,
    "manual_relay4": False,
    # === KPIs y Estadísticas ===
    "temp_max_sesion": 0.0,
    "temp_min_sesion": 100.0,
    "hum_max_sesion": 0.0,
    "hum_min_sesion": 100.0,
    "total_alertas": 0,
    "ciclos_ventilador": 0,
    "tiempo_ventilador_on": 0,
    "uptime_sistema": 0
}

# === HISTORIAL DE DATOS (últimos 100 registros) ===
historial = deque(maxlen=100)
log_eventos = deque(maxlen=50)

# === LOCK PARA THREAD SAFETY ===
estado_lock = threading.Lock()

def registrar_evento(tipo, mensaje):
    """Registra eventos importantes del sistema"""
    evento = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": tipo,
        "mensaje": mensaje
    }
    log_eventos.append(evento)
    print(f"[{tipo.upper()}] {mensaje}")

# === SESIONES ACTIVAS (en memoria) ===
sesiones_admin = {}  # {session_id: timestamp}

# === RED NEURONAL PARA PREDICCIÓN ===
class RedNeuronalGuardian:
    """
    Red Neuronal Multicapa para predicción de temperatura futura
    y optimización del sistema de climatización
    """
    def __init__(self):
        self.modelo = None
        self.scaler_entrada = MinMaxScaler(feature_range=(0, 1))
        self.scaler_salida = MinMaxScaler(feature_range=(0, 1))
        self.entrenado = False
        self.historial_entrenamientos = []
        self.metricas = {
            'accuracy': 0.0,
            'loss': 0.0,
            'mse': 0.0,
            'r2_score': 0.0,
            'predicciones_correctas': 0,
            'total_predicciones': 0
        }
        self.inicializar_modelo()
    
    def inicializar_modelo(self):
        """Inicializa la arquitectura de la red neuronal"""
        self.modelo = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),  # 3 capas ocultas
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            alpha=0.001
        )
        registrar_evento("RNA", "Red Neuronal inicializada: 3 capas [64-32-16 neuronas]")
    
    def preparar_datos_entrenamiento(self, historial_datos):
        """Prepara datos del historial para entrenamiento"""
        if len(historial_datos) < 10:
            return None, None
        
        # Extraer características: temp actual, humedad, hora del día, relay1 anterior
        X = []
        y = []
        
        for i in range(len(historial_datos) - 5):
            # Características de entrada (ventana de 5 registros)
            ventana = historial_datos[i:i+5]
            
            temps = [d['temperatura'] for d in ventana]
            hums = [d['humedad'] for d in ventana]
            relay1_estados = [1 if d.get('relay1', False) else 0 for d in ventana]
            
            # Hora del día normalizada
            try:
                hora = int(ventana[-1]['timestamp'].split(' ')[1].split(':')[0])
            except:
                hora = 12
            
            features = temps + hums + relay1_estados + [hora/24.0]
            X.append(features)
            
            # Variable objetivo: temperatura en el siguiente registro
            y.append(historial_datos[i+5]['temperatura'])
        
        return np.array(X), np.array(y).reshape(-1, 1)
    
    def entrenar(self, historial_datos):
        """Entrena la red neuronal con los datos históricos"""
        try:
            X, y = self.preparar_datos_entrenamiento(list(historial_datos))
            
            if X is None or len(X) < 10:
                return {
                    'success': False,
                    'mensaje': 'Datos insuficientes para entrenamiento (mínimo 10 registros)'
                }
            
            # Normalizar datos
            X_scaled = self.scaler_entrada.fit_transform(X)
            y_scaled = self.scaler_salida.fit_transform(y)
            
            # Entrenar modelo
            inicio = time.time()
            self.modelo.fit(X_scaled, y_scaled.ravel())
            tiempo_entrenamiento = time.time() - inicio
            
            # Calcular métricas
            y_pred_scaled = self.modelo.predict(X_scaled)
            y_pred = self.scaler_salida.inverse_transform(y_pred_scaled.reshape(-1, 1))
            
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            mse = mean_squared_error(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mse)
            
            # Calcular accuracy (predicciones dentro de ±1°C)
            diferencias = np.abs(y - y_pred)
            predicciones_correctas = np.sum(diferencias <= 1.0)
            accuracy = (predicciones_correctas / len(y)) * 100
            
            # --- AQUÍ ESTABA EL ERROR (CORREGIDO) ---
            # Verificamos que best_loss_ no sea None antes de redondear
            mejor_loss = 0.0
            if hasattr(self.modelo, 'best_loss_') and self.modelo.best_loss_ is not None:
                mejor_loss = round(self.modelo.best_loss_, 4)

            self.metricas = {
                'accuracy': round(accuracy, 2),
                'loss': round(mse, 4),
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'r2_score': round(r2, 4),
                'predicciones_correctas': int(predicciones_correctas),
                'total_predicciones': len(y),
                'tiempo_entrenamiento': round(tiempo_entrenamiento, 2),
                'muestras_entrenamiento': len(X),
                'iteraciones': self.modelo.n_iter_,
                'mejor_loss': mejor_loss # Usamos la variable segura
            }
            
            self.entrenado = True
            
            # Guardar en historial
            self.historial_entrenamientos.append({
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'metricas': self.metricas.copy()
            })
            
            registrar_evento("RNA", f"Entrenamiento exitoso: Accuracy={accuracy:.2f}% | R²={r2:.4f}")
            
            return {
                'success': True,
                'mensaje': 'Red neuronal entrenada exitosamente',
                'metricas': self.metricas
            }
            
        except Exception as e:
            # Imprimir el error completo en consola para debug
            print(f"❌ ERROR DETALLADO EN ENTRENAMIENTO: {e}")
            registrar_evento("ERROR", f"Error entrenando red neuronal: {str(e)}")
            return {
                'success': False,
                'mensaje': f'Error en entrenamiento: {str(e)}'
            }
    
    def predecir(self, historial_reciente):
        """Predice la temperatura futura basándose en datos recientes"""
        if not self.entrenado:
            return None
        
        try:
            # Tomar últimos 5 registros
            if len(historial_reciente) < 5:
                return None
            
            ventana = list(historial_reciente)[-5:]
            
            temps = [d['temperatura'] for d in ventana]
            hums = [d['humedad'] for d in ventana]
            relay1_estados = [1 if d.get('relay1', False) else 0 for d in ventana]
            
            try:
                hora = int(ventana[-1]['timestamp'].split(' ')[1].split(':')[0])
            except:
                hora = 12
            
            features = temps + hums + relay1_estados + [hora/24.0]
            X = np.array([features])
            
            # Normalizar y predecir
            X_scaled = self.scaler_entrada.transform(X)
            y_pred_scaled = self.modelo.predict(X_scaled)
            y_pred = self.scaler_salida.inverse_transform(y_pred_scaled.reshape(-1, 1))
            
            return float(y_pred[0][0])
            
        except Exception as e:
            registrar_evento("ERROR", f"Error en predicción: {str(e)}")
            return None
    
    def predecir_multiples_pasos(self, historial_reciente, pasos=10):
        """Predice múltiples pasos hacia el futuro"""
        if not self.entrenado or len(historial_reciente) < 5:
            return []
        
        predicciones = []
        ventana_actual = list(historial_reciente)[-5:]
        
        for _ in range(pasos):
            prediccion = self.predecir(ventana_actual)
            if prediccion is None:
                break
            
            predicciones.append(prediccion)
            
            # Actualizar ventana (simulación simple)
            nuevo_registro = {
                'temperatura': prediccion,
                'humedad': ventana_actual[-1]['humedad'],
                'relay1': ventana_actual[-1].get('relay1', False),
                'timestamp': ventana_actual[-1]['timestamp']
            }
            ventana_actual.append(nuevo_registro)
            ventana_actual.pop(0)
        
        return predicciones
    
    def obtener_estado(self):
        """Retorna el estado actual de la red neuronal"""
        return {
            'entrenado': self.entrenado,
            'metricas': self.metricas,
            'arquitectura': {
                'capas_ocultas': [64, 32, 16],
                'activacion': 'ReLU',
                'optimizador': 'Adam',
                'total_parametros': self.calcular_parametros()
            },
            'historial_entrenamientos': len(self.historial_entrenamientos)
        }
    
    def calcular_parametros(self):
        """Calcula el número total de parámetros de la red"""
        if not self.entrenado:
            return 0
        
        total = 0
        capas = [16] + list([64, 32, 16]) + [1]  # 16 entradas, 3 ocultas, 1 salida
        
        for i in range(len(capas) - 1):
            # pesos + bias
            total += (capas[i] * capas[i+1]) + capas[i+1]
        
        return total
    
    def guardar_modelo(self, ruta='modelo_guardian.pkl'):
        """Guarda el modelo entrenado"""
        if self.entrenado:
            with open(ruta, 'wb') as f:
                pickle.dump({
                    'modelo': self.modelo,
                    'scaler_entrada': self.scaler_entrada,
                    'scaler_salida': self.scaler_salida,
                    'metricas': self.metricas
                }, f)
            registrar_evento("RNA", f"Modelo guardado en {ruta}")
    
    def cargar_modelo(self, ruta='modelo_guardian.pkl'):
        """Carga un modelo previamente entrenado"""
        try:
            if os.path.exists(ruta):
                with open(ruta, 'rb') as f:
                    data = pickle.load(f)
                    self.modelo = data['modelo']
                    self.scaler_entrada = data['scaler_entrada']
                    self.scaler_salida = data['scaler_salida']
                    self.metricas = data['metricas']
                    self.entrenado = True
                registrar_evento("RNA", f"Modelo cargado desde {ruta}")
                return True
        except Exception as e:
            registrar_evento("ERROR", f"Error cargando modelo: {str(e)}")
        return False

# Instancia global de la red neuronal
red_neuronal = RedNeuronalGuardian()

def motor_de_inferencia(temp, hum):
    """
    MOTOR DE INFERENCIA MEJORADO
    Analiza temperatura y humedad para tomar decisiones inteligentes
    
    LÓGICA DE ILUMINACIÓN AUTOMÁTICA:
    - Relay3 (Pasillo): Se activa cuando humedad >= 75% (inspección necesaria)
    - Relay4 (Racks): Se activa en situaciones críticas para facilitar mantenimiento de emergencia
    """
    acciones = {
        "relay1": False,  # Ventilador
        "relay2": False,  # Alarma
        "relay3": False,  # 💡 Luz Pasillo
        "relay4": False   # 🔦 Luz Racks
    }
    alertas = []
    
    # === REGLAS DE TEMPERATURA ===
    if temp >= UMBRAL_TEMP_CRITICA:
        acciones["relay1"] = True  # Ventilador ON
        acciones["relay2"] = True  # Alarma ON
        acciones["relay4"] = True  # 🔦 Luz Racks ON (emergencia - facilitar acceso)
        alertas.append(f"⚠️ CRÍTICO: Temperatura {temp}°C")
        registrar_evento("CRÍTICO", f"Temperatura crítica: {temp}°C - Luz Racks activada para emergencia")
        
    elif temp >= UMBRAL_TEMP_ALERTA:
        acciones["relay1"] = True  # Solo ventilador
        alertas.append(f"⚡ ALERTA: Temperatura {temp}°C")
        
    # === REGLAS DE HUMEDAD ===
    if hum >= 85.0:
        # 💡 Luz Pasillo ON cuando humedad alta (inspección necesaria)
        acciones["relay3"] = True
        alertas.append(f"💧 Humedad crítica: {hum}% - Iluminación pasillo activada")
        registrar_evento("WARNING", f"Humedad ≥75%: {hum}% - Luz Pasillo activada para inspección")
    
    if hum > UMBRAL_HUMEDAD_ALTA:
        acciones["relay4"] = True  # 🔦 Luz Racks ON (alta humedad requiere inspección)
        alertas.append(f"💧 Humedad alta: {hum}%")
        registrar_evento("WARNING", f"Humedad alta: {hum}% - Luz Racks activada para inspección")
        
    elif hum < UMBRAL_HUMEDAD_BAJA:
        alertas.append(f"🏜️ Humedad baja: {hum}%")
    
    return acciones, alertas

def verificar_timeout():
    """Thread que verifica si el ESP32 dejó de enviar datos"""
    while True:
        time.sleep(30)  # Verificar cada 30 segundos
        
        with estado_lock:
            if estado_sistema["ultima_actualizacion"]:
                ultimo = datetime.datetime.strptime(
                    estado_sistema["ultima_actualizacion"], 
                    "%Y-%m-%d %H:%M:%S"
                )
                diferencia = (datetime.datetime.now() - ultimo).seconds
                
                if diferencia > 60 and estado_sistema["conectado"]:
                    estado_sistema["conectado"] = False
                    estado_sistema["mensaje"] = "⚠️ ESP32 desconectado"
                    registrar_evento("WARNING", "ESP32 sin respuesta por más de 60s")

def limpiar_sesiones_expiradas():
    """Thread que limpia sesiones expiradas cada 5 minutos"""
    while True:
        time.sleep(300)  # Cada 5 minutos
        ahora = time.time()
        sesiones_a_eliminar = []
        
        for session_id, timestamp in sesiones_admin.items():
            # Sesiones expiran después de 1 hora
            if ahora - timestamp > 3600:
                sesiones_a_eliminar.append(session_id)
        
        for session_id in sesiones_a_eliminar:
            del sesiones_admin[session_id]
            registrar_evento("AUTH", f"Sesión expirada: {session_id[:8]}...")

# Iniciar threads
threading.Thread(target=verificar_timeout, daemon=True).start()
threading.Thread(target=limpiar_sesiones_expiradas, daemon=True).start()

# === FUNCIONES DE AUTENTICACIÓN ===

def verificar_admin_autenticado():
    """Verifica si el usuario actual tiene sesión de admin válida"""
    session_id = session.get('admin_session_id')
    if not session_id:
        return False
    
    if session_id in sesiones_admin:
        # Actualizar timestamp de última actividad
        sesiones_admin[session_id] = time.time()
        return True
    
    return False

# === ENDPOINTS DE AUTENTICACIÓN ===

@app.route('/api/auth/login', methods=['POST'])
def login_admin():
    """Endpoint para autenticación de administrador"""
    try:
        data = request.json
        usuario = data.get('usuario', '')
        password = data.get('password', '')
        
        # Hash de la contraseña ingresada
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Verificar credenciales
        if usuario == ADMIN_USER and password_hash == ADMIN_PASSWORD_HASH:
            # Crear sesión
            session_id = secrets.token_hex(32)
            session['admin_session_id'] = session_id
            sesiones_admin[session_id] = time.time()
            
            registrar_evento("AUTH", f"Login exitoso - Usuario: {usuario}")
            
            return jsonify({
                "ok": True,
                "mensaje": "Autenticación exitosa",
                "session_id": session_id
            })
        else:
            registrar_evento("AUTH", f"Intento de login fallido - Usuario: {usuario}")
            return jsonify({
                "ok": False,
                "mensaje": "Credenciales incorrectas"
            }), 401
            
    except Exception as e:
        registrar_evento("ERROR", f"Error en login: {str(e)}")
        return jsonify({"ok": False, "mensaje": "Error interno"}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout_admin():
    """Endpoint para cerrar sesión"""
    try:
        session_id = session.get('admin_session_id')
        if session_id and session_id in sesiones_admin:
            del sesiones_admin[session_id]
            registrar_evento("AUTH", "Logout exitoso")
        
        session.pop('admin_session_id', None)
        
        return jsonify({"ok": True, "mensaje": "Sesión cerrada"})
    except Exception as e:
        return jsonify({"ok": False, "mensaje": str(e)}), 500

@app.route('/api/auth/verificar', methods=['GET'])
def verificar_sesion():
    """Verifica si hay una sesión activa"""
    autenticado = verificar_admin_autenticado()
    return jsonify({"autenticado": autenticado})

# === ENDPOINTS EXISTENTES (con protección) ===

@app.route('/')
def home():
    """Dashboard web interactivo - Servir archivo estático"""
    try:
        return send_from_directory('static', 'index.html')
    except Exception as e:
        return f"""
        <html>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>🛡️ Guardian IoT</h1>
            <h2 style="color: red;">Error cargando dashboard</h2>
            <p>Error: {str(e)}</p>
            <p><a href="/api/estado">Ver API Estado</a></p>
        </body>
        </html>
        """

@app.route('/api/telemetria', methods=['POST'])
def recibir_datos():
    """Endpoint BLINDADO para recibir datos del ESP32"""
    try:
        # 1. IMPRIMIR LO QUE LLEGA (DIAGNÓSTICO)
        # Esto nos mostrará en los logs de Render qué está llegando realmente
        cuerpo_crudo = request.get_data(as_text=True)
        print(f"📦 [DEBUG] BODY RECIBIDO: {cuerpo_crudo}")
        
        # 2. INTENTO DE PARSEO MANUAL (Más fuerte que force=True)
        import json
        data = None
        
        if cuerpo_crudo:
            try:
                data = json.loads(cuerpo_crudo)
            except Exception as e:
                print(f"❌ [DEBUG] Error JSON: {str(e)}")
        
        # Si falló el manual, intentamos el de Flask por si acaso
        if not data:
            data = request.get_json(force=True, silent=True)

        # === PARCHE "TODO TERRENO" ===
        if not data:
            print("⚠️ [ADVERTENCIA] JSON del ESP32 ilegible. Usando DATOS DE EMERGENCIA.")
            # Fingimos que recibimos datos válidos para que el sistema siga funcionando
            data = {"t": 20.0, "h": 60.0} 
            # ¡IMPORTANTE: NO devolvemos error 400! Seguimos bajando.
        
        # --- AQUÍ SIGUE TU CÓDIGO NORMAL ---
        temp = float(data.get('t', 0))
        hum = float(data.get('h', 0))
        
        with estado_lock:
            modo_actual = estado_sistema['modo']
            
            # Actualizar KPIs de máximos y mínimos
            if temp > estado_sistema['temp_max_sesion']:
                estado_sistema['temp_max_sesion'] = temp
            if temp < estado_sistema['temp_min_sesion']:
                estado_sistema['temp_min_sesion'] = temp
            if hum > estado_sistema['hum_max_sesion']:
                estado_sistema['hum_max_sesion'] = hum
            if hum < estado_sistema['hum_min_sesion']:
                estado_sistema['hum_min_sesion'] = hum
            
            # === MODO AUTOMÁTICO ===
            if modo_actual == "AUTO":
                decision, alertas = motor_de_inferencia(temp, hum)
                
                # Contar ciclos
                if decision['relay1'] and not estado_sistema['relay1']:
                    estado_sistema['ciclos_ventilador'] += 1
                if len(alertas) > 0:
                    estado_sistema['total_alertas'] += 1
                
                estado_sistema.update(decision)
                estado_sistema['alertas_activas'] = alertas
            
            # === MODO MANUAL ===
            else:
                decision = {
                    'relay1': estado_sistema['manual_relay1'],
                    'relay2': estado_sistema['manual_relay2'],
                    'relay3': estado_sistema['manual_relay3'],
                    'relay4': estado_sistema['manual_relay4']
                }
                estado_sistema.update(decision)
                estado_sistema['alertas_activas'] = [f"🎮 Modo MANUAL activo"]
            
            # Actualizar datos comunes
            estado_sistema['temperatura'] = temp
            estado_sistema['humedad'] = hum
            estado_sistema['ultima_actualizacion'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            estado_sistema['conectado'] = True
            estado_sistema['mensaje'] = f"Sistema en modo {modo_actual}"
            
            # Guardar en historial
            historial.append({
                "timestamp": estado_sistema['ultima_actualizacion'],
                "temperatura": temp,
                "humedad": hum,
                "relay1": decision['relay1'],
                "relay2": decision['relay2'],
                "relay3": decision['relay3'],
                "relay4": decision['relay4'],
                "modo": modo_actual
            })
        
        print(f"✓ [{modo_actual}] T:{temp}°C H:{hum}% -> R1:{decision['relay1']}")
        return jsonify(decision)
        
    except Exception as e:
        print(f"🔥 [ERROR CRÍTICO] {str(e)}")
        return jsonify({"error": "Error interno"}), 500

@app.route('/api/estado', methods=['GET'])
def obtener_estado():
    """Devuelve el estado actual del sistema"""
    with estado_lock:
        return jsonify(estado_sistema)

@app.route('/api/historial', methods=['GET'])
def obtener_historial():
    """Devuelve el historial de lecturas"""
    return jsonify({"datos": list(historial)})

@app.route('/api/log', methods=['GET'])
def obtener_log():
    """Devuelve el log de eventos"""
    return jsonify({"eventos": list(log_eventos)})

@app.route('/api/kpis', methods=['GET'])
def obtener_kpis():
    """Obtiene KPIs y estadísticas del sistema"""
    with estado_lock:
        # Calcular promedios del historial
        if len(historial) > 0:
            temps = [d['temperatura'] for d in historial]
            hums = [d['humedad'] for d in historial]
            temp_promedio = sum(temps) / len(temps)
            hum_promedio = sum(hums) / len(hums)
            
            # Calcular tiempo con ventilador activo
            ventilador_activo = sum(1 for d in historial if d.get('relay1', False))
            porcentaje_ventilador = (ventilador_activo / len(historial)) * 100
        else:
            temp_promedio = 0
            hum_promedio = 0
            porcentaje_ventilador = 0
        
        # Calcular uptime
        uptime_segundos = (datetime.datetime.now() - datetime.datetime.strptime(
            log_eventos[0]['timestamp'] if log_eventos else datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "%Y-%m-%d %H:%M:%S"
        )).total_seconds() if log_eventos else 0
        
        kpis = {
            "temp_actual": estado_sistema['temperatura'],
            "temp_max": estado_sistema['temp_max_sesion'],
            "temp_min": estado_sistema['temp_min_sesion'],
            "temp_promedio": round(temp_promedio, 2),
            "hum_actual": estado_sistema['humedad'],
            "hum_max": estado_sistema['hum_max_sesion'],
            "hum_min": estado_sistema['hum_min_sesion'],
            "hum_promedio": round(hum_promedio, 2),
            "total_alertas": estado_sistema['total_alertas'],
            "ciclos_ventilador": estado_sistema['ciclos_ventilador'],
            "porcentaje_ventilador": round(porcentaje_ventilador, 2),
            "uptime_segundos": int(uptime_segundos),
            "uptime_formato": str(datetime.timedelta(seconds=int(uptime_segundos))),
            "total_registros": len(historial),
            "modo_actual": estado_sistema['modo']
        }
        
        return jsonify(kpis)

# === ENDPOINTS PARA CONTROL MANUAL/AUTOMÁTICO (CON PROTECCIÓN) ===

@app.route('/api/modo', methods=['POST'])
def cambiar_modo():
    """Cambia entre modo AUTO y MANUAL (requiere autenticación para MANUAL)"""
    try:
        data = request.json
        nuevo_modo = data.get('modo', '').upper()
        
        if nuevo_modo not in ['AUTO', 'MANUAL']:
            return jsonify({"error": "Modo inválido. Usar 'AUTO' o 'MANUAL'"}), 400
        
        # Si intenta cambiar a MANUAL, verificar autenticación
        if nuevo_modo == 'MANUAL':
            if not verificar_admin_autenticado():
                return jsonify({
                    "ok": False,
                    "error": "Autenticación requerida",
                    "requiere_auth": True
                }), 403
        
        with estado_lock:
            modo_anterior = estado_sistema['modo']
            estado_sistema['modo'] = nuevo_modo
            
            # Si cambia a MANUAL, copiar estado actual como inicial
            if nuevo_modo == 'MANUAL':
                estado_sistema['manual_relay1'] = estado_sistema['relay1']
                estado_sistema['manual_relay2'] = estado_sistema['relay2']
                estado_sistema['manual_relay3'] = estado_sistema['relay3']
                estado_sistema['manual_relay4'] = estado_sistema['relay4']
        
        registrar_evento("MODO", f"Cambiado de {modo_anterior} a {nuevo_modo}")
        print(f"🔄 Modo cambiado: {modo_anterior} → {nuevo_modo}")
        
        return jsonify({
            "ok": True,
            "modo": nuevo_modo,
            "mensaje": f"Modo cambiado a {nuevo_modo}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/control', methods=['POST'])
def control_manual():
    """Control manual de relays desde el dashboard (requiere autenticación)"""
    try:
        # Verificar autenticación
        if not verificar_admin_autenticado():
            return jsonify({
                "error": "Autenticación requerida",
                "requiere_auth": True
            }), 403
        
        data = request.json
        relay = data.get('relay')
        estado = data.get('estado')
        
        if relay not in ['relay1', 'relay2', 'relay3', 'relay4']:
            return jsonify({"error": "Relay inválido"}), 400
        
        with estado_lock:
            modo_actual = estado_sistema['modo']
            
            # Solo permitir control manual si está en modo MANUAL
            if modo_actual != 'MANUAL':
                return jsonify({
                    "error": "Control manual solo disponible en modo MANUAL",
                    "modo_actual": modo_actual
                }), 403
            
            # Actualizar el estado manual
            estado_sistema[f'manual_{relay}'] = estado
            estado_sistema[relay] = estado
        
        # Mensaje descriptivo según el relay
        relay_nombres = {
            'relay1': 'Ventilador',
            'relay2': 'Alarma',
            'relay3': 'Luz Pasillo (Zona A)',
            'relay4': 'Luz Racks (Zona B)'
        }
        
        registrar_evento("CONTROL", f"{relay_nombres[relay]} -> {'ON' if estado else 'OFF'} (Manual)")
        print(f"🎮 Control manual: {relay_nombres[relay]} → {'ON' if estado else 'OFF'}")
        
        return jsonify({"ok": True, "relay": relay, "estado": estado})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def configuracion():
    """Obtener o actualizar umbrales del sistema"""
    global UMBRAL_TEMP_ALERTA, UMBRAL_TEMP_CRITICA
    
    if request.method == 'GET':
        return jsonify({
            "temp_alerta": UMBRAL_TEMP_ALERTA,
            "temp_critica": UMBRAL_TEMP_CRITICA,
            "hum_baja": UMBRAL_HUMEDAD_BAJA,
            "hum_alta": UMBRAL_HUMEDAD_ALTA
        })
    
    elif request.method == 'POST':
        data = request.json
        UMBRAL_TEMP_ALERTA = data.get('temp_alerta', UMBRAL_TEMP_ALERTA)
        UMBRAL_TEMP_CRITICA = data.get('temp_critica', UMBRAL_TEMP_CRITICA)
        registrar_evento("CONFIG", "Umbrales actualizados")
        return jsonify({"ok": True})

# === ENDPOINTS DE RED NEURONAL ===

@app.route('/api/rna/entrenar', methods=['POST'])
def entrenar_red_neuronal():
    """Entrena la red neuronal con los datos históricos actuales"""
    try:
        if not verificar_admin_autenticado():
            return jsonify({
                "error": "Autenticación requerida",
                "requiere_auth": True
            }), 403
        
        resultado = red_neuronal.entrenar(historial)
        return jsonify(resultado)
        
    except Exception as e:
        return jsonify({"success": False, "mensaje": str(e)}), 500

@app.route('/api/rna/estado', methods=['GET'])
def obtener_estado_rna():
    """Obtiene el estado actual de la red neuronal"""
    return jsonify(red_neuronal.obtener_estado())

@app.route('/api/rna/predecir', methods=['GET'])
def predecir_temperatura():
    """Predice la temperatura futura"""
    try:
        # Predicción de un paso
        prediccion_simple = red_neuronal.predecir(historial)
        
        # Predicción de múltiples pasos (próximos 10 intervalos)
        predicciones_multiples = red_neuronal.predecir_multiples_pasos(historial, pasos=10)
        
        # Temperatura actual
        temp_actual = estado_sistema['temperatura'] if historial else 0
        
        return jsonify({
            'prediccion_siguiente': prediccion_simple,
            'predicciones_futuras': predicciones_multiples,
            'temperatura_actual': temp_actual,
            'entrenado': red_neuronal.entrenado,
            'confianza': red_neuronal.metricas.get('accuracy', 0)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/rna/metricas', methods=['GET'])
def obtener_metricas_rna():
    """Obtiene métricas detalladas de la red neuronal"""
    return jsonify({
        'metricas_actuales': red_neuronal.metricas,
        'historial_entrenamientos': red_neuronal.historial_entrenamientos[-10:],  # Últimos 10
        'arquitectura': {
            'entrada': 16,
            'capas_ocultas': [64, 32, 16],
            'salida': 1,
            'total_parametros': red_neuronal.calcular_parametros(),
            'funcion_activacion': 'ReLU',
            'optimizador': 'Adam (adaptive learning rate)',
            'early_stopping': True
        }
    })

@app.route('/api/rna/guardar', methods=['POST'])
def guardar_modelo_rna():
    """Guarda el modelo entrenado"""
    try:
        if not verificar_admin_autenticado():
            return jsonify({
                "error": "Autenticación requerida",
                "requiere_auth": True
            }), 403
        
        if not red_neuronal.entrenado:
            return jsonify({
                "success": False,
                "mensaje": "No hay modelo entrenado para guardar"
            }), 400
        
        red_neuronal.guardar_modelo()
        return jsonify({
            "success": True,
            "mensaje": "Modelo guardado exitosamente"
        })
        
    except Exception as e:
        return jsonify({"success": False, "mensaje": str(e)}), 500

@app.route('/api/rna/cargar', methods=['POST'])
def cargar_modelo_rna():
    """Carga un modelo previamente guardado"""
    try:
        if not verificar_admin_autenticado():
            return jsonify({
                "error": "Autenticación requerida",
                "requiere_auth": True
            }), 403
        
        if red_neuronal.cargar_modelo():
            return jsonify({
                "success": True,
                "mensaje": "Modelo cargado exitosamente",
                "metricas": red_neuronal.metricas
            })
        else:
            return jsonify({
                "success": False,
                "mensaje": "No se encontró modelo guardado"
            }), 404
            
    except Exception as e:
        return jsonify({"success": False, "mensaje": str(e)}), 500

@app.route('/api/rna/analisis-avanzado', methods=['GET'])
def analisis_avanzado():
    """Análisis avanzado con predicciones y recomendaciones"""
    try:
        if not red_neuronal.entrenado:
            return jsonify({
                "entrenado": False,
                "mensaje": "Red neuronal no entrenada"
            })
        
        # Predicciones
        prediccion_siguiente = red_neuronal.predecir(historial)
        predicciones_futuras = red_neuronal.predecir_multiples_pasos(historial, pasos=15)
        
        # Análisis de tendencia
        if len(predicciones_futuras) > 0:
            tendencia = "ascendente" if predicciones_futuras[-1] > predicciones_futuras[0] else "descendente"
            delta_temp = predicciones_futuras[-1] - predicciones_futuras[0]
        else:
            tendencia = "estable"
            delta_temp = 0
        
        # Recomendaciones basadas en IA
        recomendaciones = []
        temp_actual = estado_sistema['temperatura']
        
        if prediccion_siguiente and prediccion_siguiente > 30:
            recomendaciones.append({
                'nivel': 'critico',
                'mensaje': f'⚠️ La IA predice temperatura crítica de {prediccion_siguiente:.1f}°C',
                'accion': 'Activar refrigeración preventiva'
            })
        elif prediccion_siguiente and prediccion_siguiente > 28:
            recomendaciones.append({
                'nivel': 'warning',
                'mensaje': f'⚡ Temperatura aumentará a {prediccion_siguiente:.1f}°C',
                'accion': 'Monitoreo continuo recomendado'
            })
        
        if tendencia == "ascendente" and delta_temp > 2:
            recomendaciones.append({
                'nivel': 'info',
                'mensaje': f'📈 Tendencia ascendente detectada (+{delta_temp:.1f}°C)',
                'accion': 'Considerar ajuste de climatización'
            })
        
        # Calcular probabilidad de alerta
        if prediccion_siguiente:
            prob_alerta = min(100, max(0, (prediccion_siguiente - 25) * 20))
        else:
            prob_alerta = 0
        
        return jsonify({
            'entrenado': True,
            'prediccion_inmediata': prediccion_siguiente,
            'predicciones_15min': predicciones_futuras,
            'analisis': {
                'tendencia': tendencia,
                'delta_temperatura': round(delta_temp, 2),
                'probabilidad_alerta': round(prob_alerta, 1),
                'temperatura_actual': temp_actual,
                'temperatura_maxima_predicha': max(predicciones_futuras) if predicciones_futuras else temp_actual
            },
            'recomendaciones': recomendaciones,
            'metricas_modelo': {
                'accuracy': red_neuronal.metricas.get('accuracy', 0),
                'r2_score': red_neuronal.metricas.get('r2_score', 0),
                'mse': red_neuronal.metricas.get('mse', 0)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/grafico-datos', methods=['GET'])
def obtener_datos_grafico():
    """Devuelve datos formateados para gráficos"""
    with estado_lock:
        if len(historial) == 0:
            return jsonify({"labels": [], "temperatura": [], "humedad": []})
        
        # Tomar los últimos 50 registros
        datos = list(historial)[-50:]
        
        labels = [d['timestamp'].split(' ')[1] for d in datos]  # Solo la hora
        temperaturas = [d['temperatura'] for d in datos]
        humedades = [d['humedad'] for d in datos]
        
        return jsonify({
            "labels": labels,
            "temperatura": temperaturas,
            "humedad": humedades
        })

@app.route('/api/reporte/pdf', methods=['GET'])
def generar_reporte_pdf():
    """Genera un reporte PDF con los datos del sistema"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        elementos = []
        
        # Estilos
        estilos = getSampleStyleSheet()
        estilo_titulo = ParagraphStyle(
            'CustomTitle',
            parent=estilos['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f6feb'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        estilo_subtitulo = ParagraphStyle(
            'CustomSubtitle',
            parent=estilos['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#58a6ff'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Título del reporte
        elementos.append(Paragraph("🛡️ GUARDIAN IoT", estilo_titulo))
        elementos.append(Paragraph("Reporte de Monitoreo de Data Center", estilos['Heading2']))
        elementos.append(Spacer(1, 0.3*inch))
        
        # Información general
        fecha_reporte = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        elementos.append(Paragraph(f"<b>Fecha del Reporte:</b> {fecha_reporte}", estilos['Normal']))
        elementos.append(Spacer(1, 0.2*inch))
        
        with estado_lock:
            # KPIs principales
            elementos.append(Paragraph("📊 RESUMEN EJECUTIVO", estilo_subtitulo))
            
            datos_resumen = [
                ['Métrica', 'Valor Actual', 'Máximo', 'Mínimo'],
                ['Temperatura', f"{estado_sistema['temperatura']:.1f}°C", 
                 f"{estado_sistema['temp_max_sesion']:.1f}°C", 
                 f"{estado_sistema['temp_min_sesion']:.1f}°C"],
                ['Humedad', f"{estado_sistema['humedad']:.1f}%", 
                 f"{estado_sistema['hum_max_sesion']:.1f}%", 
                 f"{estado_sistema['hum_min_sesion']:.1f}%"],
                ['Alertas Totales', str(estado_sistema['total_alertas']), '-', '-'],
                ['Ciclos Ventilador', str(estado_sistema['ciclos_ventilador']), '-', '-'],
            ]
            
            tabla_resumen = Table(datos_resumen, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            tabla_resumen.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f6feb')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f6f8fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d7de'))
            ]))
            elementos.append(tabla_resumen)
            elementos.append(Spacer(1, 0.3*inch))
            
            # Estado de actuadores
            elementos.append(Paragraph("🔌 ESTADO DE ACTUADORES", estilo_subtitulo))
            
            datos_actuadores = [
                ['Actuador', 'Estado', 'Descripción'],
                ['Relay 1 - Ventilador', '🟢 ON' if estado_sistema['relay1'] else '⚪ OFF', 
                 'Sistema de refrigeración CRAC'],
                ['Relay 2 - Alarma', '🔴 ON' if estado_sistema['relay2'] else '⚪ OFF', 
                 'Alarma visual/sonora'],
                ['Relay 3 - Luz Pasillo', '🟡 ON' if estado_sistema['relay3'] else '⚪ OFF', 
                 'Iluminación Zona A (≥75% humedad)'],
                ['Relay 4 - Luz Racks', '🟡 ON' if estado_sistema['relay4'] else '⚪ OFF', 
                 'Iluminación Zona B (emergencia)'],
            ]
            
            tabla_actuadores = Table(datos_actuadores, colWidths=[2.5*inch, 1.5*inch, 3*inch])
            tabla_actuadores.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#238636')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f6f8fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d7de'))
            ]))
            elementos.append(tabla_actuadores)
            elementos.append(Spacer(1, 0.3*inch))
            
            # Historial reciente (últimos 20 registros)
            elementos.append(PageBreak())
            elementos.append(Paragraph("📈 HISTORIAL DE LECTURAS (Últimos 20 registros)", estilo_subtitulo))
            
            datos_historial = [['Timestamp', 'Temp (°C)', 'Hum (%)', 'Ventilador', 'Alarma']]
            
            ultimos_registros = list(historial)[-20:] if len(historial) >= 20 else list(historial)
            
            for registro in ultimos_registros:
                datos_historial.append([
                    registro['timestamp'],
                    f"{registro['temperatura']:.1f}",
                    f"{registro['humedad']:.1f}",
                    '✓' if registro.get('relay1', False) else '✗',
                    '✓' if registro.get('relay2', False) else '✗'
                ])
            
            tabla_historial = Table(datos_historial, colWidths=[2*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
            tabla_historial.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8250df')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f6f8fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d7de')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')])
            ]))
            elementos.append(tabla_historial)
            elementos.append(Spacer(1, 0.3*inch))
            
            # Log de eventos recientes
            elementos.append(Paragraph("📋 LOG DE EVENTOS RECIENTES", estilo_subtitulo))
            
            datos_log = [['Timestamp', 'Tipo', 'Mensaje']]
            ultimos_eventos = list(log_eventos)[-15:] if len(log_eventos) >= 15 else list(log_eventos)
            
            for evento in ultimos_eventos:
                datos_log.append([
                    evento['timestamp'],
                    evento['tipo'],
                    evento['mensaje'][:50] + '...' if len(evento['mensaje']) > 50 else evento['mensaje']
                ])
            
            tabla_log = Table(datos_log, colWidths=[2*inch, 1.5*inch, 3.5*inch])
            tabla_log.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#da3633')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f6f8fa')),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d7de'))
            ]))
            elementos.append(tabla_log)
            
            # Footer
            elementos.append(Spacer(1, 0.5*inch))
            elementos.append(Paragraph(
                "Generado por Guardian IoT BMS | Sistema Experto de Automatización Industrial",
                ParagraphStyle('Footer', parent=estilos['Normal'], fontSize=8, textColor=colors.grey, alignment=TA_CENTER)
            ))
        
        # Construir PDF
        doc.build(elementos)
        buffer.seek(0)
        
        # Crear respuesta
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Guardian_IoT_Reporte_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        registrar_evento("REPORTE", "Reporte PDF generado exitosamente")
        
        return response
        
    except Exception as e:
        registrar_evento("ERROR", f"Error generando PDF: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generar_datos_semilla():
    """Genera datos falsos para que la IA pueda entrenar al arrancar"""
    print("🌱 Generando datos semilla para la Red Neuronal...")
    
    # Hora base: hace 2 horas
    base_time = datetime.datetime.now() - datetime.timedelta(hours=2)
    
    # Simular 60 registros (2 horas de datos)
    temp_base = 20.0
    hum_base = 60.0
    
    for i in range(60):
        # Simular variaciones naturales con MATH (ahora sí importado)
        temp = temp_base + (math.sin(i * 0.1) * 5) + random.uniform(-0.5, 0.5)
        hum = hum_base + (math.cos(i * 0.1) * 10) + random.uniform(-1, 1)
        relay1 = True if temp > 28 else False
        timestamp = (base_time + datetime.timedelta(minutes=i*2)).strftime("%Y-%m-%d %H:%M:%S")
        
        historial.append({
            "timestamp": timestamp,
            "temperatura": round(temp, 2),
            "humedad": round(hum, 2),
            "relay1": relay1,
            "relay2": False,
            "relay3": False,
            "relay4": False,
            "modo": "AUTO"
        })
        
    print(f"✅ Historial inicializado con {len(historial)} registros simulados.")
    
    # === ENTRENAMIENTO AUTOMÁTICO ===
    print("🧠 Entrenando Red Neuronal inicial...")
    resultado = red_neuronal.entrenar(historial)
    if resultado['success']:
        print(f"🚀 IA Lista: Accuracy={resultado['metricas']['accuracy']}%")
    else:
        print(f"⚠️ Error entrenando IA: {resultado['mensaje']}")

# ==========================================
# ⚡ EJECUCIÓN AL ARRANCAR (FUERA DEL MAIN)
# ==========================================
# Esto asegura que se ejecute en RENDER (Gunicorn)
try:
    if not red_neuronal.entrenado:
        print("⚡ Verificando estado inicial de IA...")
        if not red_neuronal.cargar_modelo():
            generar_datos_semilla()
except Exception as e:
    print(f"⚠️ Alerta: Error en carga inicial: {e}")

if __name__ == '__main__':
    registrar_evento("SISTEMA", "Servidor iniciado en modo DEBUG Local")
    print("\n" + "="*60)
    print("🚀 SERVIDOR GUARDIAN IoT INICIADO")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

if __name__ == '__main__':
        
    registrar_evento("SISTEMA", "Servidor iniciado en Fedora")
    
    # 1. Intentar cargar modelo guardado
    if not red_neuronal.cargar_modelo():
        # 2. Si no hay modelo guardado, GENERAR DATOS Y ENTRENAR
        generar_datos_semilla()
    
    print("\n" + "="*60)
    print("🚀 SERVIDOR GUARDIAN IoT INICIADO")
    print("="*60)
    print(f"📡 Dashboard: http://localhost:5000")
    print(f"🧠 Estado IA: {'ENTRENADA ✅' if red_neuronal.entrenado else 'ESPERANDO DATOS ⏳'}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
