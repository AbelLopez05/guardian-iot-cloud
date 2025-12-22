from flask import Flask, request, jsonify, send_from_directory, make_response, session
from flask_cors import CORS
import datetime
import json
import os
import math
from collections import deque
import threading
import time
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.enums import TA_CENTER
import hashlib
import secrets
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, static_folder='static')
CORS(app, supports_credentials=True, origins=["*"])
app.secret_key = secrets.token_hex(32)
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # True solo para HTTPS

# === AUTENTICACIÓN ===
ADMIN_USER = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()

# === ARCHIVO DE CONFIGURACIÓN ===
CONFIG_FILE = "config_mlp.json"

config_umbrales = {
    "usar_mlp": True,
    "modo_debug": True,
    "temp_alerta": 25.0,
    "temp_critica": 31.0,
    "humedad_baja": 30.0,
    "humedad_alta": 85.0
}

def cargar_configuracion():
    global config_umbrales
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config_cargada = json.load(f)
                config_umbrales.update(config_cargada)
            print(f"✅ Configuración cargada desde {CONFIG_FILE}")
            registrar_evento("CONFIG", "Configuración cargada exitosamente")
        else:
            guardar_configuracion()
    except Exception as e:
        print(f"⚠️ Error cargando configuración: {e}")

def guardar_configuracion():
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config_umbrales, f, indent=4)
        return True
    except Exception as e:
        print(f"❌ Error guardando configuración: {e}")
        return False

# === GEMELO DIGITAL ===
estado_sistema = {
    "temperatura": 0.0,
    "humedad": 0.0,
    "hora_actual": 0.0,
    "relay1": False,
    "relay2": False,
    "relay3": False,
    "relay4": False,
    "mensaje": "Sistema Inicializando",
    "ultima_actualizacion": None,
    "conectado": False,
    "alertas_activas": [],
    "modo": "AUTO",
    "mlp_activo": True,
    "manual_relay1": False,
    "manual_relay2": False,
    "manual_relay3": False,
    "manual_relay4": False,
    "temp_max_sesion": -100.0,
    "temp_min_sesion": 200.0,
    "hum_max_sesion": 0.0,
    "hum_min_sesion": 200.0,
    "total_alertas": 0,
    "ciclos_motor": 0,
    "tiempo_motor_on": 0,
    "uptime_sistema": 0
}

historial = deque(maxlen=500)
log_eventos = deque(maxlen=200)
estado_lock = threading.Lock()
sesiones_admin = {}

def registrar_evento(tipo, mensaje):
    evento = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo": tipo,
        "mensaje": mensaje
    }
    log_eventos.append(evento)
    print(f"[{tipo}] {mensaje}")

# ========== RED NEURONAL MLP ==========

class RedNeuronalMLP:
    """
    Red Neuronal Perceptrón Multicapa para Control de Relés
    Arquitectura: [3] → [16-8] → [4]
    
    ENTRADAS (3):
    - Temperatura (°C)
    - Humedad (%)
    - Hora (decimal 0-24)
    
    SALIDAS (4):
    - Relay 1: Motor AC
    - Relay 2: Foco 1
    - Relay 3: Foco 2
    - Relay 4: Reserva
    """
    
    def __init__(self):
        self.modelo = None
        self.scaler = MinMaxScaler()
        self.entrenado = False
        self.metricas = {
            'accuracy': 0.0,
            'samples_trained': 0,
            'architecture': 'Entrada[3] → [16-8] → Salida[4]',
            'loss': 0.0,
            'iterations': 0,
            'training_time': 0.0
        }
        self.inicializar_modelo()
    
    def inicializar_modelo(self):
        """Crear arquitectura MLP"""
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(16, 8),  # 2 capas ocultas
            activation='relu',            # Función de activación ReLU
            solver='adam',                # Optimizador Adam
            max_iter=2000,
            random_state=42,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            verbose=False
        )
        registrar_evento("MLP", "Red neuronal MLP inicializada: [3] → [16-8] → [4]")
    
    def generar_dataset_entrenamiento(self):
        """
        Genera dataset sintético basado en las 3 condiciones del examen:
        
        CONDICIÓN 1: Motor (Relay 1)
        - Temperatura: 15-18°C
        - Humedad: 80-90%
        - Hora: 17:00-17:20 (17.0-17.33)
        
        CONDICIÓN 2: Foco 1 (Relay 2)
        - Temperatura: 18-20°C
        - Humedad: 90-100%
        - Hora: 17:30-18:00 (17.5-18.0)
        
        CONDICIÓN 3: Foco 2 (Relay 3)
        - Temperatura: 20-25°C
        - Humedad: 80-90%
        - Hora: 17:30-18:00 (17.5-18.0)
        """
        X = []
        y = []
        
        # CONDICIÓN 1: Motor AC (100 muestras)
        for _ in range(100):
            temp = np.random.uniform(15, 18)
            hum = np.random.uniform(80, 90)
            hora = np.random.uniform(17.0, 17.33)
            X.append([temp, hum, hora])
            y.append('1000')  # Solo Relay 1 ON
        
        # CONDICIÓN 2: Foco 1 (100 muestras)
        for _ in range(100):
            temp = np.random.uniform(18, 20)
            hum = np.random.uniform(90, 100)
            hora = np.random.uniform(17.5, 18.0)
            X.append([temp, hum, hora])
            y.append('0100')  # Solo Relay 2 ON
        
        # CONDICIÓN 3: Foco 2 (100 muestras)
        for _ in range(100):
            temp = np.random.uniform(20, 25)
            hum = np.random.uniform(80, 90)
            hora = np.random.uniform(17.5, 18.0)
            X.append([temp, hum, hora])
            y.append('0010')  # Solo Relay 3 ON
        
        # Casos OFF - Fuera de las condiciones (150 muestras)
        for _ in range(150):
            # Generar datos que NO cumplan ninguna condición
            temp = np.random.choice([
                np.random.uniform(10, 14),   # Muy frío
                np.random.uniform(26, 35)    # Muy caliente
            ])
            hum = np.random.choice([
                np.random.uniform(20, 75),   # Humedad baja/normal
                np.random.uniform(101, 105)  # Saturado (imposible)
            ]) if temp > 25 else np.random.uniform(40, 75)
            
            # Horas fuera del rango 17:00-18:00
            hora = np.random.choice([
                np.random.uniform(0, 16),    # Antes
                np.random.uniform(19, 24)    # Después
            ])
            
            X.append([temp, hum, hora])
            y.append('0000')  # Todos OFF
        
        # Casos combinados (50 muestras) - Realismo adicional
        for _ in range(50):
            temp = np.random.uniform(10, 30)
            hum = np.random.uniform(30, 100)
            hora = np.random.uniform(0, 24)
            
            # Verificar condiciones
            cond1 = 15 <= temp <= 18 and 80 <= hum <= 90 and 17.0 <= hora <= 17.33
            cond2 = 18 <= temp <= 20 and 90 <= hum <= 100 and 17.5 <= hora <= 18.0
            cond3 = 20 <= temp <= 25 and 80 <= hum <= 90 and 17.5 <= hora <= 18.0
            
            if not (cond1 or cond2 or cond3):
                X.append([temp, hum, hora])
                y.append('0000')
        
        return np.array(X), np.array(y)
    
    def entrenar(self):
        """Entrenar la red neuronal MLP"""
        try:
            print("\n" + "="*60)
            print("🧠 ENTRENAMIENTO DE RED NEURONAL MLP")
            print("="*60)
            print("📊 Generando dataset de entrenamiento...")
            
            X, y = self.generar_dataset_entrenamiento()
            
            print(f"✓ Dataset generado: {len(X)} muestras")
            print(f"  - Condición 1 (Motor): ~100 muestras")
            print(f"  - Condición 2 (Foco 1): ~100 muestras")
            print(f"  - Condición 3 (Foco 2): ~100 muestras")
            print(f"  - Casos OFF: ~200 muestras")
            
            # Normalización de datos
            X_scaled = self.scaler.fit_transform(X)
            
            print("\n⚙️ Entrenando red neuronal...")
            print(f"  - Arquitectura: [3] → [16] → [8] → [4]")
            print(f"  - Activación: ReLU")
            print(f"  - Optimizador: Adam")
            
            inicio = time.time()
            self.modelo.fit(X_scaled, y)
            tiempo_entrenamiento = time.time() - inicio
            
            # Evaluación
            y_pred = self.modelo.predict(X_scaled)
            accuracy = np.mean(y_pred == y) * 100
            
            self.metricas = {
                'accuracy': round(accuracy, 2),
                'samples_trained': len(X),
                'training_time': round(tiempo_entrenamiento, 3),
                'iterations': self.modelo.n_iter_,
                'architecture': 'Entrada[3] → [16-8] → Salida[4]',
                'loss': round(self.modelo.loss_, 6) if hasattr(self.modelo, 'loss_') else 0.0
            }
            
            self.entrenado = True
            
            print(f"\n✅ ENTRENAMIENTO COMPLETADO")
            print(f"  - Accuracy: {accuracy:.2f}%")
            print(f"  - Tiempo: {tiempo_entrenamiento:.3f}s")
            print(f"  - Iteraciones: {self.modelo.n_iter_}")
            print(f"  - Loss final: {self.metricas['loss']}")
            print("="*60 + "\n")
            
            registrar_evento("MLP", f"Entrenamiento exitoso: Accuracy={accuracy:.2f}%")
            self.guardar_modelo()
            
            return {
                'success': True,
                'mensaje': 'Red neuronal MLP entrenada exitosamente',
                'metricas': self.metricas
            }
            
        except Exception as e:
            print(f"❌ ERROR EN ENTRENAMIENTO: {e}")
            import traceback
            traceback.print_exc()
            registrar_evento("ERROR", f"Error entrenando MLP: {str(e)}")
            return {
                'success': False,
                'mensaje': f'Error en entrenamiento: {str(e)}'
            }
    
    def predecir(self, temperatura, humedad, hora):
        """Realizar predicción (inferencia) en tiempo real"""
        if not self.entrenado:
            registrar_evento("WARNING", "MLP no entrenado, retornando estado seguro")
            return {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
        
        try:
            # Preparar entrada
            X = np.array([[temperatura, humedad, hora]])
            X_scaled = self.scaler.transform(X)
            
            # Predicción
            prediccion_str = self.modelo.predict(X_scaled)[0]
            prediccion = [int(c) for c in prediccion_str]
            
            resultado = {
                'relay1': bool(prediccion[0]),
                'relay2': bool(prediccion[1]),
                'relay3': bool(prediccion[2]),
                'relay4': bool(prediccion[3])
            }
            
            # Debug
            if config_umbrales.get('modo_debug', False):
                print(f"🤖 MLP Inferencia:")
                print(f"   Entrada: T={temperatura:.1f}°C H={humedad:.1f}% Hora={hora:.2f}")
                print(f"   Salida: {prediccion} → {resultado}")
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en predicción MLP: {e}")
            registrar_evento("ERROR", f"Error en predicción: {str(e)}")
            return {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
    
    def obtener_estado(self):
        """Retornar información completa del modelo"""
        return {
            'entrenado': self.entrenado,
            'metricas': self.metricas,
            'arquitectura': {
                'entradas': ['Temperatura (°C)', 'Humedad (%)', 'Hora (24h)'],
                'capas_ocultas': [16, 8],
                'salidas': ['Relay 1 (Motor)', 'Relay 2 (Foco 1)', 'Relay 3 (Foco 2)', 'Relay 4 (Reserva)'],
                'activacion': 'ReLU',
                'optimizador': 'Adam',
                'total_parametros': self.calcular_parametros()
            },
            'condiciones': {
                'condicion_1': 'T: 15-18°C, H: 80-90%, Hora: 17:00-17:20 → Motor',
                'condicion_2': 'T: 18-20°C, H: 90-100%, Hora: 17:30-18:00 → Foco 1',
                'condicion_3': 'T: 20-25°C, H: 80-90%, Hora: 17:30-18:00 → Foco 2'
            }
        }
    
    def calcular_parametros(self):
        """Calcular número total de parámetros de la red"""
        if not self.entrenado:
            return 0
        capas = [3] + [16, 8] + [4]
        total = 0
        for i in range(len(capas) - 1):
            # Pesos + Sesgos
            total += (capas[i] * capas[i+1]) + capas[i+1]
        return total
    
    def guardar_modelo(self, ruta='modelo_mlp.pkl'):
        """Guardar modelo entrenado"""
        if self.entrenado:
            try:
                with open(ruta, 'wb') as f:
                    pickle.dump({
                        'modelo': self.modelo,
                        'scaler': self.scaler,
                        'metricas': self.metricas,
                        'version': '4.0'
                    }, f)
                registrar_evento("MLP", f"Modelo guardado en {ruta}")
            except Exception as e:
                print(f"Error guardando modelo: {e}")
    
    def cargar_modelo(self, ruta='modelo_mlp.pkl'):
        """Cargar modelo pre-entrenado"""
        try:
            if os.path.exists(ruta):
                with open(ruta, 'rb') as f:
                    data = pickle.load(f)
                    self.modelo = data['modelo']
                    self.scaler = data['scaler']
                    self.metricas = data['metricas']
                    self.entrenado = True
                registrar_evento("MLP", f"Modelo cargado desde {ruta}")
                return True
        except Exception as e:
            print(f"Error cargando modelo: {e}")
            registrar_evento("ERROR", f"Error cargando modelo: {str(e)}")
        return False

# Instancia global del MLP
mlp = RedNeuronalMLP()

# ========== FUNCIONES AUXILIARES ==========

def obtener_hora_decimal():
    """Convertir hora actual a formato decimal (17:30 → 17.5)"""
    ahora = datetime.datetime.now()
    return ahora.hour + ahora.minute / 60.0

def verificar_timeout():
    """Thread que verifica si el ESP32 sigue conectado"""
    while True:
        time.sleep(30)
        with estado_lock:
            if estado_sistema["ultima_actualizacion"]:
                try:
                    ultimo = datetime.datetime.strptime(
                        estado_sistema["ultima_actualizacion"], 
                        "%Y-%m-%d %H:%M:%S"
                    )
                    diferencia = (datetime.datetime.now() - ultimo).seconds
                    if diferencia > 60 and estado_sistema["conectado"]:
                        estado_sistema["conectado"] = False
                        estado_sistema["mensaje"] = "⚠️ ESP32 desconectado (timeout)"
                        registrar_evento("WARNING", "ESP32 sin respuesta por más de 60s")
                except Exception as e:
                    print(f"Error en timeout check: {e}")

def limpiar_sesiones_expiradas():
    """Thread que limpia sesiones antiguas"""
    while True:
        time.sleep(300)  # Cada 5 minutos
        ahora = time.time()
        sesiones_a_eliminar = [
            sid for sid, ts in sesiones_admin.items() 
            if ahora - ts > 3600  # 1 hora
        ]
        for session_id in sesiones_a_eliminar:
            del sesiones_admin[session_id]
            print(f"Sesión expirada eliminada: {session_id[:8]}...")

# Iniciar threads de monitoreo
threading.Thread(target=verificar_timeout, daemon=True).start()
threading.Thread(target=limpiar_sesiones_expiradas, daemon=True).start()

def verificar_admin_autenticado():
    """Verificar si la sesión actual es de admin"""
    session_id = session.get('admin_session_id')
    if not session_id or session_id not in sesiones_admin:
        return False
    # Actualizar timestamp
    sesiones_admin[session_id] = time.time()
    return True

# ========== ENDPOINTS DE AUTENTICACIÓN ==========

@app.route('/api/auth/login', methods=['POST'])
def login_admin():
    """Login de administrador"""
    try:
        data = request.json
        usuario = data.get('usuario', '')
        password = data.get('password', '')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if usuario == ADMIN_USER and password_hash == ADMIN_PASSWORD_HASH:
            session_id = secrets.token_hex(32)
            session['admin_session_id'] = session_id
            sesiones_admin[session_id] = time.time()
            registrar_evento("AUTH", f"Login exitoso - Usuario: {usuario}")
            return jsonify({"ok": True, "mensaje": "Autenticación exitosa"})
        else:
            registrar_evento("AUTH", f"Intento fallido de login - Usuario: {usuario}")
            return jsonify({"ok": False, "mensaje": "Credenciales incorrectas"}), 401
    except Exception as e:
        return jsonify({"ok": False, "mensaje": str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout_admin():
    """Cerrar sesión"""
    session_id = session.get('admin_session_id')
    if session_id and session_id in sesiones_admin:
        del sesiones_admin[session_id]
        registrar_evento("AUTH", "Sesión cerrada")
    session.pop('admin_session_id', None)
    return jsonify({"ok": True})

@app.route('/api/auth/verificar', methods=['GET'])
def verificar_sesion():
    """Verificar si hay sesión activa"""
    return jsonify({"autenticado": verificar_admin_autenticado()})

# ========== ENDPOINT PRINCIPAL DE TELEMETRÍA ==========

@app.route('/api/telemetria', methods=['POST'])
def recibir_datos():
    """
    Endpoint principal que recibe datos del ESP32 y retorna decisiones del MLP
    """
    try:
        # Obtener datos del request
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        # Validar y extraer datos
        temp = float(data.get('t', 20))
        hum = float(data.get('h', 60))
        hora_decimal = obtener_hora_decimal()
        
        # Validación de rangos
        if not (-40 <= temp <= 80):
            temp = 20.0
        if not (0 <= hum <= 100):
            hum = 60.0
        
        with estado_lock:
            # Actualizar estadísticas
            if temp > estado_sistema['temp_max_sesion']:
                estado_sistema['temp_max_sesion'] = temp
            if temp < estado_sistema['temp_min_sesion']:
                estado_sistema['temp_min_sesion'] = temp
            if hum > estado_sistema['hum_max_sesion']:
                estado_sistema['hum_max_sesion'] = hum
            if hum < estado_sistema['hum_min_sesion']:
                estado_sistema['hum_min_sesion'] = hum
            
            modo_actual = estado_sistema['modo']
            
            # DECISIÓN: AUTO (MLP) o MANUAL
            if modo_actual == "AUTO":
                if config_umbrales.get('usar_mlp', True) and mlp.entrenado:
                    # ★ INFERENCIA DE LA RED NEURONAL ★
                    decision = mlp.predecir(temp, hum, hora_decimal)
                    estado_sistema['mlp_activo'] = True
                else:
                    decision = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
                    estado_sistema['mlp_activo'] = False
                
                # Contar ciclos del motor
                if decision['relay1'] and not estado_sistema['relay1']:
                    estado_sistema['ciclos_motor'] += 1
                
                # Actualizar estado
                estado_sistema.update(decision)
                estado_sistema['alertas_activas'] = []
                
            else:  # MODO MANUAL
                decision = {
                    'relay1': estado_sistema['manual_relay1'],
                    'relay2': estado_sistema['manual_relay2'],
                    'relay3': estado_sistema['manual_relay3'],
                    'relay4': estado_sistema['manual_relay4']
                }
                estado_sistema.update(decision)
                estado_sistema['alertas_activas'] = ["🎮 Modo MANUAL activo"]
            
            # Actualizar telemetría
            estado_sistema['temperatura'] = temp
            estado_sistema['humedad'] = hum
            estado_sistema['hora_actual'] = hora_decimal
            estado_sistema['ultima_actualizacion'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            estado_sistema['conectado'] = True
            estado_sistema['mensaje'] = "Sistema operando correctamente"
            
            # Agregar al historial
            historial.append({
                "timestamp": estado_sistema['ultima_actualizacion'],
                "temperatura": temp,
                "humedad": hum,
                "hora": hora_decimal,
                **decision,
                "modo": modo_actual
            })
        
        # Log de operación
        print(f"✓ [{modo_actual}] T:{temp:.1f}°C H:{hum:.1f}% H:{hora_decimal:.2f} → " +
              f"R1:{decision['relay1']} R2:{decision['relay2']} R3:{decision['relay3']} R4:{decision['relay4']}")
        
        return jsonify(decision), 200
        
    except Exception as e:
        print(f"🔥 ERROR en /api/telemetria: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========== ENDPOINTS DE DATOS ==========

@app.route('/api/estado', methods=['GET'])
def obtener_estado():
    """Estado actual del sistema"""
    with estado_lock:
        return jsonify(estado_sistema)

@app.route('/api/historial', methods=['GET'])
def obtener_historial():
    """Historial de datos"""
    return jsonify({"datos": list(historial)})

@app.route('/api/log', methods=['GET'])
def obtener_log():
    """Log de eventos del sistema"""
    return jsonify({"eventos": list(log_eventos)})

@app.route('/api/kpis', methods=['GET'])
def obtener_kpis():
    """KPIs y estadísticas"""
    with estado_lock:
        if len(historial) > 0:
            temps = [d['temperatura'] for d in historial]
            hums = [d['humedad'] for d in historial]
            temp_promedio = sum(temps) / len(temps)
            hum_promedio = sum(hums) / len(hums)
            motor_activo = sum(1 for d in historial if d.get('relay1', False))
            porcentaje_motor = (motor_activo / len(historial)) * 100
        else:
            temp_promedio = hum_promedio = porcentaje_motor = 0
        
        # Calcular uptime
        uptime_segundos = 0
        if log_eventos:
            try:
                inicio = datetime.datetime.strptime(log_eventos[0]['timestamp'], "%Y-%m-%d %H:%M:%S")
                uptime_segundos = (datetime.datetime.now() - inicio).total_seconds()
            except:
                pass
        
        return jsonify({
            "temp_actual": estado_sistema['temperatura'],
            "temp_max": estado_sistema['temp_max_sesion'],
            "temp_min": estado_sistema['temp_min_sesion'],
            "temp_promedio": round(temp_promedio, 2),
            "hum_actual": estado_sistema['humedad'],
            "hum_max": estado_sistema['hum_max_sesion'],
            "hum_min": estado_sistema['hum_min_sesion'],
            "hum_promedio": round(hum_promedio, 2),
            "total_alertas": estado_sistema['total_alertas'],
            "ciclos_motor": estado_sistema['ciclos_motor'],
            "porcentaje_motor": round(porcentaje_motor, 2),
            "uptime_segundos": int(uptime_segundos),
            "uptime_formato": str(datetime.timedelta(seconds=int(uptime_segundos))),
            "total_registros": len(historial),
            "modo_actual": estado_sistema['modo']
        })

@app.route('/api/grafico-datos', methods=['GET'])
def obtener_datos_grafico():
    """Datos para gráficos"""
    with estado_lock:
        if len(historial) == 0:
            return jsonify({"labels": [], "temperatura": [], "humedad": []})
        
        # Últimos 100 datos
        datos = list(historial)[-100:]
        labels = [d['timestamp'].split(' ')[1] for d in datos]
        temperaturas = [d['temperatura'] for d in datos]
        humedades = [d['humedad'] for d in datos]
        
        return jsonify({
            "labels": labels,
            "temperatura": temperaturas,
            "humedad": humedades
        })

# ========== ENDPOINTS DE CONTROL ==========

@app.route('/api/modo', methods=['POST'])
def cambiar_modo():
    """Cambiar entre modo AUTO y MANUAL"""
    try:
        data = request.json
        nuevo_modo = data.get('modo', '').upper()
        
        if nuevo_modo not in ['AUTO', 'MANUAL']:
            return jsonify({"error": "Modo inválido (debe ser AUTO o MANUAL)"}), 400
        
        # Verificar autenticación para modo MANUAL
        if nuevo_modo == 'MANUAL' and not verificar_admin_autenticado():
            return jsonify({"ok": False, "error": "Autenticación requerida", "requiere_auth": True}), 403
        
        with estado_lock:
            modo_anterior = estado_sistema['modo']
            estado_sistema['modo'] = nuevo_modo
            
            if nuevo_modo == 'MANUAL':
                # Preservar estado actual al cambiar a manual
                estado_sistema['manual_relay1'] = estado_sistema['relay1']
                estado_sistema['manual_relay2'] = estado_sistema['relay2']
                estado_sistema['manual_relay3'] = estado_sistema['relay3']
                estado_sistema['manual_relay4'] = estado_sistema['relay4']
            else:
                # Al volver a AUTO, el MLP tomará control
                pass
        
        registrar_evento("MODO", f"Cambiado de {modo_anterior} a {nuevo_modo}")
        return jsonify({"ok": True, "modo": nuevo_modo})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/control', methods=['POST'])
def control_manual():
    """Control manual de relés (requiere autenticación)"""
    try:
        if not verificar_admin_autenticado():
            return jsonify({"error": "Autenticación requerida", "requiere_auth": True}), 403
        
        data = request.json
        relay = data.get('relay')
        estado = data.get('estado', False)
        
        if relay not in ['relay1', 'relay2', 'relay3', 'relay4']:
            return jsonify({"error": "Relay inválido"}), 400
        
        with estado_lock:
            if estado_sistema['modo'] != 'MANUAL':
                return jsonify({"error": "Control manual solo disponible en modo MANUAL"}), 403
            
            # Actualizar estado manual
            estado_sistema[f'manual_{relay}'] = estado
            estado_sistema[relay] = estado
        
        registrar_evento("CONTROL", f"{relay} → {'ON' if estado else 'OFF'} (Manual)")
        return jsonify({"ok": True, "relay": relay, "estado": estado})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== ENDPOINTS MLP ==========

@app.route('/api/mlp/entrenar', methods=['POST'])
def entrenar_mlp():
    """Entrenar la red neuronal MLP"""
    if not verificar_admin_autenticado():
        return jsonify({"error": "Autenticación requerida", "requiere_auth": True}), 403
    
    resultado = mlp.entrenar()
    return jsonify(resultado)

@app.route('/api/mlp/estado', methods=['GET'])
def obtener_estado_mlp():
    """Obtener estado y métricas del MLP"""
    return jsonify(mlp.obtener_estado())

@app.route('/api/mlp/predecir-manual', methods=['POST'])
def predecir_manual():
    """Realizar predicción manual con valores específicos"""
    try:
        data = request.json
        temp = float(data.get('temperatura', 20))
        hum = float(data.get('humedad', 60))
        hora = float(data.get('hora', 12))
        
        resultado = mlp.predecir(temp, hum, hora)
        
        return jsonify({
            'entradas': {'temperatura': temp, 'humedad': hum, 'hora': hora},
            'salidas': resultado,
            'entrenado': mlp.entrenado,
            'arquitectura': mlp.metricas.get('architecture', 'N/A')
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ========== REPORTES ==========

@app.route('/api/reporte/pdf', methods=['GET'])
def generar_reporte_pdf():
    """Generar reporte PDF del sistema"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        elementos = []
        
        estilos = getSampleStyleSheet()
        estilo_titulo = ParagraphStyle(
            'CustomTitle', 
            parent=estilos['Heading1'], 
            fontSize=24, 
            textColor=colors.HexColor('#8250df'), 
            spaceAfter=20, 
            alignment=TA_CENTER
        )
        
        # Título
        elementos.append(Paragraph("🧠 SISTEMA MLP - CONTROL INTELIGENTE", estilo_titulo))
        elementos.append(Paragraph("Reporte de Monitoreo con Red Neuronal", estilos['Heading2']))
        elementos.append(Spacer(1, 0.3*inch))
        
        # Fecha
        fecha_reporte = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        elementos.append(Paragraph(f"<b>Fecha del Reporte:</b> {fecha_reporte}", estilos['Normal']))
        elementos.append(Spacer(1, 0.2*inch))
        
        # Tabla de datos actuales
        with estado_lock:
            datos_resumen = [
                ['Métrica', 'Valor Actual', 'Máximo', 'Mínimo'],
                ['Temperatura (°C)', f"{estado_sistema['temperatura']:.1f}", 
                 f"{estado_sistema['temp_max_sesion']:.1f}", 
                 f"{estado_sistema['temp_min_sesion']:.1f}"],
                ['Humedad (%)', f"{estado_sistema['humedad']:.1f}", 
                 f"{estado_sistema['hum_max_sesion']:.1f}", 
                 f"{estado_sistema['hum_min_sesion']:.1f}"],
                ['Modo', estado_sistema['modo'], '-', '-'],
                ['Ciclos Motor', str(estado_sistema['ciclos_motor']), '-', '-']
            ]
            
            tabla_resumen = Table(datos_resumen, colWidths=[2*inch, 1.8*inch, 1.8*inch, 1.8*inch])
            tabla_resumen.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8250df')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#d0d7de')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f6f8fa')])
            ]))
            elementos.append(tabla_resumen)
        
        elementos.append(Spacer(1, 0.3*inch))
        
        # Información MLP
        elementos.append(Paragraph("<b>Red Neuronal MLP</b>", estilos['Heading3']))
        if mlp.entrenado:
            mlp_info = [
                ['Parámetro', 'Valor'],
                ['Arquitectura', mlp.metricas['architecture']],
                ['Accuracy', f"{mlp.metricas['accuracy']}%"],
                ['Muestras Entrenadas', str(mlp.metricas['samples_trained'])],
                ['Iteraciones', str(mlp.metricas['iterations'])],
                ['Tiempo Entrenamiento', f"{mlp.metricas['training_time']}s"]
            ]
            tabla_mlp = Table(mlp_info, colWidths=[3*inch, 3.5*inch])
            tabla_mlp.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#8250df')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            elementos.append(tabla_mlp)
        
        doc.build(elementos)
        buffer.seek(0)
        
        response = make_response(buffer.getvalue())
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=Reporte_MLP_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return response
        
    except Exception as e:
        print(f"Error generando PDF: {e}")
        return jsonify({"error": str(e)}), 500

# ========== FRONTEND ==========

@app.route('/')
def home():
    """Servir página principal"""
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    """Servir archivos estáticos"""
    return send_from_directory('static', path)

# ========== INICIALIZACIÓN ==========

def inicializar_sistema():
    """Inicializar todos los componentes del sistema"""
    print("\n" + "="*70)
    print("🧠 SISTEMA MLP - CONTROL INTELIGENTE CON RED NEURONAL")
    print("="*70)
    
    # Cargar configuración
    cargar_configuracion()
    
    # Intentar cargar modelo pre-entrenado
    if not mlp.cargar_modelo():
        print("\n📦 Modelo MLP no encontrado. Entrenando desde cero...")
        resultado = mlp.entrenar()
        if resultado['success']:
            print(f"✅ Modelo entrenado exitosamente")
            print(f"   - Accuracy: {resultado['metricas']['accuracy']}%")
            print(f"   - Tiempo: {resultado['metricas']['training_time']}s")
        else:
            print(f"❌ Error en entrenamiento: {resultado['mensaje']}")
    else:
        print(f"\n✅ Modelo MLP cargado correctamente")
        print(f"   - Accuracy: {mlp.metricas['accuracy']}%")
        print(f"   - Muestras: {mlp.metricas['samples_trained']}")
    
    print("\n" + "="*70)
    print("📊 INFORMACIÓN DEL SERVIDOR")
    print("="*70)
    print(f"📡 Dashboard Web: http://localhost:5000")
    print(f"🔌 API Endpoint: http://localhost:5000/api/telemetria")
    print(f"🤖 Estado MLP: {'ENTRENADO ✅' if mlp.entrenado else 'NO ENTRENADO ⚠️'}")
    print(f"⚙️ Modos: AUTO (MLP) + MANUAL")
    print(f"🔑 Credenciales Admin: usuario='admin' / password='admin123'")
    print("="*70)
    print("\n🚀 Sistema listo. Esperando conexiones...\n")
    
    registrar_evento("SISTEMA", "Servidor Flask iniciado correctamente")

if __name__ == '__main__':
    inicializar_sistema()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
