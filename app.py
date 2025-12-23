from flask import Flask, request, jsonify, send_from_directory, make_response, session
from flask_cors import CORS
import datetime
import json
import os
import time
from collections import deque
import threading
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
app.config['SESSION_COOKIE_SECURE'] = False

# === AUTENTICACIÓN ===
ADMIN_USER = "admin"
ADMIN_PASSWORD_HASH = hashlib.sha256("admin123".encode()).hexdigest()

# === ARCHIVOS DE CONFIGURACIÓN ===
CONFIG_FILE = "config_mlp.json"
HORARIOS_FILE = "horarios_config.json"

config_umbrales = {
    "usar_mlp": True,
    "modo_debug": True,
    "temp_alerta": 25.0,
    "temp_critica": 31.0,
    "humedad_baja": 30.0,
    "humedad_alta": 85.0
}

# === ✅ CONFIGURACIÓN INICIAL (valores por defecto del proyecto) ===
horarios_config = {
    "relay1": {
        "nombre": "Motor AC",
        "habilitado": True,
        "hora_inicio": 17.0,          # 17:00 (5:00 PM)
        "hora_fin": 17.0 + (20/60),   # 17:20 (5:20 PM)
        "temp_min": 15.0,
        "temp_max": 18.0,
        "hum_min": 80.0,
        "hum_max": 90.0
    },
    "relay2": {
        "nombre": "Foco 1",
        "habilitado": True,
        "hora_inicio": 17.5,          # 17:30 (5:30 PM)
        "hora_fin": 18.0,             # 18:00 (6:00 PM)
        "temp_min": 18.0,
        "temp_max": 20.0,
        "hum_min": 90.0,
        "hum_max": 100.0
    },
    "relay3": {
        "nombre": "Foco 2",
        "habilitado": True,
        "hora_inicio": 17.5,          # 17:30 (5:30 PM)
        "hora_fin": 18.0,             # 18:00 (6:00 PM)
        "temp_min": 20.0,
        "temp_max": 25.0,
        "hum_min": 80.0,
        "hum_max": 90.0
    },
    "relay4": {
        "nombre": "Reserva",
        "habilitado": False,
        "hora_inicio": 0.0,
        "hora_fin": 23.99,
        "temp_min": 0.0,
        "temp_max": 50.0,
        "hum_min": 0.0,
        "hum_max": 100.0
    }
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

def cargar_horarios():
    global horarios_config
    try:
        if os.path.exists(HORARIOS_FILE):
            with open(HORARIOS_FILE, 'r') as f:
                horarios_cargados = json.load(f)
                horarios_config.update(horarios_cargados)
            print(f"✅ Horarios cargados desde {HORARIOS_FILE}")
            registrar_evento("CONFIG", "Horarios cargados exitosamente")
        else:
            guardar_horarios()
    except Exception as e:
        print(f"⚠️ Error cargando horarios: {e}")

def guardar_horarios():
    try:
        with open(HORARIOS_FILE, 'w') as f:
            json.dump(horarios_config, f, indent=4)
        print(f"✅ Horarios guardados en {HORARIOS_FILE}")
        registrar_evento("CONFIG", "Horarios guardados exitosamente")
        return True
    except Exception as e:
        print(f"❌ Error guardando horarios: {e}")
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
    # Usamos la nueva función para que el log salga con hora Perú
    fecha_peru = obtener_hora_actual_peru().strftime("%Y-%m-%d %H:%M:%S")
    
    evento = {
        "timestamp": fecha_peru,
        "tipo": tipo,
        "mensaje": mensaje
    }
    log_eventos.append(evento)
    print(f"[{tipo}] {mensaje}")

# ========== RED NEURONAL MLP MEJORADA ==========

class RedNeuronalMLP:
    def __init__(self):
        self.modelo = None
        self.scaler = MinMaxScaler()
        self.entrenado = False
        self.metricas = {
            'accuracy': 0.0,
            'samples_trained': 0,
            'architecture': 'Entrada[3] → [32-16-8] → Salida[4]',
            'loss': 0.0,
            'iterations': 0,
            'training_time': 0.0
        }
        self.inicializar_modelo()
    
    def inicializar_modelo(self):
        self.modelo = MLPClassifier(
            hidden_layer_sizes=(32, 16, 8),
            activation='relu',
            solver='adam',
            max_iter=3000,
            random_state=42,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=100,
            verbose=False,
            alpha=0.0001
        )
        registrar_evento("MLP", "Red neuronal MLP inicializada: [3] → [32-16-8] → [4]")
    
    def generar_dataset_entrenamiento(self):
        X = []
        y = []
        
        print("\n" + "="*70)
        print("📊 GENERANDO DATASET MULTI-LABEL (Sin Conflictos)")
        print("="*70)
        
        # Extraer rangos de todos los relés habilitados
        relays_habilitados = []
        for relay_key in ['relay1', 'relay2', 'relay3', 'relay4']:
            if horarios_config[relay_key]['habilitado']:
                config = horarios_config[relay_key]
                relay_index = int(relay_key[-1]) - 1
                
                h_ini_str = f"{int(config['hora_inicio']):02d}:{int((config['hora_inicio']%1)*60):02d}"
                h_fin_str = f"{int(config['hora_fin']):02d}:{int((config['hora_fin']%1)*60):02d}"
                
                print(f"\n✅ {config['nombre']} (Relay {relay_index + 1})")
                print(f"   📅 {h_ini_str} - {h_fin_str}")
                print(f"   🌡️  {config['temp_min']:.1f}°C - {config['temp_max']:.1f}°C")
                print(f"   💧 {config['hum_min']:.1f}% - {config['hum_max']:.1f}%")
                
                relays_habilitados.append({
                    'index': relay_index,
                    'config': config
                })
        
        if not relays_habilitados:
            print("\n⚠️  ADVERTENCIA: No hay relays habilitados!")
            # Dataset mínimo de seguridad
            for _ in range(200):
                X.append([np.random.uniform(15, 25), 
                         np.random.uniform(60, 90), 
                         np.random.uniform(0, 24)])
                y.append([0, 0, 0, 0])
            
            print("\n📈 Dataset mínimo generado: 200 muestras OFF")
            return np.array(X), np.array(y)
        
        # ========== FASE 1: MUESTRAS POSITIVAS (dentro de rangos) ==========
        print(f"\n🟢 FASE 1: Generando muestras DENTRO de rangos configurados")
        
        num_muestras_positivas = 800
        
        for _ in range(num_muestras_positivas):
            # Seleccionar un relay al azar para generar la muestra
            relay_ref = np.random.choice(relays_habilitados)
            config_ref = relay_ref['config']
            
            # Generar valores dentro del rango del relay seleccionado
            temp = np.random.uniform(config_ref['temp_min'], config_ref['temp_max'])
            hum = np.random.uniform(config_ref['hum_min'], config_ref['hum_max'])
            hora = np.random.uniform(config_ref['hora_inicio'], config_ref['hora_fin'])
            
            # Añadir ruido pequeño (±5% del rango)
            temp += np.random.uniform(-0.5, 0.5)
            hum += np.random.uniform(-2, 2)
            hora += np.random.uniform(-0.05, 0.05)
            
            # Mantener dentro de límites físicos
            temp = np.clip(temp, -40, 80)
            hum = np.clip(hum, 0, 100)
            hora = hora % 24
            
            # ✅ EVALUAR TODOS LOS RELÉS SIMULTÁNEAMENTE
            output = [0, 0, 0, 0]
            for relay in relays_habilitados:
                idx = relay['index']
                cfg = relay['config']
                
                # Verificar TODAS las condiciones
                temp_ok = cfg['temp_min'] <= temp <= cfg['temp_max']
                hum_ok = cfg['hum_min'] <= hum <= cfg['hum_max']
                
                # Manejo de horarios que cruzan medianoche
                if cfg['hora_inicio'] <= cfg['hora_fin']:
                    hora_ok = cfg['hora_inicio'] <= hora <= cfg['hora_fin']
                else:  # Cruce de medianoche (ej: 23:00 - 01:00)
                    hora_ok = hora >= cfg['hora_inicio'] or hora <= cfg['hora_fin']
                
                if temp_ok and hum_ok and hora_ok:
                    output[idx] = 1
            
            X.append([temp, hum, hora])
            y.append(output)
        
        print(f"   ✓ {num_muestras_positivas} muestras generadas con evaluación multi-label")
        
        # ========== FASE 2: MUESTRAS EN BORDES (transiciones) ==========
        print(f"\n🟡 FASE 2: Generando muestras en BORDES de rangos")
        
        num_muestras_borde = 400
        
        for _ in range(num_muestras_borde):
            relay_ref = np.random.choice(relays_habilitados)
            config_ref = relay_ref['config']
            
            # Generar valores en los límites (±10% del rango)
            borde_tipo = np.random.choice(['temp_min', 'temp_max', 'hum_min', 
                                           'hum_max', 'hora_inicio', 'hora_fin'])
            
            if borde_tipo == 'temp_min':
                temp = config_ref['temp_min'] + np.random.uniform(-1, 1)
                hum = np.random.uniform(config_ref['hum_min'], config_ref['hum_max'])
                hora = np.random.uniform(config_ref['hora_inicio'], config_ref['hora_fin'])
            elif borde_tipo == 'temp_max':
                temp = config_ref['temp_max'] + np.random.uniform(-1, 1)
                hum = np.random.uniform(config_ref['hum_min'], config_ref['hum_max'])
                hora = np.random.uniform(config_ref['hora_inicio'], config_ref['hora_fin'])
            elif borde_tipo == 'hum_min':
                temp = np.random.uniform(config_ref['temp_min'], config_ref['temp_max'])
                hum = config_ref['hum_min'] + np.random.uniform(-3, 3)
                hora = np.random.uniform(config_ref['hora_inicio'], config_ref['hora_fin'])
            elif borde_tipo == 'hum_max':
                temp = np.random.uniform(config_ref['temp_min'], config_ref['temp_max'])
                hum = config_ref['hum_max'] + np.random.uniform(-3, 3)
                hora = np.random.uniform(config_ref['hora_inicio'], config_ref['hora_fin'])
            elif borde_tipo == 'hora_inicio':
                temp = np.random.uniform(config_ref['temp_min'], config_ref['temp_max'])
                hum = np.random.uniform(config_ref['hum_min'], config_ref['hum_max'])
                hora = config_ref['hora_inicio'] + np.random.uniform(-0.1, 0.1)
            else:  # hora_fin
                temp = np.random.uniform(config_ref['temp_min'], config_ref['temp_max'])
                hum = np.random.uniform(config_ref['hum_min'], config_ref['hum_max'])
                hora = config_ref['hora_fin'] + np.random.uniform(-0.1, 0.1)
            
            temp = np.clip(temp, -40, 80)
            hum = np.clip(hum, 0, 100)
            hora = hora % 24
            
            # Evaluar todos los relés
            output = [0, 0, 0, 0]
            for relay in relays_habilitados:
                idx = relay['index']
                cfg = relay['config']
                
                temp_ok = cfg['temp_min'] <= temp <= cfg['temp_max']
                hum_ok = cfg['hum_min'] <= hum <= cfg['hum_max']
                
                if cfg['hora_inicio'] <= cfg['hora_fin']:
                    hora_ok = cfg['hora_inicio'] <= hora <= cfg['hora_fin']
                else:
                    hora_ok = hora >= cfg['hora_inicio'] or hora <= cfg['hora_fin']
                
                if temp_ok and hum_ok and hora_ok:
                    output[idx] = 1
            
            X.append([temp, hum, hora])
            y.append(output)
        
        print(f"   ✓ {num_muestras_borde} muestras en bordes generadas")
        
        # ========== FASE 3: MUESTRAS NEGATIVAS (fuera de rangos) ==========
        print(f"\n🔴 FASE 3: Generando muestras FUERA de rangos (todo OFF)")
        
        num_muestras_negativas = 600
        
        for _ in range(num_muestras_negativas):
            # Estrategia: generar valores que NO coincidan con ningún relay
            estrategia = np.random.choice(['temp_fuera', 'hum_fuera', 'hora_fuera', 'aleatorio_total'])
            
            if estrategia == 'temp_fuera':
                # Temperatura muy fuera de todos los rangos
                temp_min_global = min(r['config']['temp_min'] for r in relays_habilitados)
                temp_max_global = max(r['config']['temp_max'] for r in relays_habilitados)
                
                if np.random.random() < 0.5:
                    temp = np.random.uniform(0, temp_min_global - 2)
                else:
                    temp = np.random.uniform(temp_max_global + 2, 50)
                
                hum = np.random.uniform(0, 100)
                hora = np.random.uniform(0, 24)
                
            elif estrategia == 'hum_fuera':
                temp = np.random.uniform(10, 30)
                
                hum_min_global = min(r['config']['hum_min'] for r in relays_habilitados)
                hum_max_global = max(r['config']['hum_max'] for r in relays_habilitados)
                
                if np.random.random() < 0.5:
                    hum = np.random.uniform(0, max(0, hum_min_global - 5))
                else:
                    hum = np.random.uniform(min(100, hum_max_global + 5), 100)
                
                hora = np.random.uniform(0, 24)
                
            elif estrategia == 'hora_fuera':
                temp = np.random.uniform(10, 30)
                hum = np.random.uniform(40, 90)
                
                # Hora fuera de todos los rangos activos
                horas_activas = []
                for relay in relays_habilitados:
                    cfg = relay['config']
                    if cfg['hora_inicio'] <= cfg['hora_fin']:
                        horas_activas.extend([cfg['hora_inicio'], cfg['hora_fin']])
                
                if horas_activas:
                    hora_min = min(horas_activas)
                    hora_max = max(horas_activas)
                    
                    if hora_min > 1 and np.random.random() < 0.5:
                        hora = np.random.uniform(0, hora_min - 0.5)
                    elif hora_max < 23:
                        hora = np.random.uniform(hora_max + 0.5, 24)
                    else:
                        hora = np.random.uniform(0, 24)
                else:
                    hora = np.random.uniform(0, 24)
            
            else:  # aleatorio_total
                temp = np.random.uniform(5, 40)
                hum = np.random.uniform(10, 100)
                hora = np.random.uniform(0, 24)
            
            # Evaluar (debe dar todo OFF)
            output = [0, 0, 0, 0]
            for relay in relays_habilitados:
                idx = relay['index']
                cfg = relay['config']
                
                temp_ok = cfg['temp_min'] <= temp <= cfg['temp_max']
                hum_ok = cfg['hum_min'] <= hum <= cfg['hum_max']
                
                if cfg['hora_inicio'] <= cfg['hora_fin']:
                    hora_ok = cfg['hora_inicio'] <= hora <= cfg['hora_fin']
                else:
                    hora_ok = hora >= cfg['hora_inicio'] or hora <= cfg['hora_fin']
                
                if temp_ok and hum_ok and hora_ok:
                    output[idx] = 1
            
            X.append([temp, hum, hora])
            y.append(output)
        
        print(f"   ✓ {num_muestras_negativas} muestras negativas generadas")
        
        # ========== ESTADÍSTICAS FINALES ==========
        X = np.array(X)
        y = np.array(y)
        
        total_muestras = len(X)
        muestras_off = np.sum(np.all(y == 0, axis=1))
        muestras_on = total_muestras - muestras_off
        
        # Contar activaciones simultáneas
        activaciones_simultaneas = np.sum(y, axis=1)
        max_simultaneas = int(np.max(activaciones_simultaneas))
        
        print("\n" + "="*70)
        print("📈 RESUMEN DEL DATASET MULTI-LABEL:")
        print("="*70)
        print(f"   Total muestras: {total_muestras}")
        print(f"   Muestras con al menos 1 ON: {muestras_on}")
        print(f"   Muestras todo OFF: {muestras_off}")
        print(f"   Relays habilitados: {len(relays_habilitados)}")
        print(f"   Max activaciones simultáneas: {max_simultaneas}")
        print(f"\n   Distribución por Relay:")
        for i in range(4):
            activaciones = np.sum(y[:, i])
            porcentaje = (activaciones / total_muestras) * 100
            print(f"      Relay {i+1}: {int(activaciones)} activaciones ({porcentaje:.1f}%)")
        print("="*70 + "\n")
        
        return X, y

    def entrenar(self):
        try:
            print("\n" + "🧠"*35)
            print("   ENTRENAMIENTO DE RED NEURONAL MLP")
            print("🧠"*35 + "\n")
            
            X, y = self.generar_dataset_entrenamiento()
            
            if len(X) == 0:
                raise Exception("Dataset vacío. Configura al menos un relay.")
            
            # Escalar datos
            X_scaled = self.scaler.fit_transform(X)
            
            print("⚙️  INICIANDO ENTRENAMIENTO...")
            print(f"   Arquitectura: {self.metricas['architecture']}")
            print(f"   Optimizador: Adam (learning_rate=0.001)")
            print(f"   Max iteraciones: 3000")
            print(f"   Early stopping: Activado\n")
            
            inicio = time.time()
            self.modelo.fit(X_scaled, y)
            tiempo_entrenamiento = time.time() - inicio
            
            # Evaluar accuracy
            y_pred = self.modelo.predict(X_scaled)
            accuracy = np.mean([np.array_equal(a, b) for a, b in zip(y_pred, y)]) * 100
            
            self.metricas = {
                'accuracy': round(accuracy, 2),
                'samples_trained': len(X),
                'training_time': round(tiempo_entrenamiento, 3),
                'iterations': self.modelo.n_iter_,
                'architecture': 'Entrada[3] → [32-16-8] → Salida[4]',
                'loss': round(self.modelo.loss_, 6) if hasattr(self.modelo, 'loss_') else 0.0
            }
            
            self.entrenado = True
            
            print("="*70)
            print("✅ ENTRENAMIENTO COMPLETADO")
            print("="*70)
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Tiempo: {tiempo_entrenamiento:.3f} segundos")
            print(f"   Iteraciones: {self.modelo.n_iter_}")
            print(f"   Loss final: {self.metricas['loss']:.6f}")
            print("="*70 + "\n")
            
            registrar_evento("MLP", f"✅ Entrenamiento exitoso: Accuracy={accuracy:.2f}%, Muestras={len(X)}")
            self.guardar_modelo()
            
            return {
                'success': True,
                'mensaje': 'Red neuronal MLP entrenada exitosamente',
                'metricas': self.metricas
            }
            
        except Exception as e:
            print(f"\n❌ ERROR EN ENTRENAMIENTO: {e}")
            import traceback
            traceback.print_exc()
            registrar_evento("ERROR", f"Error entrenando MLP: {str(e)}")
            return {
                'success': False,
                'mensaje': f'Error en entrenamiento: {str(e)}'
            }
    
    def predecir(self, temperatura, humedad, hora):
        """✅ Predicción mejorada con logging detallado"""
        if not self.entrenado:
            registrar_evento("WARNING", "MLP no entrenado, retornando estado seguro")
            return {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
        
        try:
            X = np.array([[temperatura, humedad, hora]])
            X_scaled = self.scaler.transform(X)
            
            prediccion_raw = self.modelo.predict(X_scaled)[0]
            
            # Umbral de confianza: 0.5
            UMBRAL = 0.5
            prediccion = [1 if x >= UMBRAL else 0 for x in prediccion_raw]
            
            resultado = {
                'relay1': bool(prediccion[0]),
                'relay2': bool(prediccion[1]),
                'relay3': bool(prediccion[2]),
                'relay4': bool(prediccion[3])
            }
            
            if config_umbrales.get('modo_debug', False):
                h = int(hora)
                m = int((hora % 1) * 60)
                print(f"\n🤖 MLP PREDICCIÓN [{h:02d}:{m:02d}]")
                print(f"   📊 Entrada: T={temperatura:.1f}°C  H={humedad:.1f}%  Hora={hora:.2f}")
                print(f"   🧠 Salida: [{prediccion_raw[0]:.3f}, {prediccion_raw[1]:.3f}, {prediccion_raw[2]:.3f}, {prediccion_raw[3]:.3f}]")
                print(f"   ✅ Decisión: R1={resultado['relay1']}  R2={resultado['relay2']}  R3={resultado['relay3']}  R4={resultado['relay4']}")
            
            return resultado
            
        except Exception as e:
            print(f"❌ Error en predicción MLP: {e}")
            registrar_evento("ERROR", f"Error en predicción: {str(e)}")
            return {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
    
    def obtener_estado(self):
        return {
            'entrenado': self.entrenado,
            'metricas': self.metricas,
            'arquitectura': {
                'entradas': ['Temperatura (°C)', 'Humedad (%)', 'Hora (24h decimal)'],
                'capas_ocultas': [32, 16, 8],
                'salidas': ['Relay 1 (Motor)', 'Relay 2 (Foco 1)', 'Relay 3 (Foco 2)', 'Relay 4 (Reserva)'],
                'activacion': 'ReLU',
                'optimizador': 'Adam',
                'total_parametros': self.calcular_parametros()
            },
            'horarios_actuales': horarios_config
        }
    
    def calcular_parametros(self):
        if not self.entrenado:
            return 0
        capas = [3] + [32, 16, 8] + [4]
        total = 0
        for i in range(len(capas) - 1):
            total += (capas[i] * capas[i+1]) + capas[i+1]
        return total
    
    def guardar_modelo(self, ruta='modelo_mlp.pkl'):
        if self.entrenado:
            try:
                with open(ruta, 'wb') as f:
                    pickle.dump({
                        'modelo': self.modelo,
                        'scaler': self.scaler,
                        'metricas': self.metricas,
                        'horarios': horarios_config,
                        'version': '5.0'
                    }, f)
                registrar_evento("MLP", f"Modelo guardado en {ruta}")
            except Exception as e:
                print(f"Error guardando modelo: {e}")
    
    def cargar_modelo(self, ruta='modelo_mlp.pkl'):
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

mlp = RedNeuronalMLP()

# ========== FUNCIONES AUXILIARES ==========

# ========== CORRECCIÓN DE HORA (PERÚ UTC-5) ==========
def obtener_hora_actual_peru():
    """Devuelve la fecha y hora exacta ajustada a Perú"""
    # Obtenemos la hora UTC del servidor y restamos 5 horas
    return datetime.datetime.utcnow() - datetime.timedelta(hours=5)

def obtener_hora_decimal():
    """Devuelve la hora en formato decimal (ej: 17.5) ajustada a Perú"""
    ahora = obtener_hora_actual_peru()
    return ahora.hour + ahora.minute / 60.0

def verificar_timeout():
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
    while True:
        time.sleep(300)
        ahora = time.time()
        sesiones_a_eliminar = [
            sid for sid, ts in sesiones_admin.items() 
            if ahora - ts > 3600
        ]
        for session_id in sesiones_a_eliminar:
            del sesiones_admin[session_id]

threading.Thread(target=verificar_timeout, daemon=True).start()
threading.Thread(target=limpiar_sesiones_expiradas, daemon=True).start()

def verificar_admin_autenticado():
    session_id = session.get('admin_session_id')
    if not session_id or session_id not in sesiones_admin:
        return False
    sesiones_admin[session_id] = time.time()
    return True

# ========== ENDPOINTS DE HORARIOS ==========

@app.route('/api/horarios', methods=['GET'])
def obtener_horarios():
    """Obtener configuración actual de horarios"""
    return jsonify(horarios_config)

@app.route('/api/horarios', methods=['POST'])
def actualizar_horarios():
    """✅ Actualizar configuración de horarios (NO requiere re-entrenar inmediatamente)"""
    if not verificar_admin_autenticado():
        return jsonify({"error": "Autenticación requerida", "requiere_auth": True}), 403
    
    try:
        data = request.json
        relay_key = data.get('relay')
        
        if relay_key not in horarios_config:
            return jsonify({"error": "Relay inválido"}), 400
        
        # Actualizar configuración
        if 'habilitado' in data:
            horarios_config[relay_key]['habilitado'] = data['habilitado']
        if 'hora_inicio' in data:
            horarios_config[relay_key]['hora_inicio'] = float(data['hora_inicio'])
        if 'hora_fin' in data:
            horarios_config[relay_key]['hora_fin'] = float(data['hora_fin'])
        if 'temp_min' in data:
            horarios_config[relay_key]['temp_min'] = float(data['temp_min'])
        if 'temp_max' in data:
            horarios_config[relay_key]['temp_max'] = float(data['temp_max'])
        if 'hum_min' in data:
            horarios_config[relay_key]['hum_min'] = float(data['hum_min'])
        if 'hum_max' in data:
            horarios_config[relay_key]['hum_max'] = float(data['hum_max'])
        if 'nombre' in data:
            horarios_config[relay_key]['nombre'] = data['nombre']
        
        # Guardar cambios
        guardar_horarios()
        
        h_ini = horarios_config[relay_key]['hora_inicio']
        h_fin = horarios_config[relay_key]['hora_fin']
        registrar_evento("CONFIG", f"✅ {relay_key} actualizado: {int(h_ini):02d}:{int((h_ini%1)*60):02d} - {int(h_fin):02d}:{int((h_fin%1)*60):02d}")
        
        return jsonify({
            "ok": True,
            "mensaje": "Configuración guardada. Recuerda re-entrenar el MLP.",
            "horarios": horarios_config
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== ENDPOINTS DE AUTENTICACIÓN ==========

@app.route('/api/auth/login', methods=['POST'])
def login_admin():
    try:
        data = request.json
        usuario = data.get('usuario', '')
        password = data.get('password', '')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        if usuario == ADMIN_USER and password_hash == ADMIN_PASSWORD_HASH:
            session_id = secrets.token_hex(32)
            session['admin_session_id'] = session_id
            sesiones_admin[session_id] = time.time()
            registrar_evento("AUTH", f"✅ Login exitoso - Usuario: {usuario}")
            return jsonify({"ok": True, "mensaje": "Autenticación exitosa"})
        else:
            registrar_evento("AUTH", f"❌ Login fallido - Usuario: {usuario}")
            return jsonify({"ok": False, "mensaje": "Credenciales incorrectas"}), 401
    except Exception as e:
        return jsonify({"ok": False, "mensaje": str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout_admin():
    session_id = session.get('admin_session_id')
    if session_id and session_id in sesiones_admin:
        del sesiones_admin[session_id]
        registrar_evento("AUTH", "Sesión cerrada")
    session.pop('admin_session_id', None)
    return jsonify({"ok": True})

@app.route('/api/auth/verificar', methods=['GET'])
def verificar_sesion():
    return jsonify({"autenticado": verificar_admin_autenticado()})

# ========== ENDPOINT PRINCIPAL DE TELEMETRÍA ==========

@app.route('/api/telemetria', methods=['POST'])
def recibir_datos():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No data received"}), 400
        
        temp = float(data.get('t', 20))
        hum = float(data.get('h', 60))
        hora_decimal = obtener_hora_decimal()
        
        if not (-40 <= temp <= 80):
            temp = 20.0
        if not (0 <= hum <= 100):
            hum = 60.0
        
        with estado_lock:
            if temp > estado_sistema['temp_max_sesion']:
                estado_sistema['temp_max_sesion'] = temp
            if temp < estado_sistema['temp_min_sesion']:
                estado_sistema['temp_min_sesion'] = temp
            if hum > estado_sistema['hum_max_sesion']:
                estado_sistema['hum_max_sesion'] = hum
            if hum < estado_sistema['hum_min_sesion']:
                estado_sistema['hum_min_sesion'] = hum
            
            modo_actual = estado_sistema['modo']
            
            if modo_actual == "AUTO":
                if config_umbrales.get('usar_mlp', True) and mlp.entrenado:
                    decision = mlp.predecir(temp, hum, hora_decimal)
                    estado_sistema['mlp_activo'] = True
                else:
                    decision = {'relay1': False, 'relay2': False, 'relay3': False, 'relay4': False}
                    estado_sistema['mlp_activo'] = False
                
                if decision['relay1'] and not estado_sistema['relay1']:
                    estado_sistema['ciclos_motor'] += 1
                
                estado_sistema.update(decision)
                estado_sistema['alertas_activas'] = []
                
            else:
                decision = {
                    'relay1': estado_sistema['manual_relay1'],
                    'relay2': estado_sistema['manual_relay2'],
                    'relay3': estado_sistema['manual_relay3'],
                    'relay4': estado_sistema['manual_relay4']
                }
                estado_sistema.update(decision)
                estado_sistema['alertas_activas'] = ["🎮 Modo MANUAL activo"]
            
            estado_sistema['temperatura'] = temp
            estado_sistema['humedad'] = hum
            estado_sistema['hora_actual'] = hora_decimal
            estado_sistema['ultima_actualizacion'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            estado_sistema['conectado'] = True
            estado_sistema['mensaje'] = "Sistema operando correctamente"
            
            historial.append({
                "timestamp": estado_sistema['ultima_actualizacion'],
                "temperatura": temp,
                "humedad": hum,
                "hora": hora_decimal,
                **decision,
                "modo": modo_actual
            })
        
        h = int(hora_decimal)
        m = int((hora_decimal % 1) * 60)
        print(f"✓ [{modo_actual}][{h:02d}:{m:02d}] T:{temp:.1f}°C H:{hum:.1f}% → R1:{decision['relay1']} R2:{decision['relay2']} R3:{decision['relay3']} R4:{decision['relay4']}")
        
        return jsonify(decision), 200
        
    except Exception as e:
        print(f"🔥 ERROR en /api/telemetria: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ========== DEMÁS ENDPOINTS ==========

@app.route('/api/estado', methods=['GET'])
def obtener_estado():
    with estado_lock:
        return jsonify(estado_sistema)

@app.route('/api/historial', methods=['GET'])
def obtener_historial():
    return jsonify({"datos": list(historial)})

@app.route('/api/log', methods=['GET'])
def obtener_log():
    return jsonify({"eventos": list(log_eventos)})

@app.route('/api/kpis', methods=['GET'])
def obtener_kpis():
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
    with estado_lock:
        if len(historial) == 0:
            return jsonify({"labels": [], "temperatura": [], "humedad": []})
        
        datos = list(historial)[-100:]
        labels = [d['timestamp'].split(' ')[1] for d in datos]
        temperaturas = [d['temperatura'] for d in datos]
        humedades = [d['humedad'] for d in datos]
        
        return jsonify({
            "labels": labels,
            "temperatura": temperaturas,
            "humedad": humedades
        })

@app.route('/api/modo', methods=['POST'])
def cambiar_modo():
    try:
        data = request.json
        nuevo_modo = data.get('modo', '').upper()
        
        if nuevo_modo not in ['AUTO', 'MANUAL']:
            return jsonify({"error": "Modo inválido (debe ser AUTO o MANUAL)"}), 400
        
        if nuevo_modo == 'MANUAL' and not verificar_admin_autenticado():
            return jsonify({"ok": False, "error": "Autenticación requerida", "requiere_auth": True}), 403
        
        with estado_lock:
            modo_anterior = estado_sistema['modo']
            estado_sistema['modo'] = nuevo_modo
            
            if nuevo_modo == 'MANUAL':
                estado_sistema['manual_relay1'] = estado_sistema['relay1']
                estado_sistema['manual_relay2'] = estado_sistema['relay2']
                estado_sistema['manual_relay3'] = estado_sistema['relay3']
                estado_sistema['manual_relay4'] = estado_sistema['relay4']
        
        registrar_evento("MODO", f"Cambiado de {modo_anterior} a {nuevo_modo}")
        return jsonify({"ok": True, "modo": nuevo_modo})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/control', methods=['POST'])
def control_manual():
    if not verificar_admin_autenticado():
        return jsonify({"error": "Autenticación requerida", "requiere_auth": True}), 403
    
    try:
        data = request.json
        relay = data.get('relay')
        estado = data.get('estado', False)
        
        if relay not in ['relay1', 'relay2', 'relay3', 'relay4']:
            return jsonify({"error": "Relay inválido"}), 400
        
        with estado_lock:
            if estado_sistema['modo'] != 'MANUAL':
                return jsonify({"error": "Control manual solo disponible en modo MANUAL"}), 403
            
            estado_sistema[f'manual_{relay}'] = estado
            estado_sistema[relay] = estado
        
        registrar_evento("CONTROL", f"{relay} → {'ON' if estado else 'OFF'} (Manual)")
        return jsonify({"ok": True, "relay": relay, "estado": estado})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mlp/entrenar', methods=['POST'])
def entrenar_mlp():
    if not verificar_admin_autenticado():
        return jsonify({"error": "Autenticación requerida", "requiere_auth": True}), 403
    
    print("\n" + "🔄"*35)
    print("   RE-ENTRENAMIENTO SOLICITADO DESDE DASHBOARD")
    print("🔄"*35)
    
    resultado = mlp.entrenar()
    return jsonify(resultado)

@app.route('/api/mlp/estado', methods=['GET'])
def obtener_estado_mlp():
    return jsonify(mlp.obtener_estado())

@app.route('/api/reporte/pdf', methods=['GET'])
def generar_reporte_pdf():
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
        
        elementos.append(Paragraph("🧠 SISTEMA MLP - CONTROL INTELIGENTE", estilo_titulo))
        elementos.append(Paragraph("Reporte de Monitoreo con Red Neuronal", estilos['Heading2']))
        elementos.append(Spacer(1, 0.3*inch))
        
        fecha_reporte = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        elementos.append(Paragraph(f"<b>Fecha del Reporte:</b> {fecha_reporte}", estilos['Normal']))
        elementos.append(Spacer(1, 0.2*inch))
        
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

@app.route('/')
def home():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

def inicializar_sistema():
    print("\n" + "="*70)
    print("🧠 SISTEMA MLP - CONTROL INTELIGENTE CON RED NEURONAL")
    print("="*70)
    
    cargar_configuracion()
    cargar_horarios()
    
    print("\n📋 CONFIGURACIÓN DE HORARIOS ACTUAL:")
    for relay_key in ['relay1', 'relay2', 'relay3', 'relay4']:
        config = horarios_config[relay_key]
        h_ini = config['hora_inicio']
        h_fin = config['hora_fin']
        estado_txt = "✅ ACTIVO" if config['habilitado'] else "⚠️  DESHABILITADO"
        print(f"   {estado_txt} - {config['nombre']}: {int(h_ini):02d}:{int((h_ini%1)*60):02d} - {int(h_fin):02d}:{int((h_fin%1)*60):02d}")
    
    if not mlp.cargar_modelo():
        print("\n📦 Modelo MLP no encontrado. Entrenando desde cero...")
        resultado = mlp.entrenar()
        if resultado['success']:
            print(f"\n✅ Modelo entrenado exitosamente")
        else:
            print(f"\n❌ Error en entrenamiento: {resultado['mensaje']}")
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
    print(f"⚙️  Modos: AUTO (MLP) + MANUAL")
    print(f"⏰ Horarios: CONFIGURABLES desde Dashboard")
    print(f"🔑 Admin: usuario='admin' / password='admin123'")
    print("="*70)
    print("\n🚀 Sistema listo. Esperando conexiones...\n")
    
    registrar_evento("SISTEMA", "Servidor Flask iniciado correctamente")

if __name__ == '__main__':
    inicializar_sistema()
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True, use_reloader=False)
